// Unified SIFT100K mixed-workload pilot.  This is project adapter code; the
// included SIEVE tree is pristine and is never modified.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "partitioned_hnsw.h"
#include "sieve_packed_kernel.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using Hit = std::pair<uint32_t, uint32_t>;  // squared distance, ID
constexpr uint32_t N = 100000, D = 128, K = 10;
constexpr uint32_t CAT_A = 32, CAT_B = 16, BINS = 16;
constexpr uint32_t B_BASE = CAT_A;
constexpr uint32_t BIN_BASE = CAT_A + CAT_B;
constexpr uint32_t PAIR_BASE = CAT_A + CAT_B + BINS;
constexpr uint32_t NLABELS = PAIR_BASE + CAT_A * CAT_B;

enum class Family : uint32_t { Equality = 0, Conjunction = 1, Range = 2, Dnf = 3 };
enum class Arm { Joint, Staged, Sieve };

struct Spec {
  Family family{};
  uint16_t a = 0, b = 0, c = 0, d = 0, lo = 0, hi = 0;
};
struct Query { Spec spec; std::array<uint8_t, D> vector{}; };
struct Result { std::array<uint32_t, K> ids{}; };
struct Packed { std::vector<uint32_t> ids; std::vector<uint8_t> vectors; };

[[noreturn]] void fail(const std::string& text) { throw std::runtime_error(text); }
void require(bool ok, const std::string& text) { if (!ok) fail(text); }

template<class T> std::vector<T> read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  require(bool(in), "open failed: " + path);
  const auto bytes = in.tellg();
  require(bytes >= 0 && uint64_t(bytes) % sizeof(T) == 0, "extent failed: " + path);
  std::vector<T> out(size_t(bytes) / sizeof(T));
  in.seekg(0); in.read(reinterpret_cast<char*>(out.data()), bytes);
  require(bool(in), "read failed: " + path); return out;
}

uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
uint32_t pair_id(uint32_t a, uint32_t b) { return a * CAT_B + b; }
uint32_t pair_label(uint32_t a, uint32_t b) { return PAIR_BASE + pair_id(a, b); }
const char* family_name(Family f) {
  static const char* names[] = {"equality", "conjunction", "range", "dnf"};
  return names[uint32_t(f)];
}

struct Data {
  std::vector<uint8_t> base;
  std::vector<uint8_t> query_rows;
  std::vector<uint8_t> cat_a, cat_b, bin;
  std::array<std::vector<uint32_t>, CAT_A> a_post;
  std::array<std::vector<uint32_t>, CAT_B> b_post;
  std::array<std::vector<uint32_t>, BINS> bin_post;
  std::array<std::vector<uint32_t>, CAT_A * CAT_B> pair_post;
  std::unordered_map<std::string, std::vector<uint32_t>> dyadic_post;
  std::vector<std::vector<uint32_t>> expanded_rows;
  std::vector<uint32_t> valid_pairs;
};

std::string range_key(uint32_t width, uint32_t start) {
  return "R" + std::to_string(width) + "_" + std::to_string(start);
}

Data load_data(const std::string& base_path, const std::string& meta_path,
               const std::string& query_path, const std::string& attr_path) {
  Data out;
  auto base_file = read_all<uint8_t>(base_path);
  require(base_file.size() == 8 + size_t(N) * D, "base byte extent differs");
  uint32_t bn = 0, bd = 0; std::memcpy(&bn, base_file.data(), 4);
  std::memcpy(&bd, base_file.data() + 4, 4);
  require(bn == N && bd == D, "base header differs");
  out.base.assign(base_file.begin() + 8, base_file.end());

  auto query_file = read_all<uint8_t>(query_path);
  require(query_file.size() == size_t(10000) * (D + 1) * 4, "query extent differs");
  out.query_rows.resize(size_t(10000) * D);
  for (uint32_t row = 0; row < 10000; ++row) {
    int32_t dim = 0; std::memcpy(&dim, query_file.data() + size_t(row) * (D + 1) * 4, 4);
    require(dim == int32_t(D), "query dimension differs");
    for (uint32_t j = 0; j < D; ++j) {
      float value = 0; std::memcpy(&value, query_file.data() + (size_t(row) * (D + 1) + j + 1) * 4, 4);
      require(std::isfinite(value) && value >= 0 && value <= 255 && value == std::round(value), "query uint8 contract differs");
      out.query_rows[size_t(row) * D + j] = uint8_t(value);
    }
  }

  auto meta = read_all<uint8_t>(meta_path);
  require(meta.size() == 24 + size_t(N + 1) * 8 + size_t(N) * 4 * 4, "metadata extent differs");
  uint64_t mn=0, ml=0, mz=0; std::memcpy(&mn, meta.data(),8); std::memcpy(&ml,meta.data()+8,8); std::memcpy(&mz,meta.data()+16,8);
  require(mn==N && ml==16 && mz==uint64_t(N)*4, "metadata header differs");
  const size_t moff = 24 + size_t(N + 1) * 8;
  for (uint32_t row=0; row<=N; ++row) { uint64_t v=0; std::memcpy(&v,meta.data()+24+size_t(row)*8,8); require(v==uint64_t(row)*4,"metadata offsets differ"); }
  auto attrs = read_all<int32_t>(attr_path);
  require(attrs.size() == 10000000, "attribute extent differs");
  out.cat_a.resize(N); out.cat_b.resize(N); out.bin.resize(N); out.expanded_rows.resize(N);
  for (uint32_t row=0; row<N; ++row) {
    uint32_t tags[4]; std::memcpy(tags, meta.data()+moff+size_t(row)*16,16);
    require(tags[0]<tags[1] && tags[1]<tags[2] && tags[2]<tags[3] && tags[3]<16,"metadata tags differ");
    uint64_t key = uint64_t(tags[0]) | (uint64_t(tags[1])<<8) | (uint64_t(tags[2])<<16) | (uint64_t(tags[3])<<24);
    const uint32_t a = mix64(key) % CAT_A;
    const uint32_t b = mix64(key ^ 0x4f1bbcdc6765d5d9ULL) % CAT_B;
    require(attrs[row] >= 0 && attrs[row] < 5000, "attribute domain differs");
    const uint32_t bin = std::min<uint32_t>(uint32_t(attrs[row]) * BINS / 5000, BINS-1);
    out.cat_a[row]=a; out.cat_b[row]=b; out.bin[row]=bin;
    out.a_post[a].push_back(row); out.b_post[b].push_back(row);
    out.bin_post[bin].push_back(row); out.pair_post[pair_id(a,b)].push_back(row);
    out.expanded_rows[row] = {a, B_BASE+b, BIN_BASE+bin, pair_label(a,b)};
  }
  for (uint32_t p=0;p<CAT_A*CAT_B;++p) if (out.pair_post[p].size() >= K) out.valid_pairs.push_back(p);
  for (uint32_t width : {1U, 2U, 4U, 8U}) {
    for (uint32_t start = 0; start < BINS; start += width) {
      std::vector<uint32_t> ids;
      for (uint32_t bin = start; bin < start + width; ++bin)
        ids.insert(ids.end(), out.bin_post[bin].begin(), out.bin_post[bin].end());
      std::sort(ids.begin(), ids.end());
      out.dyadic_post.emplace(range_key(width, start), std::move(ids));
    }
  }
  require(out.valid_pairs.size() >= 480, "too few valid pair supports");
  std::cout << "DATA_READY n="<<N<<" labels="<<NLABELS<<" valid_pairs="<<out.valid_pairs.size()<<std::endl;
  return out;
}

std::vector<uint32_t> zipf_order(const std::vector<std::vector<uint32_t>>& postings) {
  std::vector<uint32_t> ids(postings.size()); std::iota(ids.begin(),ids.end(),0);
  std::sort(ids.begin(),ids.end(),[&](uint32_t x,uint32_t y){ if(postings[x].size()!=postings[y].size()) return postings[x].size()>postings[y].size(); return x<y;});
  return ids;
}

Spec make_spec(uint64_t index, bool heldout, const Data& data) {
  const Family f = Family(index % 4); const uint64_t slot=index/4;
  const uint64_t r=mix64(slot+(heldout?0x68656c646f7574ULL:0x646576656c6f70ULL));
  Spec s; s.family=f;
  if(f==Family::Equality){ s.a=uint16_t((r + (r>>11)) % 12); }
  else if(f==Family::Conjunction){ const uint32_t p=data.valid_pairs[(r%64)%data.valid_pairs.size()]; s.a=p/CAT_B;s.b=p%CAT_B; }
  else if(f==Family::Range){ const uint32_t width[3]={1,2,4}; const uint32_t w=width[(r>>8)%3]; s.lo=uint16_t((r>>16)%(BINS-w+1));s.hi=uint16_t(s.lo+w-1); }
  else { const uint32_t p=data.valid_pairs[(r%24)%data.valid_pairs.size()]; uint32_t q=data.valid_pairs[((r>>13)%24)%data.valid_pairs.size()]; if(q==p)q=data.valid_pairs[(q+1)%data.valid_pairs.size()]; s.a=p/CAT_B;s.b=p%CAT_B;s.c=q/CAT_B;s.d=q%CAT_B; }
  return s;
}

std::vector<Query> make_queries(const Data& data, uint32_t count, bool heldout) {
  std::vector<Query> out(count);
  const uint32_t start=heldout?400:0;
  for(uint32_t i=0;i<count;++i){ out[i].spec=make_spec(i,heldout,data); std::memcpy(out[i].vector.data(),data.query_rows.data()+size_t(start+i)*D,D); }
  return out;
}

const std::vector<uint32_t>& direct_support(const Data& d, const Spec& s) {
  if(s.family==Family::Equality) return d.a_post[s.a];
  if(s.family==Family::Conjunction) return d.pair_post[pair_id(s.a,s.b)];
  fail("direct support called on composed predicate");
}
std::vector<uint32_t> support(const Data& d,const Spec& s){
  if(s.family==Family::Equality||s.family==Family::Conjunction) return direct_support(d,s);
  std::vector<uint32_t> out;
  if(s.family==Family::Range){ for(uint32_t x=s.lo;x<=s.hi;++x) out.insert(out.end(),d.bin_post[x].begin(),d.bin_post[x].end()); }
  else { const auto& x=d.pair_post[pair_id(s.a,s.b)];const auto& y=d.pair_post[pair_id(s.c,s.d)];out.insert(out.end(),x.begin(),x.end());out.insert(out.end(),y.begin(),y.end()); }
  std::sort(out.begin(),out.end()); out.erase(std::unique(out.begin(),out.end()),out.end()); return out;
}
bool matches(const Data& d,const Spec&s,uint32_t id){
  if(id>=N)return false; if(s.family==Family::Equality)return d.cat_a[id]==s.a;
  if(s.family==Family::Conjunction)return d.cat_a[id]==s.a&&d.cat_b[id]==s.b;
  if(s.family==Family::Range)return d.bin[id]>=s.lo&&d.bin[id]<=s.hi;
  return (d.cat_a[id]==s.a&&d.cat_b[id]==s.b)||(d.cat_a[id]==s.c&&d.cat_b[id]==s.d);
}

Result scan(const Data& d,const uint8_t*q,const std::vector<uint32_t>&ids,const std::vector<uint8_t>*packed=nullptr){
  require(ids.size()>=K,"support below top-k"); std::priority_queue<Hit> heap;
  for(size_t i=0;i<ids.size();++i){const uint8_t*v=packed?packed->data()+i*D:d.base.data()+size_t(ids[i])*D;Hit h{sieve_v25_packed::l2_sq_uint8_avx2(q,v),ids[i]};if(heap.size()<K)heap.push(h);else if(h<heap.top()){heap.pop();heap.push(h);}}
  Result r;for(int i=K-1;i>=0;--i){r.ids[size_t(i)]=heap.top().second;heap.pop();}return r;
}
Result merge_results(const Data&d,const uint8_t*q,const std::vector<Result>&parts){
  std::set<uint32_t> ids;for(auto&r:parts)for(uint32_t id:r.ids)ids.insert(id);std::vector<uint32_t> v(ids.begin(),ids.end());return scan(d,q,v);
}

std::string spec_key(const Spec&s){std::ostringstream o;o<<uint32_t(s.family)<<':'<<s.a<<':'<<s.b<<':'<<s.c<<':'<<s.d<<':'<<s.lo<<':'<<s.hi;return o.str();}
std::vector<std::string> fragment_keys(const Spec&s){std::vector<std::string> out;if(s.family==Family::Equality)out.push_back("A"+std::to_string(s.a));else if(s.family==Family::Conjunction)out.push_back("P"+std::to_string(pair_id(s.a,s.b)));else if(s.family==Family::Range){for(uint32_t width:{1U,2U,4U,8U})for(uint32_t start=0;start<BINS;start+=width)if(start>=s.lo&&start+width-1<=s.hi)out.push_back(range_key(width,start));}else{out.push_back("P"+std::to_string(pair_id(s.a,s.b)));out.push_back("P"+std::to_string(pair_id(s.c,s.d)));}return out;}
struct Engine {
  struct Candidate{std::string key;const std::vector<uint32_t>*ids=nullptr;uint64_t uses=0;};
  struct Selection{std::set<std::string> ids,packed;uint64_t bytes=0;};
  struct Cover{double cost=std::numeric_limits<double>::infinity();std::vector<std::string> keys;};
  const Data& d; Arm arm; uint64_t budget,bytes=0;double predicted=0,staged_predicted=0;bool incumbent_used=false;
  std::unordered_map<std::string,Candidate> catalog;std::unordered_map<std::string,Packed> packed;std::unordered_map<std::string,std::vector<uint32_t>> materialized;std::set<std::string> admitted,universe;
  explicit Engine(const Data&data,Arm a,uint64_t b,const std::vector<Query>&dev):d(data),arm(a),budget(b){build(dev);}
  static Packed pack(const Data&d,const std::vector<uint32_t>&ids){Packed p;p.vectors.resize(ids.size()*D);for(size_t i=0;i<ids.size();++i)std::memcpy(p.vectors.data()+i*D,d.base.data()+size_t(ids[i])*D,D);return p;}
  bool is_pair(const std::string&k)const{return k[0]=='P';}
  bool is_range(const std::string&k)const{return k[0]=='R';}
  uint32_t range_width(const std::string&k)const{return std::stoul(k.substr(1,k.find('_')-1));}
  uint64_t account(const Selection&s)const{uint64_t total=0;for(const auto&k:s.ids)total+=catalog.at(k).ids->size()*4ULL;for(const auto&k:s.packed)total+=catalog.at(k).ids->size()*D;return total;}
  double fragment_cost(const std::string&k,const Selection&s)const{const double n=catalog.at(k).ids->size();if(s.packed.count(k))return 3500.0+8.0*n;if(is_pair(k)&&!s.ids.count(k)){uint32_t p=std::stoul(k.substr(1));uint32_t a=p/CAT_B,b=p%CAT_B;return 5000.0+4.0*(d.a_post[a].size()+d.b_post[b].size())+16.0*n;}return 3500.0+16.0*n;}
  Cover cover(uint32_t lo,uint32_t hi,const Selection&s)const{std::array<Cover,BINS+1>dp;dp[hi+1].cost=0;for(int pos=int(hi);pos>=int(lo);--pos){for(uint32_t w:{1U,2U,4U,8U}){if(uint32_t(pos)%w||uint32_t(pos)+w-1>hi)continue;std::string k=range_key(w,uint32_t(pos));if(w>1&&!s.ids.count(k)&&!s.packed.count(k))continue;if(!std::isfinite(dp[uint32_t(pos)+w].cost))continue;double c=fragment_cost(k,s)+dp[uint32_t(pos)+w].cost+(dp[uint32_t(pos)+w].keys.empty()?0.0:1500.0);std::vector<std::string>keys{k};keys.insert(keys.end(),dp[uint32_t(pos)+w].keys.begin(),dp[uint32_t(pos)+w].keys.end());if(c<dp[uint32_t(pos)].cost||(c==dp[uint32_t(pos)].cost&&keys.size()<dp[uint32_t(pos)].keys.size()))dp[uint32_t(pos)]={c,std::move(keys)};}}return dp[lo];}
  double query_cost(const Spec&s,const Selection&sel)const{if(s.family==Family::Equality)return fragment_cost("A"+std::to_string(s.a),sel);if(s.family==Family::Conjunction)return fragment_cost("P"+std::to_string(pair_id(s.a,s.b)),sel);if(s.family==Family::Range)return cover(s.lo,s.hi,sel).cost;return fragment_cost("P"+std::to_string(pair_id(s.a,s.b)),sel)+fragment_cost("P"+std::to_string(pair_id(s.c,s.d)),sel)+1500.0;}
  double total_cost(const std::vector<Query>&dev,const Selection&s)const{double total=0;for(const auto&q:dev)total+=query_cost(q.spec,s);return total;}
  Selection staged_select(const std::vector<Query>&dev)const{
    (void)dev;
    Selection s;
    std::vector<const Candidate*>supports;
    for(const auto&kv:catalog)
      if(is_pair(kv.first)||(is_range(kv.first)&&range_width(kv.first)>1))
        supports.push_back(&kv.second);
    std::sort(supports.begin(),supports.end(),[&](const Candidate*x,const Candidate*y){
      double xb=x->uses*(is_pair(x->key)?35000.0:3000.0)/double(x->ids->size()*4ULL);
      double yb=y->uses*(is_pair(y->key)?35000.0:3000.0)/double(y->ids->size()*4ULL);
      if(xb!=yb)return xb>yb;return x->key<y->key;
    });
    for(auto*c:supports){
      if(c->uses==0)continue;
      Selection t=s;t.ids.insert(c->key);t.bytes=account(t);
      if(t.bytes<=budget)s=std::move(t);
    }

    // Layout is deliberately a separate, frozen STAGED phase.  Rank layouts
    // by their local predicted saving per incremental byte, not raw frequency.
    // Fragment admission is never revisited in this phase.
    for(;;){
      const Candidate*best=nullptr;double best_density=0;Selection best_state=s;
      for(const auto&kv:catalog){
        const Candidate&c=kv.second;
        const bool eligible=(!is_pair(c.key)&&(!is_range(c.key)||range_width(c.key)==1))||s.ids.count(c.key);
        if(!eligible||c.uses==0||s.packed.count(c.key))continue;
        Selection t=s;t.packed.insert(c.key);t.bytes=account(t);
        if(t.bytes>budget||t.bytes<=s.bytes)continue;
        const double local_saving=double(c.uses)*(fragment_cost(c.key,s)-fragment_cost(c.key,t));
        const double density=local_saving/double(t.bytes-s.bytes);
        if(local_saving>0&&(density>best_density||(density==best_density&&(!best||c.key<best->key)))){
          best=&c;best_density=density;best_state=std::move(t);
        }
      }
      if(!best)break;
      s=std::move(best_state);
    }
    s.bytes=account(s);return s;
  }
  Selection joint_select(const std::vector<Query>&dev,const Selection&incumbent)const{Selection cur;double current=total_cost(dev,cur);for(size_t iteration=0;iteration<256;++iteration){Selection best=cur;double best_density=0,best_cost=current;auto consider=[&](Selection t){t.bytes=account(t);if(t.bytes>budget||t.bytes<=cur.bytes)return;double cost=total_cost(dev,t),benefit=current-cost,density=benefit/double(t.bytes-cur.bytes);if(benefit>0&&(density>best_density||(density==best_density&&cost<best_cost))){best_density=density;best_cost=cost;best=std::move(t);}};for(const auto&kv:catalog){const std::string&k=kv.first;if(is_pair(k)||(is_range(k)&&range_width(k)>1)){if(!cur.ids.count(k)){Selection t=cur;t.ids.insert(k);consider(t);t.packed.insert(k);consider(t);}else if(!cur.packed.count(k)){Selection t=cur;t.packed.insert(k);consider(t);}}else if(!cur.packed.count(k)){Selection t=cur;t.packed.insert(k);consider(t);}}for(const auto&q:dev)if(q.spec.family==Family::Range){Selection t=cur;for(uint32_t start=q.spec.lo;start+1<=q.spec.hi;++start)if(start%2==0){std::string k=range_key(2,start);t.ids.insert(k);t.packed.insert(k);}consider(t);}if(best_density<=0)break;cur=std::move(best);current=best_cost;}double incumbent_cost=total_cost(dev,incumbent);if(incumbent.bytes<=budget&&incumbent_cost<current)return incumbent;return cur;}
  void build(const std::vector<Query>&dev){auto add=[&](const std::string&k,const std::vector<uint32_t>&ids,uint64_t uses){auto&c=catalog[k];if(c.key.empty()){c.key=k;c.ids=&ids;}c.uses+=uses;};for(uint32_t a=0;a<CAT_A;++a)add("A"+std::to_string(a),d.a_post[a],0);for(uint32_t p=0;p<CAT_A*CAT_B;++p)add("P"+std::to_string(p),d.pair_post[p],0);for(uint32_t w:{1U,2U,4U,8U})for(uint32_t start=0;start<BINS;start+=w)add(range_key(w,start),d.dyadic_post.at(range_key(w,start)),0);for(const auto&q:dev)for(const auto&k:fragment_keys(q.spec)){if(k[0]=='A')add(k,d.a_post[std::stoul(k.substr(1))],1);else if(k[0]=='P')add(k,d.pair_post[std::stoul(k.substr(1))],1);else add(k,d.dyadic_post.at(k),1);}for(const auto&kv:catalog)universe.insert(kv.first);Selection staged=staged_select(dev);staged_predicted=total_cost(dev,staged);Selection selected=arm==Arm::Staged?staged:joint_select(dev,staged);predicted=total_cost(dev,selected);incumbent_used=arm==Arm::Joint&&selected.ids==staged.ids&&selected.packed==staged.packed;require(predicted<=staged_predicted+1e-6,"joint selector is worse than staged incumbent");selected.bytes=account(selected);require(selected.bytes<=budget,"selector exceeds byte budget");bytes=selected.bytes;admitted=selected.ids;for(const auto&k:selected.ids)materialized.emplace(k,*catalog.at(k).ids);for(const auto&k:selected.packed)packed.emplace(k,pack(d,*catalog.at(k).ids));std::ostringstream keys;bool first=true;for(const auto&key:universe){if(!first)keys<<',';first=false;keys<<key;}std::cout<<"MANIFEST arm="<<(arm==Arm::Joint?"joint":"staged")<<" candidate_universe=A_pair_dyadic catalog_states=ABSENT_IDS_PACKED candidates="<<catalog.size()<<" admitted="<<admitted.size()<<" packed="<<packed.size()<<" bytes="<<bytes<<" budget="<<budget<<" predicted_cost="<<predicted<<" staged_predicted_cost="<<staged_predicted<<" staged_incumbent_used="<<(incumbent_used?"true":"false")<<" candidate_keys="<<keys.str()<<std::endl;}
  Result one_fragment(const std::string&key,const uint8_t*q)const{const auto&ids=*catalog.at(key).ids;auto it=packed.find(key);return it==packed.end()?scan(d,q,ids):scan(d,q,ids,&it->second.vectors);}
  Result pair_fragment(uint32_t a,uint32_t b,const uint8_t*q)const{const std::string key="P"+std::to_string(pair_id(a,b));auto stored=materialized.find(key);if(stored!=materialized.end()){auto it=packed.find(key);return it==packed.end()?scan(d,q,stored->second):scan(d,q,stored->second,&it->second.vectors);}const auto&left=d.a_post[a];const auto&right=d.b_post[b];std::vector<uint32_t>ids;ids.reserve(std::min(left.size(),right.size()));size_t i=0,j=0;while(i<left.size()&&j<right.size()){if(left[i]<right[j])++i;else if(right[j]<left[i])++j;else{ids.push_back(left[i]);++i;++j;}}return scan(d,q,ids);}
  Result query(const Query&q)const{const Spec&s=q.spec;const uint8_t*v=q.vector.data();if(s.family==Family::Equality)return one_fragment("A"+std::to_string(s.a),v);if(s.family==Family::Conjunction)return pair_fragment(s.a,s.b,v);std::vector<Result>parts;if(s.family==Family::Range){Selection active;for(const auto&kv:materialized)active.ids.insert(kv.first);for(const auto&kv:packed)active.packed.insert(kv.first);for(const auto&k:cover(s.lo,s.hi,active).keys)parts.push_back(one_fragment(k,v));}else{parts.push_back(pair_fragment(s.a,s.b,v));parts.push_back(pair_fragment(s.c,s.d,v));}return merge_results(d,v,parts);}
  std::string route(const Spec&s)const{
    auto state=[&](const std::string&k){if(packed.count(k))return std::string("packed");if(materialized.count(k))return std::string("ids");return std::string("online");};
    if(s.family==Family::Equality){const std::string k="A"+std::to_string(s.a);return packed.count(k)?"equality_packed":"equality_base_ids";}
    if(s.family==Family::Conjunction)return "conjunction_"+state("P"+std::to_string(pair_id(s.a,s.b)));
    if(s.family==Family::Dnf){const std::string x=state("P"+std::to_string(pair_id(s.a,s.b))),y=state("P"+std::to_string(pair_id(s.c,s.d)));if(x=="online"||y=="online")return "dnf_has_online";if(x=="packed"&&y=="packed")return "dnf_all_packed";return "dnf_ids_or_mixed";}
    Selection active;for(const auto&kv:materialized)active.ids.insert(kv.first);for(const auto&kv:packed)active.packed.insert(kv.first);const auto c=cover(s.lo,s.hi,active);uint32_t packed_parts=0;for(const auto&k:c.keys)packed_parts+=packed.count(k);const std::string width=c.keys.size()==1?"single":"multi";const std::string layout=packed_parts==0?"ids":(packed_parts==c.keys.size()?"packed":"mixed");return "range_"+width+"_"+layout;
  }
};

hnswlib::QueryFilter sieve_filter(const Spec&s){
  std::unordered_set<int32_t> f;if(s.family==Family::Equality)f.insert(s.a);else if(s.family==Family::Conjunction)f.insert(pair_label(s.a,s.b));else if(s.family==Family::Range)for(uint32_t x=s.lo;x<=s.hi;++x)f.insert(BIN_BASE+x);else{f.insert(pair_label(s.a,s.b));f.insert(pair_label(s.c,s.d));}return hnswlib::QueryFilter(f,s.family!=Family::Range&&s.family!=Family::Dnf);
}
struct SieveEngine{
  const Data&d;std::vector<uint8_t> metadata;std::unique_ptr<hnswlib::DatasetFilters> filters;std::unique_ptr<hnswlib::L2SpaceX> space;std::unique_ptr<hnswlib::PartitionedHNSW<int,uint8_t>> index;
  SieveEngine(const Data&data,const std::vector<Query>&dev):d(data){
    const uint64_t nnz=uint64_t(N)*4;metadata.resize(24+size_t(N+1)*8+size_t(nnz)*4);uint64_t n=N,l=NLABELS;std::memcpy(metadata.data(),&n,8);std::memcpy(metadata.data()+8,&l,8);std::memcpy(metadata.data()+16,&nnz,8);for(uint32_t r=0;r<=N;++r){uint64_t o=uint64_t(r)*4;std::memcpy(metadata.data()+24+size_t(r)*8,&o,8);}size_t at=24+size_t(N+1)*8;for(auto&row:d.expanded_rows)for(uint32_t x:row){std::memcpy(metadata.data()+at,&x,4);at+=4;}
    FILE*fp=fmemopen(metadata.data(),metadata.size(),"rb");require(fp,"fmemopen failed");filters=std::make_unique<hnswlib::DatasetFilters>(fp,8,false);filters->transpose_inplace();filters->make_bvs();space=std::make_unique<hnswlib::L2SpaceX>(D/4);hnswlib::PartitionedIndexParams p{};p.dataset_size=N;p.dim=D;p.M=16;p.ef_construction=40;p.index_vector_budget=50000;p.bitvector_cutoff=512;p.historical_workload_window_size=dev.size();p.enable_heterogeneous_indexing=false;p.enable_heterogeneous_search=false;p.query_correlation_constant=.5f;p.num_threads=1;p.ef_search_scaling_constant=3.0f;p.enable_multipartition_search=false;std::vector<hnswlib::QueryFilter> history;for(auto&q:dev)history.push_back(sieve_filter(q.spec));std::cout<<"SIEVE_BUILD_START history="<<history.size()<<" index_vector_budget=50000 core_modified=false"<<std::endl;index=std::make_unique<hnswlib::PartitionedHNSW<int,uint8_t>>(const_cast<uint8_t*>(d.base.data()),space.get(),filters.get(),p,history);index->setEf(200);std::cout<<"SIEVE_BUILD_DONE"<<std::endl;
  }
  Result query(const Query&q){hnswlib::Predicate p(filters.get(),sieve_filter(q.spec));auto heap=index->searchKnn(q.vector.data(),K,p);Result r;r.ids.fill(UINT32_MAX);for(uint32_t i=0;i<K&&!heap.empty();++i){r.ids[i]=uint32_t(heap.top().second);heap.pop();}return r;}
};

struct Metrics{uint64_t elapsed=0,queries=0,hits=0,den=0,invalid=0,predicate=0,duplicates=0,nondeterministic=0;double qps()const{return double(queries)*1e9/double(elapsed);}};
template<class Fn> Metrics run_family(const Data&d,const std::vector<Query>&qs,Family f,uint32_t cycles,Fn&&fn){std::vector<size_t> rows;for(size_t i=0;i<qs.size();++i)if(qs[i].spec.family==f)rows.push_back(i);require(!rows.empty(),"family empty");std::vector<Result> out(rows.size());for(size_t i:rows)(void)fn(qs[i]);auto begin=Clock::now();for(uint32_t c=0;c<cycles;++c)for(size_t j=0;j<rows.size();++j)out[j]=fn(qs[rows[j]]);auto end=Clock::now();Metrics m;m.elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();m.queries=uint64_t(rows.size())*cycles;for(size_t j=0;j<rows.size();++j){const Result post1=fn(qs[rows[j]]),post2=fn(qs[rows[j]]);if(post1.ids!=out[j].ids||post2.ids!=out[j].ids)++m.nondeterministic;auto gt=scan(d,qs[rows[j]].vector.data(),support(d,qs[rows[j]].spec));std::set<uint32_t> exact(gt.ids.begin(),gt.ids.end()),unique;for(uint32_t id:out[j].ids){++m.den;if(!unique.insert(id).second)++m.duplicates;if(id>=N){++m.invalid;continue;}if(!matches(d,qs[rows[j]].spec,id))++m.predicate;if(exact.count(id))++m.hits;}}return m;}

double spearman(std::vector<double>x,std::vector<double>y){if(x.size()<2||x.size()!=y.size())return 0;auto rank=[](const std::vector<double>&v){std::vector<size_t>o(v.size());std::iota(o.begin(),o.end(),0);std::sort(o.begin(),o.end(),[&](size_t a,size_t b){return v[a]<v[b];});std::vector<double>r(v.size());for(size_t i=0;i<o.size();++i)r[o[i]]=i+1;return r;};auto a=rank(x),b=rank(y);double ma=std::accumulate(a.begin(),a.end(),0.0)/a.size(),mb=std::accumulate(b.begin(),b.end(),0.0)/b.size(),num=0,da=0,db=0;for(size_t i=0;i<a.size();++i){num+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)*(a[i]-ma);db+=(b[i]-mb)*(b[i]-mb);}return da>0&&db>0?num/std::sqrt(da*db):0;}

struct PredictionSample{double work=0,observed=0;std::string route,family;};
struct LinearFit{double intercept=0,slope=0;size_t rows=0;};
LinearFit fit_samples(const std::vector<PredictionSample>&rows){
  LinearFit f;f.rows=rows.size();require(!rows.empty(),"empty prediction stratum");
  double mx=0,my=0;for(const auto&r:rows){mx+=r.work;my+=r.observed;}mx/=rows.size();my/=rows.size();
  double cov=0,var=0;for(const auto&r:rows){cov+=(r.work-mx)*(r.observed-my);var+=(r.work-mx)*(r.work-mx);}
  f.slope=var>0?cov/var:0;f.intercept=my-f.slope*mx;return f;
}
double quantile(std::vector<double>v,double q){require(!v.empty(),"empty quantile");std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(std::ceil(q*v.size())-1))];}

template<class Fn,class RouteFn> void prediction_check(const Data&d,const std::vector<Query>&dev,const std::vector<Query>&held,Fn&&fn,RouteFn&&route_fn,const std::string&name){
  auto work=[&](const Query&q){return double(support(d,q.spec).size());};
  auto measure=[&](const Query&q){constexpr uint32_t reps=32;(void)fn(q);auto s=Clock::now();for(uint32_t i=0;i<reps;++i)(void)fn(q);auto e=Clock::now();return double(std::chrono::duration_cast<std::chrono::nanoseconds>(e-s).count())/reps;};
  std::map<std::string,std::vector<PredictionSample>>dev_route,dev_family;
  for(const auto&q:dev){PredictionSample r{work(q),measure(q),route_fn(q.spec),family_name(q.spec.family)};dev_route[r.route].push_back(r);dev_family[r.family].push_back(r);}
  std::map<std::string,LinearFit>route_fit,family_fit;
  for(const auto&kv:dev_route)route_fit.emplace(kv.first,fit_samples(kv.second));
  for(const auto&kv:dev_family)family_fit.emplace(kv.first,fit_samples(kv.second));
  std::vector<double>all_pred,all_obs,all_ape;std::map<std::string,std::vector<double>>route_ape;size_t route_covered=0;
  for(const auto&q:held){const std::string route=route_fn(q.spec),family=family_name(q.spec.family);const auto it=route_fit.find(route);const bool covered=it!=route_fit.end()&&it->second.rows>=2;const LinearFit&fit=covered?it->second:family_fit.at(family);route_covered+=covered;const double p=std::max(1.0,fit.intercept+fit.slope*work(q)),o=measure(q),ape=std::abs(p-o)/o;all_pred.push_back(p);all_obs.push_back(o);all_ape.push_back(ape);route_ape[route].push_back(ape);}
  for(const auto&kv:route_ape)std::cout<<"PREDICTION_STRATUM arm="<<name<<" family_route="<<kv.first<<" heldout_rows="<<kv.second.size()<<" median_ape="<<quantile(kv.second,.5)<<" p95_ape="<<quantile(kv.second,.95)<<" development_rows="<<(route_fit.count(kv.first)?route_fit.at(kv.first).rows:0)<<std::endl;
  std::cout<<"PREDICTION arm="<<name<<" model=family_route_stratified median_ape="<<quantile(all_ape,.5)<<" p95_ape="<<quantile(all_ape,.95)<<" spearman="<<spearman(all_pred,all_obs)<<" exact_route_coverage="<<double(route_covered)/held.size()<<" heldout_rows="<<held.size()<<std::endl;
}

int main_impl(int argc,char**argv){
  std::string system="joint",base,meta,query,attr;uint32_t dev_n=400,held_n=32,cycles=64;uint64_t budget=16ULL*1024*1024;
  for(int i=1;i<argc;++i){std::string k=argv[i];require(i+1<argc,"missing option value");std::string v=argv[++i];if(k=="--system")system=v;else if(k=="--base")base=v;else if(k=="--metadata")meta=v;else if(k=="--query")query=v;else if(k=="--attribute")attr=v;else if(k=="--dev")dev_n=std::stoul(v);else if(k=="--heldout")held_n=std::stoul(v);else if(k=="--cycles")cycles=std::stoul(v);else if(k=="--budget-bytes")budget=std::stoull(v);else fail("unknown option "+k);}
  require(!base.empty()&&!meta.empty()&&!query.empty()&&!attr.empty(),"input options required");require(dev_n<=400&&held_n<=400&&held_n%4==0,"query counts differ");Data d=load_data(base,meta,query,attr);auto dev=make_queries(d,dev_n,false),held=make_queries(d,held_n,true);std::array<Metrics,4> ms;
  if(system=="sieve"){SieveEngine e(d,dev);std::cout<<"SIEVE_ADAPTER_RESOURCE metadata_bytes="<<e.metadata.size()<<" label_memberships="<<uint64_t(N)*4<<" pair_labels=512 derived_pair_label_adapter=true pristine_official_sieve_engine=true heldout_composite_cache_warmed=true same_byte_budget=false"<<std::endl;for(uint32_t f=0;f<4;++f){ms[f]=run_family(d,held,Family(f),cycles,[&](const Query&q){return e.query(q);});std::cout<<"BLOCK system=sieve family="<<family_name(Family(f))<<" queries="<<ms[f].queries<<" elapsed_ns="<<ms[f].elapsed<<" qps="<<std::setprecision(12)<<ms[f].qps()<<" recall="<<double(ms[f].hits)/ms[f].den<<" invalid="<<ms[f].invalid<<" predicate_fail="<<ms[f].predicate<<" duplicate="<<ms[f].duplicates<<" nondeterministic="<<ms[f].nondeterministic<<std::endl;}}
  else {Arm a=system=="joint"?Arm::Joint:Arm::Staged;require(system=="joint"||system=="staged","system differs");Engine e(d,a,budget,dev);uint64_t reused=0,total_keys=0;for(const Query&q:held)for(const std::string&key:fragment_keys(q.spec)){++total_keys;if(e.universe.count(key))++reused;}std::cout<<"HELDOUT_REUSE system="<<system<<" reused_keys="<<reused<<" total_keys="<<total_keys<<" nonzero="<<(reused>0?"true":"false")<<std::endl;for(uint32_t f=0;f<4;++f){ms[f]=run_family(d,held,Family(f),cycles,[&](const Query&q){return e.query(q);});std::cout<<"BLOCK system="<<system<<" family="<<family_name(Family(f))<<" queries="<<ms[f].queries<<" elapsed_ns="<<ms[f].elapsed<<" qps="<<std::setprecision(12)<<ms[f].qps()<<" recall="<<double(ms[f].hits)/ms[f].den<<" invalid="<<ms[f].invalid<<" predicate_fail="<<ms[f].predicate<<" duplicate="<<ms[f].duplicates<<" nondeterministic="<<ms[f].nondeterministic<<std::endl;}prediction_check(d,dev,held,[&](const Query&q){return e.query(q);},[&](const Spec&s){return e.route(s);},system);std::cout<<"BYTES system="<<system<<" packed_bytes="<<e.bytes<<" budget_bytes="<<budget<<std::endl;}
  Metrics total;for(auto&m:ms){total.elapsed+=m.elapsed;total.queries+=m.queries;total.hits+=m.hits;total.den+=m.den;total.invalid+=m.invalid;total.predicate+=m.predicate;total.duplicates+=m.duplicates;total.nondeterministic+=m.nondeterministic;}std::cout<<"SUMMARY system="<<system<<" queries="<<total.queries<<" elapsed_ns="<<total.elapsed<<" qps="<<std::setprecision(12)<<total.qps()<<" recall="<<double(total.hits)/total.den<<" invalid="<<total.invalid<<" predicate_fail="<<total.predicate<<" duplicate="<<total.duplicates<<" nondeterministic="<<total.nondeterministic<<std::endl;return 0;
}
} // namespace
int main(int argc,char**argv){try{return main_impl(argc,argv);}catch(const std::exception&e){std::cerr<<"FATAL "<<e.what()<<std::endl;return 2;}}
