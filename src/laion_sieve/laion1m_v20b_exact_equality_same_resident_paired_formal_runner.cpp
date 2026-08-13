// LAION1M v20b paired runner: v20 exact routes with a finite-domain equality
// census frozen by the manifest/controller population contract.

#define BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
#define BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS "evidence_class=frozen_formal"
#define BOOLEANN_LAION_HEARTBEAT_RECORD \
  "FORMAL_HEARTBEAT schema=laion1m_v20b_exact_equality_formal_heartbeat_v1"
#define BOOLEANN_LAION_QUERY_AUDIT_SCHEMA \
  "laion1m-v20b-exact-equality-formal-query-audit/v1"
#define BOOLEANN_LAION_BLOCK_SUMMARY_SCHEMA \
  "laion1m-v20b-exact-equality-formal-block-summary/v1"
#define BOOLEANN_LAION_RESOURCE_SCHEMA \
  "laion1m_v20b_exact_equality_same_resident_resource_v1"
#define BOOLEANN_LAION_ACTIVATION_SCHEMA \
  "laion1m_v20b_exact_equality_same_resident_activation_v1"
#define BOOLEANN_LAION_FINAL_SCHEMA \
  "laion1m_v20b_exact_equality_same_resident_final_v1"
#define BOOLEANN_LAION_V19_NO_MAIN
#include "laion1m_v19_same_resident_paired_formal_runner.cpp"
#undef BOOLEANN_LAION_V19_NO_MAIN
#undef BOOLEANN_LAION_FINAL_SCHEMA
#undef BOOLEANN_LAION_ACTIVATION_SCHEMA
#undef BOOLEANN_LAION_RESOURCE_SCHEMA
#undef BOOLEANN_LAION_BLOCK_SUMMARY_SCHEMA
#undef BOOLEANN_LAION_QUERY_AUDIT_SCHEMA
#undef BOOLEANN_LAION_HEARTBEAT_RECORD
#undef BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS
#undef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE

int main(int argc, char** argv) {
  try {
    return laion1m_v19_same_resident_paired_formal_runner::formal_main(
        argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
