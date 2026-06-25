// BCI main entrypoint: instantiate FilteredIVFIndex as the
// fallback executor underneath the BCI router layer that will be added
// in Week 2). For now this file just verifies the build wiring; the IVF
// index itself is templated and heavy, so we keep the include but skip
// the actual fit. Real fit is invoked

#include <cstdio>
#include <iostream>
#include <string>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

int main(int argc, char** argv) {
  printf("=== BCI main entrypoint ===\n");
  printf("parlay workers = %ld\n", parlay::num_workers());
  printf("        Week 2 ports HAMCG; Week 3 router + bench.\n");
  return 0;
}
