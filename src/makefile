#--------------------------------------------------------------------------
# High-level makefile.
#
# Created on: 2014-07-27
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
ROOTDIR = .
include $(ROOTDIR)/make.defs

.PHONY: benchmark generator graph500 test

benchmark:
	@$(MAKE) -C $(BENCHMARK_PATH)

generator:
	@$(MAKE) -C $(GENERATOR_PATH)

graph500:
	@$(MAKE) -C $(GRAPH500_PATH)

test:
	@$(MAKE) -C $(TEST_PATH)

wattsup:
	@$(MAKE) -C $(WATTSUP_PATH)

clean-build:
	@rm -rf *~ $(BUILDDIR)

clean-all:
	@rm -rf *~ $(BUILDBASEDIR)
