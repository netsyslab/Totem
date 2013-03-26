#--------------------------------------------------------------------------
# Unit tests makefile
#
# Created on: 2011-03-18
# Author: Abdullah Gharaibeh
#
# To compile all tests: make
# To compile a test   : make TEST=unit_test
#--------------------------------------------------------------------------
include ../make.defs

.PHONY: clean

TEST_FILE  = $(filter-out %~, $(wildcard totem_$(TEST)_unittest.*))
EXE_EXT   = .test

GTEST_PATH = $(GTEST)
GTEST_SRC  = $(GTEST_PATH)/src/gtest_main.cc $(GTEST_PATH)/src/gtest-all.cc
GTEST_INC  = -I$(GTEST_PATH)/include/ -I$(GTEST_PATH)
GEST_LIB   = -L$(GTEST_PATH)/lib

TOTEM_PATH   = ../totem
TOTEM_INC    = -I$(TOTEM_PATH)
TOTEM_LIB    = -ltotem -L$(TOTEM_PATH)
TOTEM_DEPS   = $(wildcard $(TOTEM_PATH)/*.[cuh,h])

ALG_PATH   = ../alg
ALG_INC    = -I$(ALG_PATH)
ALG_LIB    =  -lalg -L$(ALG_PATH)
ALG_DEPS   = $(wildcard $(ALG_PATH)/*.[cuh,h])

ifdef TEST
CCSRC = $(filter-out %.cu, $(TEST_FILE))
CUSRC = $(filter-out %.cc, $(TEST_FILE))
CCOBJS = $(CCSRC:%.cc=$(OBJDIR)/%.o)
CUOBJS = $(CUSRC:%.cu=$(OBJDIR)/%.cu_o)
endif
ALL_OBJS = $(CCOBJS) $(CUOBJS)

NVCCFLAGS += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC)
CCFLAGS   += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC)
LFLAGS    += $(TOTEM_LIB) $(ALG_LIB)
BUILD_LINE = $(ALL_OBJS) $(GTEST_SRC) $(GTEST_INC) $(LFLAGS)

DEPS = $(CCDEPS) $(CUDEPS) $(TOTEM_DEPS) $(ALG_DEPS)

all: environment totem alg objects
ifndef TEST
	@printf "\nBuilding all unit tests ...\n"
	$(CC) $(BUILD_LINE) -o all$(EXE_EXT)
else
	@if test ! $(TEST_FILE); then \
	  printf "\n%s unit test does not exist\n" $(TEST); \
	  exit 2; \
	fi
	@printf "\nBuilding %s unit test ...\n" $(TEST_FILE)
	$(CC) $(BUILD_LINE) -o $(TEST)$(EXE_EXT)
endif

environment:
ifndef GTEST
	@echo "GTEST environment variable is not defined"
	@exit 2
endif

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.o: %.cc $(DEPS)
	$(CC) $(CCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu_o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

totem:
	@$(MAKE) -C $(TOTEM_PATH)

alg:
	@$(MAKE) -C $(ALG_PATH)

clean:
	@$(CLEAN_CMD) *$(EXE_EXT)

clean-all: clean
	@$(CLEAN_ALL_CMD)
	@$(MAKE) -C $(TOTEM_PATH) clean
	@$(MAKE) -C $(ALG_PATH) clean
