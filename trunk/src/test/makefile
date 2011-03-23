#--------------------------------------------------------------------------
# Unit tests makefile
#
# Created on: 2011-03-18
# Author: Abdullah Gharaibeh
#
# To compile all test: make
# To compile a test  : make TEST=unit_test
#--------------------------------------------------------------------------
include ../make.defs

.PHONY: clean

TEST_FILE         = totem_$(TEST)_unittest.cc
ALL_TEST_FILES    = $(wildcard totem_*_unittest.cc)
EXE_EXT           = .test

GTEST_PATH = $(GTEST)
GTEST_SRC  = $(GTEST_PATH)/src/gtest_main.cc $(GTEST_PATH)/src/gtest-all.cc
GTEST_INC  = -I$(GTEST_PATH)/include/ -I$(GTEST_PATH)
GEST_LIB   = -L$(GTEST_PATH)/lib

TOTEM_PATH   = ../totem
TOTEM_INC    = -I$(TOTEM_PATH)
TOTEM_CCSRC  = $(filter-out %_main.cc, $(wildcard $(TOTEM_PATH)/*.cc))
TOTEM_CCOBJS = $(TOTEM_CCSRC:$(TOTEM_PATH)/%.cc=$(TOTEM_PATH)/$(OBJDIR)/%.o)
TOTEM_CUSRC  = $(wildcard $(TOTEM_PATH)/*.cu)
TOTEM_CUOBJS = $(TOTEM_CUSRC:$(TOTEM_PATH)/%.cu=$(TOTEM_PATH)/$(OBJDIR)/%.cu_o)

BUILD_LINE = $(GTEST_SRC) $(TOTEM_INC) $(GTEST_INC) $(CCFLAGS) $(LFLAGS) \
             -lpthread $(TOTEM_CCOBJS) $(TOTEM_CUOBJS)

# TODO(abdullah) compile all tests
all: environment
ifndef TEST
	@printf "\nBuilding all unit tests ...\n"
	$(CC) $(ALL_TEST_FILES) $(BUILD_LINE) -o all$(EXE_EXT)
else
	@if test ! -f $(TEST_FILE); then echo $(TEST) \
	    "unit test does not exist"; exit 2; fi
	@printf "\nBuilding %s unit test ...\n" $(TEST)
	$(CC) $(TEST_FILE) $(BUILD_LINE) -o $(TEST)$(EXE_EXT)
endif

environment:
ifndef GTEST
	@echo "GTEST environment variable is not defined"
	@exit 2
endif
	@$(MAKE) -C $(TOTEM_PATH) objects MP=$(MP)

# TODO(abdullah) clean tests binaries
clean:
	@$(CLEAN_CMD) *$(EXE_EXT)

clean-all: clean
	@$(MAKE) -C $(TOTEM_PATH) clean

