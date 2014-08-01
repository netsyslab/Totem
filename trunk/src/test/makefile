#--------------------------------------------------------------------------
# Unit tests makefile.
#
# Created on: 2011-03-18
# Author: Abdullah Gharaibeh
#
# To compile all tests: make
# To compile a test   : make TEST=unit_test
#--------------------------------------------------------------------------
TARGET  = tests
ROOTDIR = ..
include $(ROOTDIR)/make.defs

ifndef GTEST
  $(error GTEST environment variable is not defined)
endif

ifdef TEST
  TEST_FILE = $(filter-out %~, $(wildcard totem_$(TEST)_unittest.*))
  ifeq ($(strip $(TEST_FILE)),)
    $(error $(TEST) unit test does not exist)
  endif
  CCSRC = $(filter-out %.cu, $(TEST_FILE))
  CUSRC = $(filter-out %.cc, $(TEST_FILE))
  CCOBJS = $(CCSRC:%.cc=$(OBJDIR)/%.cc.o)
  CUOBJS = $(CUSRC:%.cu=$(OBJDIR)/%.cu.o)
endif
ALL_OBJS = $(CCOBJS) $(CUOBJS)

INC := -I. -I$(GTEST_PATH)/include/ -I$(GTEST_PATH) -I$(TOTEM_PATH) \
       -I$(ALG_PATH)
LIBS = -lalg -ltotem -L$(LIBDIR) -L$(GTEST_PATH)/lib
DEPS := $(DEPS) $(TOTEM_DEPS) $(ALG_DEPS)
BUILD_LINE = $(ALL_OBJS) $(GTEST_SRC) $(INC) $(LIBS) $(LFLAGS)
EXE_EXT   = .test

all: objects libalg libtotem $(BINDIR)
ifndef TEST
	@printf "\nBuilding all unit tests ...\n"
	$(NVCC) $(BUILD_LINE) -o $(BINDIR)/all$(EXE_EXT)
else
	@printf "\nBuilding %s unit test ...\n" $(TEST_FILE)
	$(NVCC) $(BUILD_LINE) -o $(BINDIR)/$(TEST)$(EXE_EXT)
endif

include $(ROOTDIR)/make.rules
