#--------------------------------------------------------------------------
# Benchmark makefile.
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
TARGET  = benchmark
ROOTDIR = ..
include $(ROOTDIR)/make.defs

DEPS := $(DEPS) $(TOTEM_DEPS) $(ALG_DEPS)
INC  := $(INC) -I$(TOTEM_PATH) -I$(ALG_PATH)
LIBS = -l$(TARGET) -lalg -ltotem -L$(LIBDIR)

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(BINDIR) lib libtotem libalg
	@printf "\nLinking %s executable ...\n" $(BINDIR)/$(TARGET)
	$(NVCC) -o $@ $(CBINOBJS) $(CCBINOBJS) $(CUBINOBJS) $(LIBS) $(LFLAGS)

include $(ROOTDIR)/make.rules
