#--------------------------------------------------------------------------
# Graph500 makefile.
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
TARGET  = graph500
ROOTDIR = ../../..
include $(ROOTDIR)/make.defs

DEPS := $(DEPS) $(TOTEM_DEPS) $(ALG_DEPS) $(BENCHMARK_DEPS)
INC  := $(INC) -I$(TOTEM_PATH) -I$(ALG_PATH) -I$(BENCHMARK_PATH) -I.. \
        -I../generator
LIBS = -lalg -ltotem -lbenchmark -L$(LIBDIR)

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): objects $(BINDIR) libtotem libalg libbenchmark
	@printf "\nLinking %s executable ...\n" $(BINDIR)/$(TARGET)
	$(NVCC) -o $@ $(COBJS) $(CCOBJS) $(CUOBJS) $(LIBS) $(LFLAGS)

include $(ROOTDIR)/make.rules
