#--------------------------------------------------------------------------
# Wattsup makefile.
#
# Created on: 2014-08-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
TARGET  = wattsup
ROOTDIR = ../..
include $(ROOTDIR)/make.defs

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(BINDIR) objects
	@printf "\nLinking %s executable ...\n" $(BINDIR)/$(TARGET)
	$(NVCC) -o $@ $(COBJS) $(CCOBJS) $(CUOBJS) $(LIBS) $(LFLAGS)

include $(ROOTDIR)/make.rules
