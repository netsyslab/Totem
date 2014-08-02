#--------------------------------------------------------------------------
# Utility tool makefile.
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
TARGET  = generator
ROOTDIR = ..
include $(ROOTDIR)/make.defs

DEPS := $(DEPS) $(TOTEM_DEPS)
INC  := $(INC) -I$(TOTEM_PATH)
LIBS = -ltotem -L$(LIBDIR)

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): objects $(BINDIR) libtotem
	@printf "\nLinking %s executable ...\n" $(BINDIR)/$(TARGET)
	$(NVCC) -o $@ $(COBJS) $(CCOBJS) $(CUOBJS) $(LIBS) $(LFLAGS)

include $(ROOTDIR)/make.rules
