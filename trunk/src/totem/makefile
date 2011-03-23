#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
include ../make.defs

TARGET = graph

.PHONY: objects clean clean-all tags

all: $(TARGET)

$(TARGET): objects
	@printf "\nLinking %s ...\n" $(TARGET)
	$(CC) -o $@ $(CCOBJS) $(CUOBJS) $(LFLAGS)

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.o: %.cc $(CCDEPS)
	$(CC) $(CCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu_o: %.cu $(CDEPS) $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

clean:
	@$(CLEAN_CMD)

clean-all:
	@$(CLEAN_ALL_CMD)

tags:
	@$(ETAGS_CMD)
