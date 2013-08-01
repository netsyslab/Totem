#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
include ../make.defs

TARGET  = totem

.PHONY: objects clean clean-all tags

all: lib

lib: objects
	@printf "\nCreating %s library ...\n" $(TARGET)
	ar -r -u lib$(TARGET).a $(CCOBJS) $(CUOBJS)
	ranlib lib$(TARGET).a

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.o: %.cc $(CCDEPS)
	$(CC) $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.o: %.cu $(CCDEPS) $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	@$(CLEAN_CMD) lib$(TARGET).a

clean-all:
	@$(CLEAN_ALL_CMD) lib$(TARGET).a

tags:
	@$(ETAGS_CMD)
