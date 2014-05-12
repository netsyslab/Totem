#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2013-03-24
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
include ../make.defs

TARGET  = alg

TOTEM_PATH   = ../totem
TOTEM_INC    = -I$(TOTEM_PATH)
TOTEM_DEPS   = $(wildcard $(TOTEM_PATH)/*.cuh) $(wildcard $(TOTEM_PATH)/*.h)

DEPS = $(CCDEPS) $(CUDEPS) $(TOTEM_DEPS)

.PHONY: objects clean clean-all tags

all: lib

lib: objects
	@printf "\nCreating %s library ...\n" $(TARGET)
	ar -r -u lib$(TARGET).a $(CCOBJS) $(CUOBJS)
	ranlib lib$(TARGET).a

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.cc.o: %.cc $(DEPS)
	$(CC) $(CCFLAGS) $(TOTEM_INC) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(TOTEM_INC) -c $< -o $@

clean:
	@$(CLEAN_CMD) lib$(TARGET).a

clean-all:
	@$(CLEAN_ALL_CMD) lib$(TARGET).a

tags:
	@$(ETAGS_CMD)
