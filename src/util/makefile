#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2011-03-18
# Author: Abdullah Gharaibeh
#         Sidney Pontes Filho
#--------------------------------------------------------------------------
include ../make.defs

.PHONY: clean

TOTEM_PATH   = ../totem
TOTEM_INC    = -I$(TOTEM_PATH)
TOTEM_LIB    = -ltotem -L$(TOTEM_PATH)
TOTEM_DEPS   = $(wildcard $(TOTEM_PATH)/*.[cuh,h])

CCOBJS = $(CCSRC:%.cc=$(OBJDIR)/%.o)
CUOBJS = $(CUSRC:%.cu=$(OBJDIR)/%.cu_o)

ALL_OBJS = $(CCOBJS) $(CUOBJS)

NVCCFLAGS += $(TOTEM_INC)
CCFLAGS   += $(TOTEM_INC)
LFLAGS    += $(TOTEM_LIB)
BUILD_LINE = $(ALL_OBJS) $(LFLAGS)

DEPS = $(CCDEPS) $(CUDEPS) $(TOTEM_DEPS)

all: totem objects
	@printf "\nBuilding executable ...\n"
	$(CC) $(BUILD_LINE) -o benchmark

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.o: %.cc $(DEPS)
	$(CC) $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu_o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

totem:
	@$(MAKE) -C $(TOTEM_PATH)

clean:
	@$(CLEAN_CMD) *$(EXE_EXT)

clean-all: clean
	@$(CLEAN_ALL_CMD)
	@$(MAKE) -C $(TOTEM_PATH) clean

