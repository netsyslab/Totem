#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
include ../../../make.defs

TARGET  = graph500

TOTEM_PATH   = "../../../totem"
TOTEM_INC    = -I$(TOTEM_PATH)
TOTEM_LIB    = -ltotem -L$(TOTEM_PATH)
TOTEM_DEPS   = $(wildcard $(ALG_PATH)/*.[cuh,h])

ALG_PATH   = "../../../alg"
ALG_INC    = -I$(ALG_PATH)
ALG_LIB    =  -lalg -L$(ALG_PATH)
ALG_DEPS   = $(wildcard $(ALG_PATH)/*.[cuh,h])

GRAPH500_INC = -I"../" -I"../generator"

NVCCFLAGS += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC) $(GRAPH500_INC)
CCFLAGS   += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC) $(GRAPH500_INC)
DEPS = $(CCDEPS) $(CUDEPS) $(TOTEM_DEPS) $(ALG_DEPS)

.PHONY: objects clean clean-all tags

all: $(TARGET)

$(TARGET): objects totem alg
	@printf "\nLinking %s executable ...\n" $(TARGET)
	$(NVCC) -o $@ $(COBJS) $(CCOBJS) $(CUOBJS) $(TOTEM_LIB) $(ALG_LIB) $(LFLAGS)

objects: $(OBJDIR) $(COBJS) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(COBJS): $(OBJDIR)/%.c.o: %.c $(DEPS)
	$(C) $(CCFLAGS) -std=c99 -c $< -o $@

$(CCOBJS): $(OBJDIR)/%.cc.o: %.cc $(DEPS)
	$(CC) $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

totem:
	@$(MAKE) -C $(TOTEM_PATH)

alg:
	@$(MAKE) -C $(ALG_PATH)

clean:
	@$(CLEAN_CMD)

clean-all:
	@$(CLEAN_ALL_CMD) $(TARGET)
	@$(MAKE) -C $(TOTEM_PATH) clean-all
	@$(MAKE) -C $(ALG_PATH) clean-all

tags:
	@$(ETAGS_CMD)
