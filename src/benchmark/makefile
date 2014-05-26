#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
include ../make.defs

TARGET  = benchmark

TOTEM_PATH   = ../totem
TOTEM_INC    = -I$(TOTEM_PATH)
TOTEM_LIB    = -ltotem -L$(TOTEM_PATH)
TOTEM_DEPS   = $(wildcard $(TOTEM_PATH)/*.cuh) $(wildcard $(TOTEM_PATH)/*.h)

ALG_PATH   = ../alg
ALG_INC    = -I$(ALG_PATH)
ALG_LIB    =  -lalg -L$(ALG_PATH)
ALG_DEPS   = $(wildcard $(ALG_PATH)/*.[cuh,h])

NVCCFLAGS += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC)
CCFLAGS   += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC)
DEPS = $(CCDEPS) $(CUDEPS) $(TOTEM_DEPS) $(ALG_DEPS)

CCLIBOBJS=$(filter-out %_binary.cc.o, $(CCOBJS))
CULIBOBJS=$(filter-out %_binary.cu.o, $(CUOBJS))
CCBINOBJS=$(filter %_binary.cc.o, $(CCOBJS))
CUBINOBJS=$(filter %_binary.cu.o, $(CUOBJS))


.PHONY: objects clean clean-all tags

all: $(TARGET)

lib: objects
	@printf "\nCreating %s library ...\n" $(TARGET)
	ar -r -u lib$(TARGET).a $(CCLIBOBJS) $(CULIBOBJS)
	ranlib lib$(TARGET).a

$(TARGET): lib totem alg
	@printf "\nLinking %s executable ...\n" $(TARGET)
	$(NVCC) -o $@ $(CCBINOBJS) $(CUBINOBJS) -l$(TARGET) $(ALG_LIB) $(TOTEM_LIB) $(LFLAGS)

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.cc.o: %.cc $(DEPS)
	$(CC) $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

totem:
	@$(MAKE) -C $(TOTEM_PATH)

alg:
	@$(MAKE) -C $(ALG_PATH)

clean:
	@$(CLEAN_CMD) $(TARGET)

clean-all:
	@$(CLEAN_ALL_CMD) lib$(TARGET).a
	@$(MAKE) -C $(TOTEM_PATH) clean-all
	@$(MAKE) -C $(ALG_PATH) clean-all

tags:
	@$(ETAGS_CMD)
