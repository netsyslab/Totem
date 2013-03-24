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

ALG_PATH   = ../alg
ALG_INC    = -I$(ALG_PATH)
ALG_LIB    =  -lalg -L$(ALG_PATH)

NVCCFLAGS += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC)
CCFLAGS   += $(GTEST_INC) $(TOTEM_INC) $(ALG_INC)

.PHONY: objects clean clean-all tags

all: $(TARGET)

$(TARGET): objects totem alg
	@printf "\nLinking %s executable ...\n" $(TARGET)
	$(CC) -o $@ $(CCOBJS) $(CUOBJS) $(TOTEM_LIB) $(ALG_LIB) $(LFLAGS)

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.o: %.cc $(CCDEPS)
	$(CC) $(CCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu_o: %.cu $(CCDEPS) $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

totem:
	@$(MAKE) -C $(TOTEM_PATH)

alg:
	@$(MAKE) -C $(ALG_PATH)

clean:
	@$(CLEAN_CMD) lib$(TARGET).a

clean-all:
	@$(CLEAN_ALL_CMD) lib$(TARGET).a

tags:
	@$(ETAGS_CMD)
