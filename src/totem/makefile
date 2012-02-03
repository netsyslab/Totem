#--------------------------------------------------------------------------
# Main makefile
#
# Created on: 2011-02-28
# Author: Abdullah Gharaibeh
#--------------------------------------------------------------------------
include ../make.defs

TARGET  = totem
LIBOBJS = $(filter-out %_main.o, $(CCOBJS)) $(filter-out %_main.cu_o, $(CUOBJS))
MAINOBJS =  $(filter %_main.o, $(CCOBJS)) $(filter %_main.cu_o, $(CUOBJS))

.PHONY: objects clean clean-all tags

all: $(TARGET)

$(TARGET): lib 
	@printf "\nLinking %s executable ...\n" $(TARGET)
	$(CC) -o $@ $(MAINOBJS) -l$(TARGET) $(LFLAGS)

lib: objects
	@printf "\nCreating %s library ...\n" $(TARGET)
	ar -r -u lib$(TARGET).a $(LIBOBJS)
	ranlib lib$(TARGET).a

objects: $(OBJDIR) $(CCOBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(CCOBJS): $(OBJDIR)/%.o: %.cc $(CCDEPS)
	$(CC) $(CCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu_o: %.cu $(CCDEPS) $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) $(TOTEM_FEATURES) -c $< -o $@

clean:
	@$(CLEAN_CMD) lib$(TARGET).a

clean-all:
	@$(CLEAN_ALL_CMD) lib$(TARGET).a

tags:
	@$(ETAGS_CMD)
