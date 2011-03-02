#--------------------------------------------------------------------------
# Main makefile
#--------------------------------------------------------------------------
include make.defs

TARGET = graph

.PHONY: objects clean clean-all tags

all: $(TARGET)

$(TARGET): objects
	@printf "\nLinking %s ...\n" $(TARGET)
	$(CC) -o $@ $(COBJS) $(CUOBJS) $(LFLAGS)
	@$(ETAGSCMD)

objects: $(OBJDIR) $(COBJS) $(CUOBJS)

$(OBJDIR):
	@if test ! -d $(OBJDIR); then mkdir $(OBJDIR); fi

$(COBJS): $(OBJDIR)/%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(MSG_LVL) -c $< -o $@

$(CUOBJS): $(OBJDIR)/%.cu_o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(MSG_LVL) -c $< -o $@

clean:
	@$(CLEAN_CMD)

clean-all:
	@$(CLEAN_ALL_CMD)

tags:
	@$(ETAGS_CMD)
