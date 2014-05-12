# -*- Makefile -*-
# Copyright 2010, Georgia Institute of Technology, USA.
# See COPYING for license.
CFLAGS = -g -std=c99
#CFLAGS = -g -std=c99 -O3 -march=native -fgcse-sm -fgcse-las -fgcse-after-reload -floop-strip-mine -ftree-loop-im -fivopts -funswitch-loops

# TOTEM_PATH   ="../totem-graph/trunk/src/totem"
# TOTEM_INC    = -I$(TOTEM_PATH)
# TOTEM_LIB    = -ltotem -L$(TOTEM_PATH)

LDLIBS = -lm -lrt $(TOTEM_LIB)
CPPFLAGS =  -D__STDC_LIMIT_MACROS $(TOTEM_INC)

#-DUSE_MMAP_LARGE -DUSE_MMAP_LARGE_EXT

BUILD_OPENMP = Yes
CFLAGS_OPENMP = -fopenmp 
