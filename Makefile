## Flags
LD = g++
CPP = g++
CXXFLAGS = -lrt -Wall -g -DLINUX -O0

LFLAGS = 

ASMDIR = ./asm
EXECDIR = ./exe
OBJDIR = ./obj
SRCDIR = ./src

HEADERS = matrix.h 
SRCS = $(notdir $(shell find $(SRCDIR) -name "*.cpp"))
OBJSRCS = $(notdir $(shell find $(SRCDIR) -name "*.cc"))
OBJ = $(OBJSRCS:.cc=.o)
EXES = $(SRCS:.cpp=)
o0FILES = $(EXES)o0.asm
o2FILES = $(EXES)o2.asm
funFILES = $(EXES)fun.asm

all : tile_test.csv

clean :
	rm -f ./exe/*
	rm -f ./asm/*
	rm -f gnutest.csv
	rm -f test.csv

$(EXECDIR)/tile_test : $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h
	$(LD) $(CXXFLAGS) -o $@ $^ ${LFLAGS}

$(EXECDIR)/% : $(SRCDIR)/%.cpp
	$(LD) $(CXXFLAGS) -o $@ $^ ${LFLAGS}

exes : $(addprefix $(EXECDIR)/,$(EXES))

tile_test.csv : $(addprefix $(EXECDIR)/,$(EXES))
	$(EXECDIR)/tile_test > tile_test.csv

plot : gnutest.csv
	gnuplot plotfile -

gnutest.csv : $(addprefix $(EXECDIR)/,$(EXES))
	$(EXECDIR)/matrix 1024 64 > test.csv
	$(EXECDIR)/matrixun2 1024 64 >> test.csv
	$(EXECDIR)/matrixun4 1024 64 >> test.csv
	python csvtrans.py test.csv  gnutest.csv
	rm test.csv

asms : $(o0FILES) $(o2FILES) $(funFILES)

$(ASMDIR)/%o0.asm: $(SRCDIR)/%.cpp 
	$(LD) -O0 -S $(CXXFLAGS) -o $@ $^ ${LFLAGS}
	
$(ASMDIR)/%o2.asm: $(SRCDIR)/%.cpp
	$(LD) -O2 -S $(CXXFLAGS) -o $@ $^ ${LFLAGS}

$(ASMDIR)/%fun.asm: $(SRCDIR)/%.cpp
	$(LD) -O2 -S -funroll-loops $(CXXFLAGS) -o $@ $^ ${LFLAGS}
