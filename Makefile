## Flags
LD = g++
CXXFLAGS = -lrt -Wall -g -DLINUX -O2

LFLAGS = 

ASMDIR = ./asm
EXECDIR = ./exe
SRCDIR = ./src

HEADERS = matrix.h 
SRCS = $(notdir $(shell find $(SRCDIR) -name "*.cpp"))
EXES = $(SRCS:.cpp=)
o0FILES = $(EXES)o0.asm
o2FILES = $(EXES)o2.asm
funFILES = $(EXES)fun.asm

all : plot asm

clean :
	rm -f ./exe/*
	rm -f ./asm/*
$(EXECDIR)/% : $(SRCDIR)/%.cpp
	$(LD) $(CXXFLAGS) -o $@ $^ ${LFLAGS}

exes : $(addprefix $(EXECDIR)/,$(EXES))

plot : gnutest.csv
	gnuplot plotfile -

gnutest.csv : $(addprefix $(EXECDIR)/,$(EXES))
	$(EXECDIR)/matrix 2048 256 > test.csv
	$(EXECDIR)/matrixun2 2048 256 >> test.csv
	$(EXECDIR)/matrixun4 2048 256 >> test.csv
	python csvtrans.py test.csv  gnutest.csv

asms : $(o0FILES) $(o2FILES) $(funFILES)

$(ASMDIR)/%o0.asm: $(SRCDIR)/%.cpp 
	$(LD) -O0 -S $(CXXFLAGS) -o $@ $^ ${LFLAGS}
	
$(ASMDIR)/%o2.asm: $(SRCDIR)/%.cpp
	$(LD) -O2 -S $(CXXFLAGS) -o $@ $^ ${LFLAGS}

$(ASMDIR)/%fun.asm: $(SRCDIR)/%.cpp
	$(LD) -O2 -S -funroll-loops $(CXXFLAGS) -o $@ $^ ${LFLAGS}
