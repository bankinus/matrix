## Flags
LD = g++
CPP = g++
CXXFLAGS = -lrt -Wall -g -DLINUX -O2

LFLAGS = 

ASMDIR = ./asm
EXECDIR = ./exe
OBJDIR = ./obj
SRCDIR = ./src
TILERESDIR = ./src

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
	$(LD) $(CXXFLAGS) -o $(EXECDIR)/tile_test $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h ${LFLAGS}
	$(EXECDIR)/tile_test 256 256 > $(TILERESDIR)/tile_test_unroll1.csv
	$(EXECDIR)/tile_test 2048 1024 >> $(TILERESDIR)/tile_test_unroll1.csv
	$(LD) $(CXXFLAGS) -o $(EXECDIR)/tile_test $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h ${LFLAGS} -unrollfactor=2
	$(EXECDIR)/tile_test 256 256 > $(TILERESDIR)/tile_test_unroll2.csv
	$(EXECDIR)/tile_test 2048 1024 >> $(TILERESDIR)/tile_test_unroll2.csv
	$(LD) $(CXXFLAGS) -o $(EXECDIR)/tile_test $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h ${LFLAGS} -unrollfactor=4
	$(EXECDIR)/tile_test 256 256 > $(TILERESDIR)/tile_test_unroll4.csv
	$(EXECDIR)/tile_test 2048 1024 >> $(TILERESDIR)/tile_test_unroll4.csv
	$(LD) $(CXXFLAGS) -o $(EXECDIR)/tile_test $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h ${LFLAGS} -unrollfactor=8
	$(EXECDIR)/tile_test 256 256 > $(TILERESDIR)/tile_test_unroll8.csv
	$(EXECDIR)/tile_test 2048 1024 >> $(TILERESDIR)/tile_test_unroll8.csv
	$(LD) $(CXXFLAGS) -o $(EXECDIR)/tile_test $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h ${LFLAGS} -unrollfactor=16
	$(EXECDIR)/tile_test 256 256 > $(TILERESDIR)/tile_test_unroll16.csv
	$(EXECDIR)/tile_test 2048 1024 >> $(TILERESDIR)/tile_test_unroll16.csv
    

unrollplot : gnutest.csv
	gnuplot plotfile -

plottile : tileplot
	gnuplot tileplot -

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
