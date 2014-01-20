## Flags
LD = g++
CPP = g++
CUCC = nvcc
CXXFLAGS = -Wall -g -DLINUX -O2
CUFLAGS =
LFLAGS = -lrt

ASMDIR = ./asm
EXECDIR = ./exe
OBJDIR = ./obj
SRCDIR = ./src
TILERESDIR = ./tile_test_results
VECRESDIR = ./vec_test_results
THREADRESDIR = ./thread_test_results
CUDARESDIR = ./cuda_test_results

HEADERS = matrix.h 
SRCS = $(notdir $(shell find $(SRCDIR) -name "*.cpp"))
OBJSRCS = $(notdir $(shell find $(SRCDIR) -name "*.cc"))
OBJ = $(OBJSRCS:.cc=.o)
EXES = $(SRCS:.cpp=)
o0FILES = $(EXES)o0.asm
o2FILES = $(EXES)o2.asm
funFILES = $(EXES)fun.asm

all : cuda_test.csv

clean :
	rm -f ./exe/*
	rm -f ./asm/*

$(EXECDIR)/cuda_test : $(SRCDIR)/cuda_test.cu $(SRCDIR)/cudamatrix.h 
	$(CUCC) $(CUFLAGS) -o $@ $(SRCDIR)/cuda_test.cu ${LFLAGS}
    
$(EXECDIR)/hello_cuda : $(SRCDIR)/hello_cuda.cu
	$(CUCC) $(CUFLAGS) -o $@ $^ ${LFLAGS}

$(EXECDIR)/tile_test : $(SRCDIR)/tile_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/rowtmatrix.h $(SRCDIR)/coltmatrix.h $(SRCDIR)/blocktmatrix.h
	$(LD) $(CXXFLAGS) -o $@ $^ ${LFLAGS}

$(EXECDIR)/vec_test : $(SRCDIR)/vec_test.cpp $(SRCDIR)/absmatrix.h $(SRCDIR)/vecmatrix.h
	$(LD) $(CXXFLAGS) -o $@ $^ -mavx ${LFLAGS}

$(EXECDIR)/thread_test : $(SRCDIR)/thread_test.cpp $(SRCDIR)/threadmatrix.h
	$(LD) $(CXXFLAGS) -o $@ $^ -std=c++11 -fopenmp ${LFLAGS}

$(EXECDIR)/% : $(SRCDIR)/%.cpp
	$(LD) $(CXXFLAGS) -o $@ $^ ${LFLAGS}

exes : $(addprefix $(EXECDIR)/,$(EXES))

cuda_test.csv : $(EXECDIR)/cuda_test
	$(EXECDIR)/cuda_test 64 64 > $(CUDARESDIR)/cuda_test.csv
	$(EXECDIR)/cuda_test 1024 128 >> $(CUDARESDIR)/cuda_test.csv

thread_test.csv : $(EXECDIR)/thread_test
	$(EXECDIR)/thread_test 64 64 > $(THREADRESDIR)/thread_test.csv
	$(EXECDIR)/thread_test 1024 128 >> $(THREADRESDIR)/thread_test.csv

vec_test.csv : $(EXECDIR)/vec_test
	$(EXECDIR)/vec_test 64 64 > $(VECRESDIR)/vec_test.csv
	$(EXECDIR)/vec_test 1024 128 >> $(VECRESDIR)/vec_test.csv

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
