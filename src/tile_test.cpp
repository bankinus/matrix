#include <iostream>
#include "rowtmatrix.h"
#include "coltmatrix.h"
#include "blocktmatrix.h"

#define repeats 2

int main (int argc, char** argv){
    int maxdim = 2048;
	int step = 512;
	if (argc > 1) maxdim = atoi(argv[1]);
	if (argc > 2) step = atoi(argv[2]);
	
	for (int dim = step; dim <= maxdim; dim += step){
		struct timespec begin;
        struct timespec end;
        time_t sec;
        long nsec;
        long msec;
		int* d1 = new int[dim * dim];
		int* d2 = new int[dim * dim];
		for (int i = 0; i < dim; i++){
			for (int j = 0; j < dim; j++){
				d1[dim * i + j] = 1;
				d2[dim * i + j] = 1;
			}
		}
        std::cout << dim << "\t";
		RowtMatrix<int> r1(dim, dim, d1);
		RowtMatrix<int> r2(dim, dim, d2);
		RowtMatrix<int> r3(dim, dim);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			r3 = r1 * r2;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec + nsec;
        msec = msec / repeats;
		std::cout << msec << "\t";

		ColtMatrix<int> c1(dim, dim, d1);
	    ColtMatrix<int> c2(dim, dim, d2);
		ColtMatrix<int> c3(dim, dim);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			c3 = c1 * c2;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = msec / repeats;
		std::cout << msec << "\t";

   		BlocktMatrix<int> b1(dim, dim, d1);
		BlocktMatrix<int> b2(dim, dim, d2);
		BlocktMatrix<int> b3(dim, dim);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			b3 = b1 * b2;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = msec / repeats;
		std::cout << msec << std::endl;    
	}
	return 0;
   
}
