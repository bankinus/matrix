#include <iostream>
#include "vecmatrix.h"
#include "blocktmatrix.h"

#define repeats 20

int main (int argc, char** argv){
    int maxdim = 1024;
	int step = 256;
	if (argc > 1) maxdim = atoi(argv[1]);
	if (argc > 2) step = atoi(argv[2]);
	
	for (int dim = step; dim <= maxdim; dim += step){
		struct timespec begin;
        struct timespec end;
        time_t sec;
        long nsec;
        long msec;
		float* d1 = new float[dim * dim];
		float* d2 = new float[dim * dim];
		for (int i = 0; i < dim; i++){
			for (int j = 0; j < dim; j++){
				d1[dim * i + j] = 1.0;
				d2[dim * i + j] = 1.0;
			}
		}
        std::cout << dim << "\t";
       	BlocktMatrix<float> b1(dim, dim, d1);
		BlocktMatrix<float> b2(dim, dim, d2);
		BlocktMatrix<float> b3(dim, dim);
        clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			b3 = b1 * b2;
			b2 = b1 * b3;
			b1 = b2 * b3;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec + nsec;
        msec = msec;
		std::cout << msec << "\t";

		SSEVecFloatMatrix s1(dim, dim, d1);
		SSEVecFloatMatrix s2(dim, dim, d2);
		SSEVecFloatMatrix s3(dim, dim);
        clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			s3 = s1 * s2;
			s2 = s1 * s3;
			s1 = s2 * s3;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec + nsec;
        msec = msec;
		std::cout << msec << "\t";
 
		AVXVecFloatMatrix a1(dim, dim, d1);
		AVXVecFloatMatrix a2(dim, dim, d2);
		AVXVecFloatMatrix a3(dim, dim);
        clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			a3 = a1 * a2;
			a2 = a1 * a3;
			a1 = a2 * a3;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec + nsec;
        msec = msec;
		std::cout << msec << std::endl; 
	}
	return 0;
   
}
