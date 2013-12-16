#include <iostream>
#include "threadmatrix.h"

#define repeats 5

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

   		OpenMPMatrix<int> o1(dim, dim, d1);
		OpenMPMatrix<int> o2(dim, dim, d2);
		OpenMPMatrix<int> o3(dim, dim);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			o3 = o1 * o2;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << std::endl;    
	}
	return 0;
   
}
