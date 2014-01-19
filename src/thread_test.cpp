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

   		ThreadMatrix<int> t11(dim, dim, d1, 1);
		ThreadMatrix<int> t12(dim, dim, d2, 1);
		ThreadMatrix<int> t13(dim, dim, 1);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			t13 = t11 * t12;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << "\t";

   		ThreadMatrix<int> t21(dim, dim, d1, 2);
		ThreadMatrix<int> t22(dim, dim, d2, 2);
		ThreadMatrix<int> t23(dim, dim, 2);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			t23 = t21 * t22;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << "\t";

   		ThreadMatrix<int> t41(dim, dim, d1, 4);
		ThreadMatrix<int> t42(dim, dim, d2, 4);
		ThreadMatrix<int> t43(dim, dim, 4);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			t43 = t41 * t42;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << "\t";

   		ThreadMatrix<int> t81(dim, dim, d1, 8);
		ThreadMatrix<int> t82(dim, dim, d2, 8);
		ThreadMatrix<int> t83(dim, dim, 8);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			t83 = t81 * t82;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << "\t";
        
        ThreadMatrix<int> t161(dim, dim, d1, 16);
		ThreadMatrix<int> t162(dim, dim, d2, 16);
		ThreadMatrix<int> t163(dim, dim, 16);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			t163 = t161 * t162;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << "\t";

        ThreadMatrix<int> t321(dim, dim, d1, 32);
		ThreadMatrix<int> t322(dim, dim, d2, 32);
		ThreadMatrix<int> t323(dim, dim, 32);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(volatile int i=0; i<repeats; i++){
			t323 = t321 * t322;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
        sec = end.tv_sec - begin.tv_sec;
        nsec = end.tv_nsec - begin.tv_nsec;
        nsec /= 1000000;
        sec *= 1000;
        msec = sec+nsec;
		std::cout << msec << "\t";

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
