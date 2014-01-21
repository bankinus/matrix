#ifndef _CUDAMT_H_
#define _CUDAMT_H_

#include "absmatrix.h"

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

template <class T>
__global__
void mult_kernel(T *d_a, T *d_b, T *d_c, unsigned int DIM){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    d_c[DIM * y + x] = 0;
    for (int i = 0; i < DIM; i++){
        d_c[DIM * y + x] += d_a[DIM * y + i] * d_b[DIM * i + x];
    }
}

template <class T>
class CudaMatrix : public AbsMatrix<T>{
    private:
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        CudaMatrix(unsigned int m, unsigned int n) : AbsMatrix<T>(m, n){}

        //! additional contructor which copies data directly
        CudaMatrix(unsigned int m, unsigned int n, T* data): AbsMatrix<T>(m, n, data){}

        //! copy constructor; we want to copy real data not references
        CudaMatrix(const AbsMatrix<T>& other) : AbsMatrix<T>(other){}

        ~CudaMatrix(){}

        CudaMatrix operator*(const CudaMatrix<T> &other){
            CudaMatrix<T> h_c = CudaMatrix(this->getHeight(), other.getWidth());

            const unsigned int a_size = this->getHeight() * this->getWidth();
            const unsigned int a_byte = a_size * sizeof(T);
            const unsigned int b_size = other.getHeight() * other.getWidth();
            const unsigned int b_byte = b_size * sizeof(T);
            const unsigned int c_size = h_c.getHeight() * h_c.getWidth();
            const unsigned int c_byte = c_size * sizeof(T);

            T *d_a;//this
            T *d_b;//other
            T *d_c;//result

            cudaMalloc((void **) &d_a, a_byte);
            cudaMalloc((void **) &d_b, b_byte);
            cudaMalloc((void **) &d_c, c_byte);

            cudaMemcpy(d_a, this->_data, a_byte, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, other._data, b_byte, cudaMemcpyHostToDevice);

            dim3 blockDim(h_c.getWidth(), h_c.getHeight());
            mult_kernel<<<1, blockDim>>>(d_a, d_b, d_c, h_c.getHeight());

            cudaMemcpy(h_c._data, d_c, c_byte, cudaMemcpyDeviceToHost);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

            return h_c;
        }
};


template <class T>
__global__
void smult_kernel(T *d_a, T *d_b, T *d_c, unsigned int DIM){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float tileA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float tileB[BLOCK_DIM][BLOCK_DIM];

    float sum = 0;

    for (int z = 0; z < DIM; z+=BLOCK_DIM){
        tileA[threadIdx.y][threadIdx.x] = d_a[y * DIM + z + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = d_b[(z + threadIdx.y)* DIM + x];
        __syncthreads();
        for (int i = 0; i < BLOCK_DIM; ++i){
            sum += tileA [threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    d_c[y * DIM + x] = sum;
}

template <class T>
class SCudaMatrix : public AbsMatrix<T>{
    private:
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        SCudaMatrix(unsigned int m, unsigned int n) : AbsMatrix<T>(m, n){}

        //! additional contructor which copies data directly
        SCudaMatrix(unsigned int m, unsigned int n, T* data): AbsMatrix<T>(m, n, data){}

        //! copy constructor; we want to copy real data not references
        SCudaMatrix(const AbsMatrix<T>& other) : AbsMatrix<T>(other){}

        ~SCudaMatrix(){}

        SCudaMatrix operator*(const SCudaMatrix<T> &other){
            SCudaMatrix<T> h_c = SCudaMatrix(this->getHeight(), other.getWidth());

            const unsigned int a_size = this->getHeight() * this->getWidth();
            const unsigned int a_byte = a_size * sizeof(T);
            const unsigned int b_size = other.getHeight() * other.getWidth();
            const unsigned int b_byte = b_size * sizeof(T);
            const unsigned int c_size = h_c.getHeight() * h_c.getWidth();
            const unsigned int c_byte = c_size * sizeof(T);

            T *d_a;//this
            T *d_b;//other
            T *d_c;//result

            cudaMalloc((void **) &d_a, a_byte);
            cudaMalloc((void **) &d_b, b_byte);
            cudaMalloc((void **) &d_c, c_byte);

            cudaMemcpy(d_a, this->_data, a_byte, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, other._data, b_byte, cudaMemcpyHostToDevice);
            
            unsigned int DIM = h_c.getHeight(); 
            dim3 gridDim(DIM/BLOCK_DIM, DIM/BLOCK_DIM);
            dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
            smult_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, DIM);

            cudaMemcpy(h_c._data, d_c, c_byte, cudaMemcpyDeviceToHost);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

            return h_c;
        }
};

#endif

