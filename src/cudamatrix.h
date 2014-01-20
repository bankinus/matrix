#ifndef _CUDAMT_H_
#define _CUDAMT_H_

#include "absmatrix.h"
#define expand_blockt(z, count, data)   c.setEntry(i, j1+count, c.getEntry(i,j1+count) + this->getEntry(i,k1) * other.getEntry(k1,j1+count));
template <class T>
__global__
void mult_kernel(T *d_a, T *d_b, T *d_c){
    d_c[blockDim.x*threadIdx.y+threadIdx.x]=0;
    for (int i = 0; i < blockDim.x; i++){
        d_c[blockDim.x*threadIdx.y+threadIdx.x] += d_a[blockDim.x*threadIdx.y+i]
                                                * d_b[blockDim.x*i+threadIdx.x];
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
            mult_kernel<<<1, blockDim>>>(d_a, d_b, d_c);

            cudaMemcpy(h_c._data, d_c, c_byte, cudaMemcpyDeviceToHost);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

            return h_c;
        }
};

#endif

