#ifndef _BLOCKMT_H_
#define _BLOCKMT_H_

#include "absmatrix.h"
#include "xmmintrin.h"
#define expand_blockt(z, count, data)   c.setEntry(i, j1+count, c.getEntry(i,j1+count) + this->getEntry(i,k1) * other.getEntry(k1,j1+count));

inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*ldb], row1);
     _mm_store_ps(&B[1*ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}

class VecFloatMatrix : public AbsMatrix<float>{
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        BlocktMatrix(unsigned int m, unsigned int n) : AbsMatrix<T>(m, n){}

        //! additional contructor which copies data directly
        BlocktMatrix(unsigned int m, unsigned int n, T* data): AbsMatrix<T>(m, n, data){}

        //! copy constructor; we want to copy real data not references
        BlocktMatrix(const AbsMatrix<T>& other) : AbsMatrix<T>(other){}

        ~BlocktMatrix(){}

        BlocktMatrix operator*(const BlocktMatrix<T> &other){
            BlocktMatrix<T> c = BlocktMatrix(this->getHeight(), other.getWidth());
            for (unsigned int j2 = 0; j2 < c.getHeight(); j2+=4){
                for (unsigned int k2 = 0; k2 < c.getWidth(); k2+=4){
                    
                    float kore[16];
                    float sore[16];
                    float are[16];

                    for (int j=0;j<4;j++){
                        for (int k=0;k<4;k++){
                            kore[4*j+k] = getEntry(j2+j, k2+k);
                            sore[4*j+k] = other.getEntry(k2+k, j2+j);
                        }
                    }
                    for (int j=0;j<4;j++){
                        __m128 res[4];
                        __m128 row = _mm_load_ps(&kore[4*j]);
                        for (int k=0;k<4;k++){
                            __m128 col = _mm_load_ps(&sore[4*k]);
                            res[k] = _mm_mul_ps(row, col);
                        }
                        _MM_TRANSPOSE4_PS(res[0], res[1], res[2], res[3]);
                        for (int k=0;k<3;k++){
                            res[3] = _mm_add_ps(res[3], res[k]);
                        }
                        _mm_store_ps(&(are[j]), res[3])
                    }
                    for (int j=0;j<4;j++){
                        for (int k=0;k<4;k++){
                            c.setentry(j, k, are[4*k+j]);
                        }
                    }
                }
            }
            return c;
        }

};

#endif

