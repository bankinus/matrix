#ifndef _VECM_H_
#define _VECM_H_

#include "absmatrix.h"
#include <xmmintrin.h>
#include <immintrin.h>
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

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
__t0 = _mm256_unpacklo_ps(row0, row1);
__t1 = _mm256_unpackhi_ps(row0, row1);
__t2 = _mm256_unpacklo_ps(row2, row3);
__t3 = _mm256_unpackhi_ps(row2, row3);
__t4 = _mm256_unpacklo_ps(row4, row5);
__t5 = _mm256_unpackhi_ps(row4, row5);
__t6 = _mm256_unpacklo_ps(row6, row7);
__t7 = _mm256_unpackhi_ps(row6, row7);
__tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
__tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
__tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
__tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
__tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
__tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
__tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
__tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

class SSEVecFloatMatrix : public AbsMatrix<float>{
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        SSEVecFloatMatrix(unsigned int m, unsigned int n) : AbsMatrix<float>(m, n){}

        //! additional contructor which copies data directly
        SSEVecFloatMatrix(unsigned int m, unsigned int n, float* data): AbsMatrix<float>(m, n, data){}

        //! copy constructor; we want to copy real data not references
        SSEVecFloatMatrix(const AbsMatrix<float>& other) : AbsMatrix<float>(other){}

        ~SSEVecFloatMatrix(){}

        SSEVecFloatMatrix operator*(const SSEVecFloatMatrix &other){
            SSEVecFloatMatrix c = SSEVecFloatMatrix(this->getHeight(), other.getWidth());
            for (unsigned int j2 = 0; j2 < c.getHeight(); j2+=4){
                for (unsigned int k2 = 0; k2 < c.getWidth(); k2+=4){
                    for (unsigned int i = 0; i < c.getHeight(); i+=4){
                        float kore[16];
                        float sore[16];
                        float are[16];
                        /* fetch block */
                        for (int j=0;j<4;j++){
                            for (int k=0;k<4;k++){
                                kore[4*j+k] = getEntry(j2+j, k2+k);
                                sore[4*j+k] = other.getEntry(k2+k, i+j);
                            }
                        }
                        /* calculate */
                        for (int j=0;j<4;j++){
                            __m128 res[4];
                            __m128 row = _mm_load_ps(kore+(4*j));
                            for (int k=0;k<4;k++){
                                __m128 col = _mm_load_ps(sore+(4*k));
                                res[k] = _mm_mul_ps(row, col);
                            }
                            _MM_TRANSPOSE4_PS(res[0], res[1], res[2], res[3]);
                            for (int k=0;k<3;k++){
                                res[3] = _mm_add_ps(res[3], res[k]);
                            }
                            _mm_store_ps(&are[j*4], res[3]);
                        }
                        /* write results */
                        for (int j=0;j<4;j++){
                            for (int k=0;k<4;k++){
                                c.setEntry(j, k, (c.getEntry(j, k) + are[4*k+j]) );
                            }
                        }
                    }
                }
            }
            return c;
        }

};

class AVXVecFloatMatrix : public AbsMatrix<float>{
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        AVXVecFloatMatrix(unsigned int m, unsigned int n) : AbsMatrix<float>(m, n){}

        //! additional contructor which copies data directly
        AVXVecFloatMatrix(unsigned int m, unsigned int n, float* data): AbsMatrix<float>(m, n, data){}

        //! copy constructor; we want to copy real data not references
        AVXVecFloatMatrix(const AbsMatrix<float>& other) : AbsMatrix<float>(other){}

        ~AVXVecFloatMatrix(){}

        AVXVecFloatMatrix operator*(const AVXVecFloatMatrix &other){
            AVXVecFloatMatrix c = AVXVecFloatMatrix(this->getHeight(), other.getWidth());
            for (unsigned int j2 = 0; j2 < c.getHeight(); j2+=8){
                for (unsigned int k2 = 0; k2 < c.getWidth(); k2+=8){
                    for (unsigned int i = 0; i < c.getHeight(); i+=8){
                        float kore[64];
                        float sore[64];
                        float are[64];
                        /* fetch block */
                        for (int j=0;j<8;j++){
                            for (int k=0;k<8;k++){
                                kore[8*j+k] = getEntry(j2+j, k2+k);
                                sore[8*j+k] = other.getEntry(k2+k, i+j);
                            }
                        }
                        /* calculate */
                        for (int j=0;j<8;j++){
                            __m256 res[8];
                            __m256 row = _mm256_load_ps(kore+(8*j));
                            for (int k=0;k<8;k++){
                                __m256 col = _mm256_load_ps(sore+(8*k));
                                res[k] = _mm256_mul_ps(row, col);
                            }
                            transpose8_ps(  res[0], res[1], res[2], res[3],
                                            res[4], res[5], res[6], res[7]);
                            for (int k=0;k<7;k++){
                                res[7] = _mm256_add_ps(res[7], res[k]);
                            }
                            _mm256_store_ps(&are[j*8], res[7]);
                        }
                        /* write results */
                        for (int j=0;j<8;j++){
                            for (int k=0;k<8;k++){
                                c.setEntry(j, k, (c.getEntry(j, k) + are[4*k+j]) );
                            }
                        }
                    }
                }
            }
            return c;
        }

};

#endif

