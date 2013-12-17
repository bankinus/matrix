#ifndef _BLOCKMT_H_
#define _BLOCKMT_H_

#include "absmatrix.h"
#include <thread>
#define expand_blockt(z, count, data)   c.setEntry(i, j1+count, c.getEntry(i,j1+count) + this->getEntry(i,k1) * other.getEntry(k1,j1+count));
#define expand_blockthread(z, count, data)   c->setEntry(i, j1+count, c->getEntry(i,j1+count) + this->getEntry(i,k1) * other.getEntry(k1,j1+count));
template <class T>

class OpenMPMatrix : public AbsMatrix<T>{
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        OpenMPMatrix(unsigned int m, unsigned int n) : AbsMatrix<T>(m, n){}

        //! additional contructor which copies data directly
        OpenMPMatrix(unsigned int m, unsigned int n, T* data): AbsMatrix<T>(m, n, data){}

        //! copy constructor; we want to copy real data not references
        OpenMPMatrix(const AbsMatrix<T>& other) : AbsMatrix<T>(other){}

        ~OpenMPMatrix(){}

        OpenMPMatrix operator*(const OpenMPMatrix<T> &other){
            OpenMPMatrix<T> c = OpenMPMatrix(this->getHeight(), other.getWidth());
            for (unsigned int k2 = 0; k2 < c.getHeight(); k2+=tilefactor){
                for (unsigned int j2 = 0; j2 < c.getWidth(); j2+=tilefactor){
                    #pragma omp parallel for
        	        for (unsigned int i=0; i<c.getHeight(); i++){
        	    	    for (unsigned int k1 = k2; (k1 < k2+tilefactor) && (k1 < c.getWidth()); k1++){
        	    		    for (unsigned int j1 = j2; (j1 < j2+tilefactor) && (j1 < c.getHeight()); j1+=unrollfactor){
        	    			    asm (
        	    			    	"#loop"
        	    			    );
        	    		        //c.setEntry(i, j1, c.getEntry(i,j1) + this->getEntry(i,k1) * other.getEntry(k1,j1));
                                BOOST_PP_REPEAT(unrollfactor, expand_blockt, data);
        	    		    }
        	    	    }
        	        }
                }
            }
            return c;
        }

};

template <class T>

class ThreadMatrix : public AbsMatrix<T>{
    protected:
        unsigned int threads;
        std::thread* threadarray;
        void computeBlock(unsigned int j, const ThreadMatrix<T> &other, ThreadMatrix<T> *c){
            for (unsigned int k2 = 0; k2 < c->getHeight(); k2+=tilefactor){
                for (unsigned int j2 = j; j2 < c->getWidth(); j2+=tilefactor*threads){
        	        for (unsigned int i=0; i<c->getHeight(); i++){
        	    	    for (unsigned int k1 = k2; (k1 < k2+tilefactor) && (k1 < c->getWidth()); k1++){
        	    		    for (unsigned int j1 = j2; (j1 < j2+tilefactor) && (j1 < c->getHeight()); j1+=unrollfactor){
        	    			    asm (
        	    			    	"#loop"
        	    			    );
        	    		        //c->setEntry(i, j1, c->getEntry(i,j1) + this->getEntry(i,k1) * other.getEntry(k1,j1));
                                BOOST_PP_REPEAT(unrollfactor, expand_blockthread, data);
        	    		    }
        	    	    }
        	        }
                }
            }
        }
    public:
        //! contructor: takes row (m) and column (n) count as parameters
        ThreadMatrix(unsigned int m, unsigned int n) : AbsMatrix<T>(m, n), threads(1), threadarray(new std::thread[threads]){}
        ThreadMatrix(unsigned int m, unsigned int n, unsigned int t) : AbsMatrix<T>(m, n), threads(t), threadarray(new std::thread[threads]){}

        //! additional contructor which copies data directly
        ThreadMatrix(unsigned int m, unsigned int n, T* data): AbsMatrix<T>(m, n, data), threads(1), threadarray(new std::thread[threads]){}
        ThreadMatrix(unsigned int m, unsigned int n, T* data, unsigned int t): AbsMatrix<T>(m, n, data), threads(t), threadarray(new std::thread[threads]){}

        //! copy constructor; we want to copy real data not references
        ThreadMatrix(const AbsMatrix<T>& other) : AbsMatrix<T>(other), threads(other.threads), threadarray(new std::thread[threads]){}

        ~ThreadMatrix(){}

        ThreadMatrix operator*(const ThreadMatrix<T> &other){
            ThreadMatrix<T> c = ThreadMatrix(this->getHeight(), other.getWidth());
            for (unsigned int j=0;j<threads;j++){
                threadarray[j]=std::thread(&ThreadMatrix::computeBlock, this, j*tilefactor, other, &c);
            }
            for (unsigned int j=0;j<threads;j++){
                threadarray[j].join();
            }
            return c;
        }

};
#endif

