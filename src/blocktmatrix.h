#ifndef _BLOCKMT_H_
#define _BLOCKMT_H_

#include "absmatrix.h"
#define expand_blockt(z, count, data)   c.setEntry(i, j1+count, c.getEntry(i,j1+count) + this->getEntry(i,k1) * other.getEntry(k1,j1+count));
template <class T>

class BlocktMatrix : public AbsMatrix<T>{
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
            for (unsigned int k2 = 0; k2 < c.getHeight(); k2+=tilefactor){
                for (unsigned int j2 = 0; j2 < c.getWidth(); j2+=tilefactor){
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

#endif

