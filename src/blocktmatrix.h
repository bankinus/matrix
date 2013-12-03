#ifndef _BLOCKMT_H_
#define _BLOCKMT_H_

#include "absmatrix.h"

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
            for (unsigned int k2 = 0; k2 < this->_width; k2+=tilefactor){
                for (unsigned int j2 = 0; j2 < this->_width; j2+=tilefactor){
        	        for (unsigned int i=0; i<c.getHeight(); i++){
        	    	    for (unsigned int k1 = k2; (k1 < k2) && (k1 < c.getWidth()); k1++){
        	    		    for (unsigned int j1 = j2; (j1 < j2) && (j1 < this->_width); j1++){
        	    			    asm (
        	    			    	"#loop"
        	    			    );
        	    		        c.setEntry(i, j1, c.getEntry(i,j1) + this->getEntry(i,k1) * other.getEntry(k1,j1));
        	    		    }
        	    	    }
        	        }
                }
            }
            return c;
        }

};

#endif

