#ifndef _ROWTM_H_
#define _ROWTM_H_

#include "absmatrix.h"
#define expand_rowt(z, count, data)   c.setEntry(i, j1+count, c.getEntry(i,j1+count) + this->getEntry(i,k1) * other.getEntry(k1,j1+count));

template <class T>
class RowtMatrix : public AbsMatrix<T>{
    public:
        //! contructor: takes row (m) and column (n) count as parameters
		RowtMatrix(unsigned int m, unsigned int n) : AbsMatrix<T>(m, n){}
		
		//! additional contructor which copies data directly
		RowtMatrix(unsigned int m, unsigned int n, T* data): AbsMatrix<T>(m, n, data){}

		//! copy constructor; we want to copy real data not references
	    RowtMatrix(const AbsMatrix<T>& other) : AbsMatrix<T>(other){}

        ~RowtMatrix(){}

        RowtMatrix operator*(const RowtMatrix<T> &other){
            RowtMatrix<T> c = RowtMatrix(this->getHeight(), other.getWidth());
            for (unsigned int j2 = 0; j2 < this->_width; j2+=tilefactor){
        	    for (unsigned int i=0; i<c.getHeight(); i++){
        	    	for (unsigned int k1 = 0; k1 < c.getWidth(); k1++){
        	    		for (unsigned int j1 = j2; (j1 < j2+tilefactor) && (j1 < this->_width); j1+=unrollfactor){
        	    			asm (
        	    				"#loop"
        	    			);
        	    		    //c.setEntry(i, j1, c.getEntry(i,j1) + this->getEntry(i,k1) * other.getEntry(k1,j1));
                            BOOST_PP_REPEAT(unrollfactor, expand_rowt, data);
        	    		}
        	    	}
        	    }
            }
            return c;
        }
};

#endif

