#include "absmatrix.h"

template <class T>
class RowtMatrix : AbsMatrix<T>{
    inline RowtMatrix<T> operator*(const RowtMatrix<T> &other) {
    	RowtMatrix<T> c = RowtMatrix(this->getHeight(), other.getWidth());
        for (unsigned int j2 = 0; j2 < _width; j2+=tilefactor){
    	    for (unsigned int i=0; i<c.getHeight(); i++){
    	    	for (unsigned int k1 = 0; k1 < c.getWidth(); k1++){
    	    		T e = 0;
    	    		for (unsigned int j1 = j2; (j1 < j2) && (j1 < _width); j1++){
    	    			asm (
    	    				"#loop"
    	    			);
    	    		    c.setEntry(i, j1, c.getEntry(i,j1) + getEntry(i,k1) * other.getEntry(k1,j1));
    	    		}
    	    	}
    	    }
        }
    	return c;
    }
}

