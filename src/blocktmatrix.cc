#include "absmatrix.h"

template <class T>
class BlocktMatrix : AbsMatrix<T>{
   template<class T>
    inline AbsMatrix<T> AbsMatrix<T>::operator*(const AbsMatrix<T> &other) {
    	AbsMatrix<T> c = AbsMatrix(this->getHeight(), other.getWidth());
        for (unsigned int k2 = 0; k2 < _width; k2+=tilefactor){
            for (unsigned int j2 = 0; j2 < _width; j2+=tilefactor){
    	        for (unsigned int i=0; i<c.getHeight(); i++){
    	    	    for (unsigned int k1 = k2; (k1 < k2) && (k1 < c.getWidth()); k1++){
    	    		    T e = 0;
    	    		    for (unsigned int j1 = j2; (j1 < j2) && (j1 < _width); j1++){
    	    			    asm (
    	    			    	"#loop"
    	    			    );
    	    		        c.setEntry(i, j, c.getEntry(i,j) + getEntry(i,k) * other.getEntry(k,j));
    	    		    }
    	    	    }
    	        }
            }
        }
    	return c;

    }
}

