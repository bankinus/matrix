
#ifndef _ABSMATRIX_H_
#define _ABSMATRIX_H_

#include <ostream>
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <cstring>
#include "absmatrix.h"
#include <ctime>
#include <boost/preprocessor/repetition/repeat.hpp>

#ifndef tilefactor
#define tilefactor 4
#endif

#ifndef unrollfactor
#define unrollfactor 4
#endif

template <class T>
class AbsMatrix {
	public:

		//! contructor: takes row (m) and column (n) count as parameters
		AbsMatrix(unsigned int m, unsigned int n) {
        	_height=m;
        	_width=n;
        	_data=new T[ _height * _width ];
            std::memset(_data, 0, sizeof(T) * m * n);
        }
		
		//! additional contructor which copies data directly
		AbsMatrix(unsigned int m, unsigned int n, T* data) {
        	_height=m;
        	_width=n;
        	_data=new T[ _height * _width ];
        	std::memcpy(_data, data, (_height*_width)*sizeof(T));
        }

		//! copy constructor; we want to copy real data not references
    		AbsMatrix(const AbsMatrix<T>& other) {
         	_height=other._height;
        	_width=other._width;
        	_data=new T[ _height * _width ];
    	    std::memcpy(_data, other._data, (_height*_width)*sizeof(T));
        }
		
		//! destructor
		virtual ~AbsMatrix(){
	        _width=0;
	        _height=0;
	        delete[] _data;
        }

		//! returns an entry; remember lines first and indices start with 0
		T getEntry(unsigned int i, unsigned j) const {
            T entry=_data[i*_width+j];
	        return entry;
        }

		//! sets an entry; remember lines first and indices start with 0
		void setEntry(unsigned int i, unsigned int j, T value){
            _data[i*_width+j]=value;
        }

		//! returns the width of the matrix
		inline unsigned int getWidth() const {
            return _width;
        }

		//! returns the height of the matirx
		inline unsigned int getHeight() const {
	        return _height;
        }

		//! prints the matrix
		void print() const;

		//! operator overload for matrix-matrix multiplication
		AbsMatrix<T> operator*(const AbsMatrix<T> &other) {
        	AbsMatrix<T> c = AbsMatrix(this->getHeight(), other.getWidth());
        	for (unsigned int i=0; i<c.getHeight(); i++){
        		for (unsigned int j = 0; j < c.getWidth(); j++){
        			T e = 0;
        			for (unsigned int k = 0; k < _width; k++){
        					asm (
        						"#loop"
        					);
        					e += getEntry(i,k)*other.getEntry(k,j);
        			}
        			c.setEntry(i,j,e);
        		}
        	}
        	return c;
        }

		//! operator overload for matrix-scalar multiplication
		AbsMatrix<T> operator*(const T &other) {
        	AbsMatrix<T> m = *this;
        		for (unsigned int i=0; i<_width; i++){
        			for (unsigned int j=0; j<_height; j++){
        				m.setEntry(i,j,other*getEntry(i,j));
        			}
        		}
        	return m;
        }

		//! operator overload for assignment, we copy real data not references
		AbsMatrix<T> &operator=(const AbsMatrix<T> &other){
        	_height=other._height;
        	_width=other._width;
        	T* newdata=new T[ _height * _width ];
        	std::memcpy(newdata, other._data, (_height*_width)*sizeof(T));
        	delete[] _data;
        	_data=newdata;
        	return *this;
        }

		//! operator overload for comparison
		bool operator==(const AbsMatrix<T> &other);
	protected:
		//! internal representation of the matrix
		T* _data;
		unsigned int _width;
		unsigned int _height;
};

//! output stream operator
template<class T>
std::ostream& operator<<(std::ostream& stream, const AbsMatrix<T> &m) {
	for (unsigned int i = 0; i < m.getHeight(); i++){
		for (unsigned int j = 0; j < m.getWidth(); j++){
			stream << m.getEntry(i,j) << " ";
		}
		stream << std::endl;
	}
	return stream;
}


#endif

