
#include <cassert>
#include <iostream>
#include <ostream>
#include <cstring>
#include "matrix.h"
#include <ctime>


template<class T>
Matrix<T>::Matrix(unsigned int m, unsigned int n) {
	_height=m;
	_width=n;
	_data=new T[ _height * _width ];
}

template<class T>
Matrix<T>::Matrix(unsigned int m, unsigned int n, T* data) {
	_height=m;
	_width=n;
	_data=new T[ _height * _width ];
	std::memcpy(_data, data, (_height*_width)*sizeof(T));
}

template<class T>
Matrix<T>::Matrix(const Matrix<T> &other) {
	_height=other._height;
	_width=other._width;
	_data=new T[ _height * _width ];
	std::memcpy(_data, other._data, (_height*_width)*sizeof(T));
}

template<class T>
Matrix<T>::~Matrix() {
	_width=0;
	_height=0;
	delete[] _data;
}

template<class T>
inline T Matrix<T>::getEntry(unsigned int i, unsigned int j) const{
	T entry=_data[i*_width+j];
	return entry;
}

template<class T>
inline void Matrix<T>::setEntry(unsigned int i, unsigned int j, T value) {
	_data[i*_width+j]=value;
}

template<class T>
inline unsigned int Matrix<T>::getWidth() const {
	return _width;
}

template<class T>
inline unsigned int Matrix<T>::getHeight() const {
	return _height;
}


template<class T>
inline Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) {
	Matrix<T> c = Matrix(this->getHeight(), other.getWidth());
	for (unsigned int i=0; i<c.getHeight(); i++){
		for (unsigned int j = 0; j < c.getWidth(); j++){
			T e = 0;
			for (unsigned int k = 0; k < _width; k += 4){
					asm (
						"#loop"
					);
					e += getEntry(i, k) * other.getEntry(k, j);
					e += getEntry(i, k + 1) * other.getEntry(k + 1, j);
					e += getEntry(i, k + 2) * other.getEntry(k + 2, j);
					e += getEntry(i, k + 3) * other.getEntry(k + 3, j);
			}
			c.setEntry(i,j,e);
		}
	}
	return c;
}
/*
template<class T>
inline Matrix<T> Matrix<T>::operator*(const T &other) {
	Matrix<T> m = *this;
		for (unsigned int i=0; i<_width; i++){
			for (unsigned int j=0; j<_height; j++){
				m.setEntry(i,j,other*getEntry(i,j));
			}
		}
	return m;
}
*/

template<class T>
inline Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
	_height=other._height;
	_width=other._width;
	T* newdata=new T[ _height * _width ];
	std::memcpy(newdata, other._data, (_height*_width)*sizeof(T));
	delete[] _data;
	_data=newdata;
	return *this;
}

template<class T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T> &m)  {
	for (unsigned int i = 0; i < m.getHeight(); i++){
		for (unsigned int j = 0; j < m.getWidth(); j++){
			stream << m.getEntry(i,j) << " ";
		}
		stream << std::endl;
	}
	return stream;
}

int main(void) {
	for (int dim = 16; dim <= 512; dim += 16){
		struct timespec begin;
                struct timespec end;
		int* d1 = new int[dim * dim];
		int* d2 = new int[dim * dim];
		for (int i = 0; i < dim; i++){
			for (int j = 0; j < dim; j++){
				d1[dim * i + j] = 1;
				d2[dim * i + j] = 1;
			}
		}
		Matrix<int> m1(dim, dim, d1);
		Matrix<int> m2(dim, dim, d2);
		Matrix<int> m3(dim, dim);
                clock_gettime(CLOCK_MONOTONIC, &begin);
		for(int i=0; i<10; i++){
			m3 = m1 * m2;
		}
		clock_gettime(CLOCK_MONOTONIC, &end);
                time_t sec;
                long nsec;
                sec = end.tv_sec - begin.tv_sec;
                nsec = end.tv_nsec - begin.tv_nsec;
                nsec /= 1000000;
                sec *= 1000;
		//std::cout << "dim: " << dim << " ms: ";
		std::cout << sec + nsec << "\t";
	}
	std::cout << std::endl;
	return 0;
}

