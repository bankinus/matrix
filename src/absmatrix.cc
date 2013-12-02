
#include <cassert>
#include <iostream>
#include <ostream>
#include <cstring>
#include "absmatrix.h"
#include <ctime>


template<class T>
AbsMatrix<T>::AbsMatrix(unsigned int m, unsigned int n) {
	_height=m;
	_width=n;
	_data=new T[ _height * _width ];
}

template<class T>
AbsMatrix<T>::AbsMatrix(unsigned int m, unsigned int n, T* data) {
	_height=m;
	_width=n;
	_data=new T[ _height * _width ];
	std::memcpy(_data, data, (_height*_width)*sizeof(T));
}

template<class T>
AbsMatrix<T>::AbsMatrix(const AbsMatrix<T> &other) {
	_height=other._height;
	_width=other._width;
	_data=new T[ _height * _width ];
	std::memcpy(_data, other._data, (_height*_width)*sizeof(T));
}

template<class T>
AbsMatrix<T>::~AbsMatrix() {
	_width=0;
	_height=0;
	delete[] _data;
}

template<class T>
inline T AbsMatrix<T>::getEntry(unsigned int i, unsigned int j) const{
	T entry=_data[i*_width+j];
	return entry;
}

template<class T>
inline void AbsMatrix<T>::setEntry(unsigned int i, unsigned int j, T value) {
	_data[i*_width+j]=value;
}

template<class T>
inline unsigned int AbsMatrix<T>::getWidth() const {
	return _width;
}

template<class T>
inline unsigned int AbsMatrix<T>::getHeight() const {
	return _height;
}


template<class T>
inline AbsMatrix<T> AbsMatrix<T>::operator*(const AbsMatrix<T> &other) {
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
/*
template<class T>
inline AbsMatrix<T> AbsMatrix<T>::operator*(const T &other) {
	AbsMatrix<T> m = *this;
		for (unsigned int i=0; i<_width; i++){
			for (unsigned int j=0; j<_height; j++){
				m.setEntry(i,j,other*getEntry(i,j));
			}
		}
	return m;
}
*/

template<class T>
inline AbsMatrix<T> &AbsMatrix<T>::operator=(const AbsMatrix<T> &other) {
	_height=other._height;
	_width=other._width;
	T* newdata=new T[ _height * _width ];
	std::memcpy(newdata, other._data, (_height*_width)*sizeof(T));
	delete[] _data;
	_data=newdata;
	return *this;
}

template<class T>
std::ostream& operator<<(std::ostream& stream, const AbsMatrix<T> &m)  {
	for (unsigned int i = 0; i < m.getHeight(); i++){
		for (unsigned int j = 0; j < m.getWidth(); j++){
			stream << m.getEntry(i,j) << " ";
		}
		stream << std::endl;
	}
	return stream;
}

