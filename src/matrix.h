
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <ostream>
#include <stdlib.h>


template <class T>
class Matrix {
	public:

		//! contructor: takes row (m) and column (n) count as parameters
		Matrix(unsigned int m, unsigned int n);
		
		//! additional contructor which copies data directly
		Matrix(unsigned int m, unsigned int n, T* data);

		//! copy constructor; we want to copy real data not references
		Matrix(const Matrix<T>& other);
		
		//! destructor
		~Matrix();

		//! returns an entry; remember lines first and indices start with 0
		T getEntry(unsigned int i, unsigned j) const;

		//! sets an entry; remember lines first and indices start with 0
		void setEntry(unsigned int i, unsigned int j, T value);

		//! returns the width of the matrix
		inline unsigned int getWidth() const;

		//! returns the height of the matirx
		inline unsigned int getHeight() const;

		//! prints the matrix
		void print() const;

		//! operator overload for matrix-matrix multiplication
		Matrix<T> operator*(const Matrix<T> &other);

		//! operator overload for matrix-scalar multiplication
		Matrix<T> operator*(const T &other);

		//! operator overload for assignment, we copy real data not references
		Matrix<T> &operator=(const Matrix<T> &other);

		//! operator overload for comparison
		bool operator==(const Matrix<T> &other);
	private:
		//! internal representation of the matrix
		T* _data;
		unsigned int _width;
		unsigned int _height;
};

//! output stream operator
template<class T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T> &m) ;


//! predefined types, not neccessarily requiered
typedef Matrix<double> matd;
typedef Matrix<float> matf;
typedef Matrix<int> mati;
typedef Matrix<short> mats;

#endif

