
#ifndef _ABSMATRIX_H_
#define _ABSMATRIX_H_

#include <ostream>
#include <stdlib.h>

#define tilefactor 16

template <class T>
class AbsMatrix {
	public:

		//! contructor: takes row (m) and column (n) count as parameters
		AbsMatrix(unsigned int m, unsigned int n);
		
		//! additional contructor which copies data directly
		AbsMatrix(unsigned int m, unsigned int n, T* data);

		//! copy constructor; we want to copy real data not references
		AbsMatrix(const AbsMatrix<T>& other);
		
		//! destructor
		~AbsMatrix();

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
		AbsMatrix<T> operator*(const AbsMatrix<T> &other);

		//! operator overload for matrix-scalar multiplication
		AbsMatrix<T> operator*(const T &other);

		//! operator overload for assignment, we copy real data not references
		AbsMatrix<T> &operator=(const AbsMatrix<T> &other);

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
std::ostream& operator<<(std::ostream& stream, const AbsMatrix<T> &m) ;


//! predefined types, not neccessarily requiered
typedef AbsMatrix<double> matd;
typedef AbsMatrix<float> matf;
typedef AbsMatrix<int> mati;
typedef AbsMatrix<short> mats;

#endif

