package gokalman

import (
	"errors"
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// Identity returns an identity matrix of the provided size.
func Identity(n int) mat64.Symmetric {
	vals := make([]float64, n*n)
	for j := 0; j < n*n; j++ {
		if j%(n+1) == 0 {
			vals[j] = 1
		} else {
			vals[j] = 0
		}
	}
	return mat64.NewSymDense(n, vals)
}

// IsNil returns whether the provided matrix only has zero values
func IsNil(m mat64.Matrix) bool {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if m.At(i, j) != 0 {
				return false
			}
		}
	}
	return true
}

// AsSymDense attempts return a SymDense from the provided Dense.
func AsSymDense(m *mat64.Dense) (*mat64.SymDense, error) {
	r, c := m.Dims()
	if r != c {
		return nil, errors.New("matrix must be square")
	}
	mT := m.T()
	vals := make([]float64, r*c)
	idx := 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if mT.At(i, j) != m.At(i, j) {
				return nil, errors.New("matrix is not symmetric")
			}
			vals[idx] = m.At(i, j)
			idx++
		}
	}

	return mat64.NewSymDense(r, vals), nil
}

// DimensionAgreement defines how two matrices' dimensions should agree.
type DimensionAgreement uint8

const (
	dimErrMsg                    = "dimensions must agree: "
	rows2cols DimensionAgreement = iota + 1
	cols2rows
	cols2cols
	rows2rows
	rowsAndcols
)

// checkMatDims checks the matrix dimensions match provided a DimensionAgreement. Returns an error if not.
func checkMatDims(m1, m2 mat64.Matrix, name1, name2 string, method DimensionAgreement) error {
	r1, c1 := m1.Dims()
	r2, c2 := m2.Dims()
	switch method {
	case rows2cols:
		if r1 != c2 {
			return fmt.Errorf("%s%s(%dx...) %s(...x%d)", dimErrMsg, name1, r1, name2, c2)
		}
		break
	case cols2rows:
		if c1 != r2 {
			return fmt.Errorf("%s%s(...x%d) %s(%dx...)", dimErrMsg, name1, c1, name2, r2)
		}
		break
	case cols2cols:
		if c1 != c2 {
			return fmt.Errorf("%s%s(...x%d) %s(...x%d)", dimErrMsg, name1, c1, name2, c2)
		}
		break
	case rows2rows:
		if r1 != r2 {
			return fmt.Errorf("%s%s(%dx...) %s(%dx...)", dimErrMsg, name1, r1, name2, r2)
		}
		break
	case rowsAndcols:
		if c1 != c2 || r1 != r2 {
			return fmt.Errorf("%s%s(%dx%d) %s(%dx%d)", dimErrMsg, name1, r1, c1, name2, r2, c2)
		}
		break
	}
	return nil
}
