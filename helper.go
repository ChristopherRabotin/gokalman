package gokalman

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// ScaledIdentity returns an identity matrix time a scaling factor of the provided size.
func ScaledIdentity(n int, s float64) *mat64.SymDense {
	vals := make([]float64, n*n)
	for j := 0; j < n*n; j++ {
		if j%(n+1) == 0 {
			vals[j] = s
		} else {
			vals[j] = 0
		}
	}
	return mat64.NewSymDense(n, vals)
}

// DenseIdentity returns an identity matrix of type Dense and of the provided size.
func DenseIdentity(n int) *mat64.Dense {
	return ScaledDenseIdentity(n, 1)
}

// ScaledDenseIdentity returns an identity matrix time of type Dense a scaling factor of the provided size.
func ScaledDenseIdentity(n int, s float64) *mat64.Dense {
	vals := make([]float64, n*n)
	for j := 0; j < n*n; j++ {
		if j%(n+1) == 0 {
			vals[j] = s
		} else {
			vals[j] = 0
		}
	}
	return mat64.NewDense(n, n, vals)
}

// Identity returns an identity matrix of the provided size.
func Identity(n int) *mat64.SymDense {
	return ScaledIdentity(n, 1)
}

// IsNil returns whether the provided matrix only has zero values
func IsNil(m mat64.Matrix) bool {
	if m == nil {
		return true
	}
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
			if i != j && !floats.EqualWithinAbsOrRel(mT.At(i, j), m.At(i, j), 1e-6, 1e-2) {
				return nil, fmt.Errorf("matrix is not symmetric (%d, %d): %.40f != %.40f\n%v", i, j, mT.At(i, j), m.At(i, j), mat64.Formatted(m, mat64.Prefix("")))
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

// Sign returns the sign of a given number.
func Sign(v float64) float64 {
	if floats.EqualWithinAbs(v, 0, 1e-12) {
		return 1
	}
	return v / math.Abs(v)
}

// HouseholderSRIF prepare the matrix and performs the Householder transformation.
func HouseholderSRIF(R, H *mat64.Dense, b, y *mat64.Vector) (*mat64.Dense, error) {
	if err := checkMatDims(R, H, "R", "H", cols2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(R, b, "R", "b", rows2rows); err != nil {
		return nil, err
	}
	if err := checkMatDims(H, y, "H", "y", rows2rows); err != nil {
		return nil, err
	}
	n, _ := b.Dims()
	m, _ := y.Dims()
	A0 := mat64.NewDense(m+n, n, nil)
	A0.Stack(R, H)
	col := mat64.NewVector(m+n, nil)
	for i := 0; i < m+n; i++ {
		if i < n {
			col.SetVec(i, b.At(i, 0))
		} else {
			col.SetVec(i, y.At(i-n, 0))
		}
	}
	A := mat64.NewDense(m+n, n+1, nil)
	A.Augment(A0, col)

	HouseholderTransf(A, n, m)

	return A, nil
}

// HouseholderTransf performs the Householder transformation of the given A matrix.
// Changes are done directly in the provided matrix.
func HouseholderTransf(A *mat64.Dense, n, m int) {
	// TODO: Check indexing!
	for k := 0; k < n; k++ {
		//sigma := sign(A.At(k, k))
		sigma := 0.0
		for i := 0; i < m+n; i++ {
			sigma += math.Pow(A.At(i, k), 2)
		}
		sigma = math.Sqrt(sigma) * Sign(A.At(k, k))
		uk := A.At(k, k) + sigma
		A.Set(k, k, -sigma)
		u := make([]float64, m+n-k-1)
		for i := k + 1; i < m+n; i++ {
			u[i-k-1] = A.At(i, k)
		}
		beta := 1 / (sigma * uk)

		for j := k + 1; j < n+1; j++ {
			gamma := 0.0
			for i := k; i < m+n; i++ {
				// Should I simply replace u[i] with A.At(i, k) ? Also seems like there won't be enough u[i]s.
				//gamma += u[i] * A.At(i, j)
				gamma += A.At(i, k) * A.At(i, j)
			}
			gamma *= beta
			for i := k; i < m+n; i++ {
				A.Set(i, j, A.At(i, j)-gamma*A.At(i, k))
			}
			// Next j?!
			for i := k + 1; i < m+n; i++ {
				A.Set(i, k, 0)
			}
		}

	}
}
