package gokalman

import (
	"fmt"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func Robot1DMatrices() (F, G mat64.Matrix, Δt float64) {
	Δt = 0.1
	F = mat64.NewDense(2, 2, []float64{1, Δt, 0, 1})
	G = mat64.NewDense(2, 1, []float64{0.5 * Δt * Δt, Δt})
	return
}

func Midterm2Matrices() (F, G mat64.Matrix, Δt float64) {
	Δt = 0.01
	F = mat64.NewDense(3, 3, []float64{1, 0.01, 5e-5, 0, 1, 0.01, 0, 0, 1})
	G = mat64.NewDense(3, 1, []float64{(5e-7) / 3, 5e-5, 0.01})
	return
}

func assertPanic(t *testing.T, f func()) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("code did not panic")
		}
	}()
	f()
}

func TestIsNil(t *testing.T) {
	if !IsNil(nil) {
		t.Fatal("nil said to not be nil")
	}
	if IsNil(Identity(2)) {
		t.Fatal("i22 said to be nil")
	}
	if !IsNil(mat64.NewSymDense(2, []float64{0, 0, 0, 0})) {
		t.Fatal("zeros 4x4 said to NOT be nil")
	}
}

func TestIdentity(t *testing.T) {
	n := 3
	i33s := Identity(n)
	i33d := DenseIdentity(n)
	for _, i33 := range []mat64.Matrix{i33s, i33d} {
		if r, c := i33.Dims(); r != n || r != c {
			t.Fatalf("i11 has dimensions (%dx%d)", r, c)
		}
		for i := 0; i < n; i++ {
			if i33.At(i, i) != 1 {
				t.Fatalf("i33(%d,%d) != 1", i, i)
			}
			for j := 0; j < n; j++ {
				if i != j && i33.At(i, j) != 0 {
					t.Fatalf("i33(%d,%d) != 0", i, j)
				}
			}
		}
	}
}

func TestAsSymDense(t *testing.T) {
	d := mat64.NewDense(3, 3, []float64{1, 0.1, 2, 0.1, 3, 5, 2, 5, 7})
	dsym, err := AsSymDense(d)
	if err != nil {
		t.Fatal("AsSymDense failed on 3x3")
	}
	r, c := d.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if i == j {
				continue
			}
			if dsym.At(i, j) != d.At(i, j) {
				t.Fatalf("returned symmetric matrix invalid (%d,%d): %+v %+v", i, j, dsym, d)
			}
		}
	}
	_, err = AsSymDense(mat64.NewDense(3, 3, []float64{1, 0, 3, 0, 1, 0, 1, 2, 1}))
	if err == nil {
		t.Fatal("non symmetric matrix did not fail")
	}

	_, err = AsSymDense(mat64.NewDense(2, 3, []float64{1, 0, 1, 1, 2, 3}))
	if err == nil {
		t.Fatal("non square matrix did not fail")
	}
}

func TestCheckDims(t *testing.T) {
	i22 := Identity(2)
	i33 := Identity(3)
	methods := []DimensionAgreement{rows2cols, cols2rows, cols2cols, rows2rows, rowsAndcols}
	for _, meth := range methods {
		if err := checkMatDims(i22, i22, "i22", "i22", meth); err != nil {
			t.Fatalf("method %+v fails: %s", meth, err)
		}
		if err := checkMatDims(i22, i33, "i22", "i33", meth); err == nil {
			t.Fatalf("method %+v does not error when using i22 and i33 ", meth)
		}
	}
}

func TestHouseholderTransf(t *testing.T) {
	A := mat64.NewDense(3, 3, []float64{1, -2, -1, 2, -1, 1, 1, 1, 2})
	HouseholderTransf(A, 2, 1)
	fmt.Printf("%+v\n", mat64.Formatted(A))
}
