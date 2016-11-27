package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestCheckDims(t *testing.T) {
	i22 := mat64.NewDense(2, 2, []float64{1, 0, 0, 1})
	i33 := mat64.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
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
