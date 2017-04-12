package gokalman

import (
	"math"
	"testing"

	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

func TestSRIFR0(t *testing.T) {
	x0 := mat64.NewVector(3, []float64{0, 0.35, 0})
	P0 := ScaledIdentity(3, 10)
	Q := mat64.NewSymDense(6, nil)
	R := mat64.NewSymDense(2, []float64{math.Pow(5e-3, 2), 0, 0, math.Pow(5e-6, 2)})
	noise := NewNoiseless(Q, R)
	_, est0, err := NewSRIF(x0, P0, 3, true, noise)
	if err != nil {
		t.Fatal(err)
	}

	if !mat64.Equal(est0.Covariance(), P0) {
		t.Fatalf("difference in P0 and computed covariance:\n%+v\n%+v", mat64.Formatted(P0), mat64.Formatted(est0.Covariance()))
	}

	mat64.NewTriDense(3, matrix.Upper, nil)
	t.Logf("\n%+v\n", mat64.Formatted(est0.matR0))

}
