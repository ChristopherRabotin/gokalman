package gokalman

import (
	"fmt"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestSRIFR0(t *testing.T) {

	F, G, Δt := Midterm2Matrices()
	Q := mat64.NewSymDense(3, []float64{2.5e-15, 6.25e-13, (25e-11) / 3, 6.25e-13, (5e-7) / 3, 2.5e-8, (25e-11) / 3, 2.5e-8, 5e-6})
	R := mat64.NewSymDense(1, []float64{0.005 / Δt})
	H := mat64.NewDense(1, 3, []float64{1, 0, 0})
	noise := NewAWGN(Q, R)
	x0 := mat64.NewVector(3, []float64{0, 0.35, 0})
	P0 := ScaledIdentity(3, 10)
	_, est0, err := NewSquareRootInformation(x0, P0, F, G, H, noise)
	if err != nil {
		t.Fatal(err)
	}
	est0.stddev
	// Sanity check

	fmt.Printf("P0:\n%+v\n\nPc:\n%+v", mat64.Formatted(P0))

}
