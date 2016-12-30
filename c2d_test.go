package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestVanLoan(t *testing.T) {
	A := mat64.NewDense(2, 2, []float64{0, 1, 0, 0})
	Γ := mat64.NewDense(2, 1, []float64{0, 1})
	W := mat64.NewDense(1, 1, []float64{1})
	F, Q := VanLoan(A, Γ, W, 0.1)
	Fexp := mat64.NewDense(2, 2, []float64{1, 0.1, 0, 1})
	Qexp := mat64.NewSymDense(2, []float64{0.0003, 0.005, 0.005, 0.1})

	if !mat64.EqualApprox(F, Fexp, 1e-3) {
		t.Fatal("F incorrectly computed")
	}

	if !mat64.EqualApprox(Q, Qexp, 1e-3) {
		t.Fatal("Q incorrectly computed")
	}
}
