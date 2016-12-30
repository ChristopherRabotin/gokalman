package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestVanLoan(t *testing.T) {
	A := mat64.NewDense(2, 2, []float64{0, 1, 0, 0})
	Γ := mat64.NewDense(2, 1, []float64{0, 1})
	W := mat64.NewDense(1, 1, []float64{1})
	F, Q, err := VanLoan(A, Γ, W, 0.1)
	if err != nil {
		t.Fatalf("error is not nil: %s", err)
	}
	Fexp := mat64.NewDense(2, 2, []float64{1, 0.1, 0, 1})
	Qexp := mat64.NewSymDense(2, []float64{0.0003, 0.005, 0.005, 0.1})

	if !mat64.EqualApprox(F, Fexp, 1e-3) {
		t.Fatal("F incorrectly computed")
	}

	if !mat64.EqualApprox(Q, Qexp, 1e-3) {
		t.Fatal("Q incorrectly computed")
	}

	// Check in the output that a warning is displayed
	_, _, err = VanLoan(mat64.NewDense(2, 2, []float64{1, 1, 0, 1}), Γ, W, 10)
	if err == nil {
		t.Fatal("error not set for matrix which does not fulfill Nyquist")
	}
}
