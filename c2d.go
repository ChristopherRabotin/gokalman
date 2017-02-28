package gokalman

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/gonum/matrix/mat64"
)

// VanLoan computes the F and Q matrices from the provided CT system A, Γ, W and
// the sampling rate Δt.
func VanLoan(A, Γ, W *mat64.Dense, Δt float64) (*mat64.Dense, *mat64.SymDense, error) {
	var err error
	// Check aliasing
	λ := mat64.Eigen{}
	λ.Factorize(A, false)
	λmaxImag := -math.MaxFloat64
	var λmax complex128
	for _, λ := range λ.Values(nil) {
		if im := imag(λ); im > λmaxImag {
			λmax = λ
		}
	}

	if 2*cmplx.Abs(λmax)*Δt >= math.Pi {
		err = fmt.Errorf("gokalman: Nyquist sampling criterion not fulfilled with Δt=%f", Δt)
	}

	// Compute F and Q.
	var ΓW, ΓWΓ, Ap mat64.Dense
	ΓW.Mul(Γ, W)
	ΓWΓ.Mul(&ΓW, Γ.T())
	ΓWΓ.Scale(Δt, &ΓWΓ)
	Ap.Scale(Δt, A)
	// Find the size of the M matrix.
	rA, cA := A.Dims()
	r1, c1 := ΓWΓ.Dims()
	rM := rA + cA
	cM := cA + c1
	M := mat64.NewDense(rM, cM, nil)

	// Populate M
	for i := 0; i < rA; i++ {
		for j := 0; j < cA; j++ {
			M.Set(i, j, -Ap.At(i, j))
			M.Set(i+rA, j+cA, Ap.T().At(i, j))
		}
	}
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			M.Set(i, j+cA, ΓWΓ.At(i, j))
		}
	}

	// Compute exponential
	var expM mat64.Dense
	expM.Exp(M)
	reM, ceM := expM.Dims()

	// Extract F transpose (and F^-1*Q) knowing it has the same size as A.
	F := mat64.NewDense(rA, cA, nil)
	F1Q := mat64.NewDense(rA, cA, nil)
	for i := 0; i < rA; i++ {
		for j := 0; j < cA; j++ {
			F1Q.Set(i, j, expM.At(i, ceM-cA+j))
			F.Set(i, j, expM.At(reM-rA+i, ceM-cA+j))
		}
	}
	F = mat64.DenseCopyOf(F.T())
	var Q mat64.Dense
	Q.Mul(F, F1Q)
	QSym, _ := AsSymDense(&Q)
	return F, QSym, err
}
