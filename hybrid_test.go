package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestHybridBasic(t *testing.T) {
	prevXHat := mat64.NewVector(6, nil)
	prevP := mat64.NewSymDense(6, nil)
	var covarDistance float64 = 50
	var covarVelocity float64 = 1
	for i := 0; i < 3; i++ {
		prevP.SetSym(i, i, covarDistance)
		prevP.SetSym(i+3, i+3, covarVelocity)
	}

	Q := mat64.NewSymDense(6, nil)
	R := mat64.NewSymDense(2, []float64{1e-3, 0, 0, 1e-6})
	noiseKF := NewNoiseless(Q, R)

	hkf, _, err := NewHybridKF(prevXHat, prevP, noiseKF, 2)
	if err != nil {
		t.Fatalf("%s", err)
	}
	// Check that calling Update before "Prepare" returns an error
	_, err = hkf.Update(mat64.NewVector(2, nil), mat64.NewVector(2, nil))
	if err == nil {
		t.Fatal("error should not have been nil when calling Update before Prepare")
	}

	hkf.EnableEKF()
	if hkf.ekfMode == false || !hkf.EKFEnabled() {
		t.Fatal("the KF is still in CKF mode after EKF switch")
	}
}
