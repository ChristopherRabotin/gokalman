package gokalman

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func Robot1DMatrices() (F, G mat64.Matrix, Δt float64) {
	Δt = 0.1
	F = mat64.NewDense(2, 2, []float64{1, Δt, 0, 1})
	G = mat64.NewDense(2, 1, []float64{0.5 * Δt * Δt, Δt})
	return
}

func TestNewVanillaErrors(t *testing.T) {
	F, G, _ := Robot1DMatrices()
	H := mat64.NewDense(2, 2, nil)
	x0 := mat64.NewVector(2, nil)
	Covar0 := mat64.NewSymDense(3, nil)
	if _, err := NewVanilla(x0, Covar0, F, G, H, Noiseless{}); err == nil {
		t.Fatal("x0 and Covar0 of incompatible sizes does not fail")
	}
	x0 = mat64.NewVector(3, nil)
	if _, err := NewVanilla(x0, Covar0, F, G, H, Noiseless{}); err == nil {
		t.Fatal("F and Covar0 of incompatible sizes does not fail")
	}
	x0 = mat64.NewVector(2, nil)
	Covar0 = mat64.NewSymDense(2, nil)
	H = mat64.NewDense(3, 3, nil)
	if _, err := NewVanilla(x0, Covar0, F, G, H, Noiseless{}); err == nil {
		t.Fatal("H and x0 of incompatible sizes does not fail")
	}
}

func TestVanilla(t *testing.T) {
	F, G, Δt := Robot1DMatrices()
	noise := NewAWGN(mat64.NewSymDense(2, []float64{0.0003, 0.005, 0.005, 0.1}), mat64.NewSymDense(1, []float64{0.5 / Δt}))
	x0 := mat64.NewVector(2, nil)
	Covar0 := mat64.NewSymDense(2, nil)
	kf, err := NewVanilla(x0, Covar0, F, G, mat64.NewDense(1, 2, []float64{0, 1}), noise)
	if err != nil {
		t.Fatal(err)
	}
	var est Estimate
	for k := 1; k < 50; k++ {
		est, err = kf.Update(mat64.NewVector(1, nil), mat64.NewVector(1, []float64{2 * math.Cos(0.75*float64(k))}))
		if err != nil {
			t.Fatal(err)
		}
		if !est.IsWithin2σ() {
			t.Logf("k=%d estimation not within 2σ bounds", k)
		}
	}
}
