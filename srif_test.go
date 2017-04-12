package gokalman

import (
	"fmt"
	"math"
	"testing"

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
}

func TestSRIFUpdate(t *testing.T) {
	R := mat64.NewDense(2, 2, []float64{0.1, 0, 0, 0.1})
	H := mat64.NewDense(3, 2, []float64{1, -2, 2, -1, 1, 1})
	b := mat64.NewVector(2, []float64{0.2, 0.2})
	y := mat64.NewVector(3, []float64{-1.1, 1.2, 1.8})
	Rk, bk, ek, err := measurementSRIFUpdate(R, H, b, y)
	if err != nil {
		t.Fatalf("%s", err)
	}
	expEk := mat64.NewVector(3, []float64{-0.1319, 0.0871, -0.2810})
	if !mat64.EqualApprox(ek, expEk, 1e-4) {
		fmt.Printf("%+v", mat64.Formatted(ek))
		expEk.SubVec(ek, expEk)
		t.Fatalf("ek wrong by:\n%+v", mat64.Formatted(expEk))
	}
	expBk := mat64.NewVector(2, []float64{-1.2727, -2.0607})
	if !mat64.EqualApprox(bk, expBk, 1e-4) {
		expBk.SubVec(bk, expBk)
		t.Fatalf("bk wrong by:\n%+v", mat64.Formatted(expBk))
	}
	expRk := mat64.NewDense(2, 2, []float64{-2.4515, 1.2237, 0, -2.1243})
	if !mat64.EqualApprox(Rk, expRk, 1e-4) {
		expRk.Sub(expRk, Rk)
		t.Fatalf("Rk wrong by:\n%+v", mat64.Formatted(expRk))
	}
}
