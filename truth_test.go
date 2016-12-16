package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestBatchError(t *testing.T) {
	truth := NewBatchGroundTruth([]*mat64.Vector{
		mat64.NewVector(2, []float64{1, 1}), mat64.NewVector(2, []float64{2, 2}),
	}, []*mat64.Vector{
		mat64.NewVector(2, []float64{3, 3}), mat64.NewVector(2, []float64{4, 4}),
	})

	iest := InformationEstimate{cachedState: mat64.NewVector(2, []float64{1, 1}), meas: mat64.NewVector(2, []float64{4, 4}), cachedCovar: ScaledIdentity(2, 1)}
	sqest := NewSqrtEstimate(mat64.NewVector(2, []float64{1, 1}), mat64.NewVector(2, []float64{4, 4}), mat64.NewDense(2, 2, []float64{1, 0, 0, 0}), mat64.NewDense(2, 2, nil))

	for kest, est := range []Estimate{iest, sqest} {
		for kexp, exp := range []struct {
			state, meas *mat64.Vector
		}{
			{state: mat64.NewVector(2, nil), meas: mat64.NewVector(2, []float64{1, 1})},
			{state: mat64.NewVector(2, []float64{-1, -1}), meas: mat64.NewVector(2, nil)},
		} {
			trueEst := truth.Error(est)
			if !mat64.Equal(exp.state, trueEst.State()) {
				t.Fatalf("state failed with est#%d exp#%d: \n%v\n%v", kest, kexp, mat64.Formatted(exp.state, mat64.Prefix("")), mat64.Formatted(trueEst.State(), mat64.Prefix("")))
			}

			if !mat64.Equal(exp.meas, trueEst.Measurement()) {
				t.Fatalf("measurement failed with est#%d exp#%d: \n%v\n%v", kest, kexp, mat64.Formatted(exp.meas, mat64.Prefix("")), mat64.Formatted(trueEst.Measurement(), mat64.Prefix("")))
			}

		}
		truth.k = 0 // Reset here for tests
	}

	// Test panics
	assertPanic(t, func() {
		wrongStateSize := InformationEstimate{cachedState: mat64.NewVector(3, []float64{1, 1}), meas: mat64.NewVector(2, []float64{4, 4}), cachedCovar: ScaledIdentity(2, 1)}
		truth.Error(wrongStateSize)
	})

	assertPanic(t, func() {
		wrongMeasSize := InformationEstimate{cachedState: mat64.NewVector(2, []float64{1, 1}), meas: mat64.NewVector(3, []float64{4, 4}), cachedCovar: ScaledIdentity(2, 1)}
		truth.Error(wrongMeasSize)
	})
}
