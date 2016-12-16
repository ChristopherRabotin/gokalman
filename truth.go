package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// BatchGroundTruth computes the error of a given estimate from a known batch of states and measurements.
type BatchGroundTruth struct {
	states       []*mat64.Vector
	measurements []*mat64.Vector
	k            int // Current step
}

func (t *BatchGroundTruth) Error(est Estimate) Estimate {
	esR, _ := est.State().Dims()
	estState := mat64.NewVector(esR, nil)
	estState.CopyVec(est.State())
	trueState := mat64.NewVector(esR, nil)
	if t.states != nil {
		trueState = t.states[t.k]
		tsR, _ := trueState.Dims()
		if esR != tsR {
			panic(fmt.Errorf("ground truth state size different from estimated state size (k=%d)", t.k))
		}
		// Inline modification of the estimated state.
		for i := 0; i < tsR; i++ {
			estState.SetVec(i, estState.At(i, 0)-trueState.At(i, 0))
		}
	}

	emR, _ := est.Measurement().Dims()
	estMeas := mat64.NewVector(emR, nil)
	estMeas.CopyVec(est.Measurement())
	trueMeas := mat64.NewVector(esR, nil)
	if t.states != nil {
		trueMeas = t.measurements[t.k]
		tmR, _ := trueMeas.Dims()
		if emR != tmR {
			panic(fmt.Errorf("ground truth measurement size different from estimated measurement size (k=%d)", t.k))
		}
		// Inline modification of the estimated measurement.
		for i := 0; i < tmR; i++ {
			estMeas.SetVec(i, estMeas.At(i, 0)-trueMeas.At(i, 0))
		}
	}
	t.k++
	return ErrorEstimate{VanillaEstimate{state: estState, meas: estMeas, covar: est.Covariance()}}
}

// NewBatchGroundTruth initializes a new batch ground truth.
func NewBatchGroundTruth(states, measurements []*mat64.Vector) *BatchGroundTruth {
	return &BatchGroundTruth{states, measurements, 0}
}

// ErrorEstimate implements the Estimate interface and is used to show the error of an estimate.
type ErrorEstimate struct {
	VanillaEstimate // This is effectively the same as a VanillaEstimate, so no change.
}
