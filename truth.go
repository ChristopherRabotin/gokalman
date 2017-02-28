package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// BatchGroundTruth computes the error of a given estimate from a known batch of states and measurements.
type BatchGroundTruth struct {
	states       []*mat64.Vector
	measurements []*mat64.Vector
}

// Error returns an ErrorEstimate after comparing the provided state and measurements with the ground truths.
func (t *BatchGroundTruth) Error(k int, est Estimate) Estimate {
	return t.ErrorWithOffset(k, est, nil)
}

// ErrorWithOffset returns an ErrorEstimate after comparing the provided state, adding the offset and measurements with the ground truths.
func (t *BatchGroundTruth) ErrorWithOffset(k int, est Estimate, offset *mat64.Vector) Estimate {
	esR, _ := est.State().Dims()
	estState := mat64.NewVector(esR, nil)
	estState.CopyVec(est.State())
	if offset != nil {
		estState.AddVec(estState, offset)
	}
	trueState := mat64.NewVector(esR, nil)
	if t.states != nil {
		trueState = t.states[k]
		tsR, _ := trueState.Dims()
		if esR != tsR {
			panic(fmt.Errorf("ground truth state size different from estimated state size (k=%d)", k))
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
		trueMeas = t.measurements[k]
		tmR, _ := trueMeas.Dims()
		if emR != tmR {
			panic(fmt.Errorf("ground truth measurement size different from estimated measurement size (k=%d)", k))
		}
		// Inline modification of the estimated measurement.
		for i := 0; i < tmR; i++ {
			estMeas.SetVec(i, estMeas.At(i, 0)-trueMeas.At(i, 0))
		}
	}
	return ErrorEstimate{VanillaEstimate{state: estState, meas: estMeas, covar: est.Covariance()}}
}

// NewBatchGroundTruth initializes a new batch ground truth.
func NewBatchGroundTruth(states, measurements []*mat64.Vector) *BatchGroundTruth {
	return &BatchGroundTruth{states, measurements}
}

// ErrorEstimate implements the Estimate interface and is used to show the error of an estimate.
type ErrorEstimate struct {
	VanillaEstimate // This is effectively the same as a VanillaEstimate, so no change.
}
