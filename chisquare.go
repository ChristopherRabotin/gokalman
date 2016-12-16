package gokalman

import (
	"errors"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// NewChiSquare runs the Chi square tests on the provided KF.
// The runs should be from the ground truth generation.
// Returns NEESmeans, NISmeans and an error if applicable
func NewChiSquare(kf *Vanilla, runs MonteCarloRuns, colsG int, withNEES, withNIS bool) ([]float64, []float64, error) {
	if !withNEES && !withNIS {
		return nil, nil, errors.New("Chi Square requires either NEES or NIS or both")
	}

	numRuns := runs.runs
	numSteps := len(runs.Runs[0].Estimates)
	NISsamples := make(map[int][]float64)
	NEESsamples := make(map[int][]float64)

	for rNo, run := range runs.Runs {
		for k, mcEst := range run.Estimates {

			est, err := kf.Update(mcEst.Measurement(), mat64.NewVector(colsG, nil))
			if err != nil {
				panic(err)
			}

			if withNEES {
				if NEESsamples[k] == nil {
					NEESsamples[k] = make([]float64, numRuns)
				}
				var neesVect mat64.Vector
				neesVect.SubVec(mcEst.State(), est.State())
				var PInv mat64.Dense
				PInv.Inverse(est.Covariance())
				var nees, nees0 mat64.Vector
				nees0.MulVec(&PInv, &neesVect)
				nees.MulVec(neesVect.T(), &nees0)
				NEESsamples[k][rNo] = nees.At(0, 0) // Should just be a scalar.
			}

			if withNIS {
				if NISsamples[k] == nil {
					NISsamples[k] = make([]float64, numRuns)
				}
				// Store the innovation.
				var innovation mat64.Vector
				innovation.SubVec(mcEst.Measurement(), est.Measurement())
				// Compute the actual NIS.
				var Pyy, Pyy0, PyyInv mat64.Dense
				Pyy0.Mul(mcEst.Covariance(), kf.H.T())
				Pyy.Mul(kf.H, &Pyy0)
				// This corresponds to the pure prediction: H*Pkp1_minus*H' + Rtrue;
				// Which we can find as the covariance of the MC run estimate.
				PyyInv.Inverse(&Pyy)
				var nis, nis0 mat64.Vector
				nis0.MulVec(&PyyInv, &innovation)
				nis.MulVec(innovation.T(), &nis0)
				NISsamples[k][rNo] = nis.At(0, 0) // Should just be a scalar.
			}
		}
	}

	// Let's compute the means for each step.
	NISmeans := make([]float64, numSteps)
	NEESmeans := make([]float64, numSteps)

	for k := 0; k < numSteps; k++ {
		if withNIS {
			NISmeans[k] = stat.Mean(NISsamples[k], nil)
		}
		if withNEES {
			NEESmeans[k] = stat.Mean(NEESsamples[k], nil)
		}
	}

	return NISmeans, NEESmeans, nil
}
