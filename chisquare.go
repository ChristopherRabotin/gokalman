package gokalman

import (
	"errors"
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// NewChiSquare runs the Chi square tests from the MonteCarlo runs. These runs
// and the KF used are the ones tested via Chi square. The KF provided must be a
// pure predictor Vanilla KF and will be used to compute the intermediate steps
// of both the NEES and NIS tests.
// Returns NEESmeans, NISmeans and an error if applicable
// TODO: Change order of parameters.
func NewChiSquare(kf KalmanFilter, runs MonteCarloRuns, colsG int, withNEES, withNIS bool) ([]float64, []float64, error) {
	if !withNEES && !withNIS {
		return nil, nil, errors.New("Chi Square requires either NEES or NIS or both")
	}

	numRuns := runs.runs
	numSteps := len(runs.Runs[0].Estimates)
	NISsamples := make(map[int][]float64)
	NEESsamples := make(map[int][]float64)

	for rNo, run := range runs.Runs {
		for k, mcTruth := range run.Estimates {

			if k < 2 {
				fmt.Printf("---\n%d / %d\n%s\n", rNo, k, mcTruth)
			}

			est, err := kf.Update(mcTruth.Measurement(), mat64.NewVector(colsG, nil))
			if err != nil {
				panic(err)
			}

			if withNEES {
				if NEESsamples[k] == nil {
					NEESsamples[k] = make([]float64, numRuns)
				}
				var PInv mat64.Dense
				PInv.Inverse(est.Covariance())

				var nees, nees0, nees1 mat64.Vector
				nees0.SubVec(mcTruth.State(), est.State())
				nees1.MulVec(&PInv, &nees0)
				nees.MulVec(nees0.T(), &nees1)
				NEESsamples[k][rNo] = nees.At(0, 0) // Will be just a scalar.
				//fmt.Printf("nees=%f\n", nees.At(0, 0))
			}

			if withNIS {
				if NISsamples[k] == nil {
					NISsamples[k] = make([]float64, numRuns)
				}
				// Compute the actual NIS.
				var Pyy, Pyy0, PyyInv mat64.Dense
				H := kf.GetMeasurementMatrix()
				Pyy0.Mul(est.PredCovariance(), H.T())
				Pyy.Mul(H, &Pyy0)
				Pyy.Add(&Pyy, kf.GetNoise().MeasurementMatrix())
				// This corresponds to the pure prediction: H*Pkp1_minus*H' + Rtrue;
				PyyInv.Inverse(&Pyy)
				var nis, nis0 mat64.Vector
				nis0.MulVec(&PyyInv, est.Innovation())
				nis.MulVec(est.Innovation().T(), &nis0)
				NISsamples[k][rNo] = nis.At(0, 0) // Will be just be a scalar.
			}
		}
		kf.Reset()
	}

	// Let's compute the means for each step.
	NISmeans := make([]float64, numSteps)
	NEESmeans := make([]float64, numSteps)

	for k := 0; k < numSteps; k++ {
		if withNEES {
			NEESmeans[k] = stat.Mean(NEESsamples[k], nil)
			fmt.Printf("nees[%d]=%f\n", k, NEESmeans[k])
		}
		if withNIS {
			NISmeans[k] = stat.Mean(NISsamples[k], nil)
		}
	}

	fmt.Printf("avr=%f\n", stat.Mean(NEESmeans, nil))

	return NISmeans, NEESmeans, nil
}
