package gokalman

import (
	"errors"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// NewChiSquare runs the Chi square tests from the MonteCarlo runs. These runs
// and the KF used are the ones tested via Chi square. The KF provided must be a
// pure predictor Vanilla KF and will be used to compute the intermediate steps
// of both the NEES and NIS tests.
// Returns NEESmeans, NISmeans and an error if applicable
// TODO: Change order of parameters.
func NewChiSquare(kf KalmanFilter, runs MonteCarloRuns, controls []*mat64.Vector, withNEES, withNIS bool) ([]float64, []float64, error) {
	if !withNEES && !withNIS {
		return nil, nil, errors.New("Chi Square requires either NEES or NIS or both")
	}

	numRuns := runs.runs
	steps := len(runs.Runs[0].Estimates)
	NISsamples := make(map[int][]float64)
	NEESsamples := make(map[int][]float64)

	if len(controls) == 1 {
		ctrlSize, _ := controls[0].Dims()
		controls = make([]*mat64.Vector, steps)
		// Populate with zero controls
		for k := 0; k < steps; k++ {
			controls[k] = mat64.NewVector(ctrlSize, nil)
		}
	} else if len(controls) != steps {
		return nil, nil, errors.New("must provide as much control vectors as steps, or just one control vector")
	}

	for rNo, run := range runs.Runs {
		kf.Reset()
		for k, mcTruth := range run.Estimates {

			est, err := kf.Update(mcTruth.Measurement(), controls[k])
			if err != nil {
				panic(err)
			}

			if withNEES {
				if NEESsamples[k] == nil {
					NEESsamples[k] = make([]float64, numRuns)
				}
				var PInv mat64.Dense
				PInv.Inverse(est.Covariance()) // XXX: Pinv is OK

				var nees, nees0, nees1 mat64.Vector
				nees0.SubVec(mcTruth.State(), est.State())
				//				fmt.Printf("d=%v\n", mat64.Formatted(&nees0, mat64.Prefix("  ")))
				nees1.MulVec(&PInv, &nees0)
				nees.MulVec(nees0.T(), &nees1)
				NEESsamples[k][rNo] = nees.At(0, 0) // Will be just a scalar.
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
	}

	// Let's compute the means for each step.
	NISmeans := make([]float64, steps)
	NEESmeans := make([]float64, steps)

	for k := 0; k < steps; k++ {
		if withNEES {
			NEESmeans[k] = stat.Mean(NEESsamples[k], nil)
		}
		if withNIS {
			NISmeans[k] = stat.Mean(NISsamples[k], nil)
		}
	}

	return NISmeans, NEESmeans, nil
}
