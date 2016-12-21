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
func NewChiSquare(kf *Vanilla, runs MonteCarloRuns, colsG int, withNEES, withNIS bool) ([]float64, []float64, error) {
	if !withNEES && !withNIS {
		return nil, nil, errors.New("Chi Square requires either NEES or NIS or both")
	}

	numRuns := runs.runs
	numSteps := len(runs.Runs[0].Estimates)
	NISsamples := make(map[int][]float64)
	NEESsamples := make(map[int][]float64)

	for rNo, run := range runs.Runs {
		for k, mcTruth := range run.Estimates {

			est, err := kf.Update(mcTruth.Measurement(), mat64.NewVector(colsG, nil))
			if err != nil {
				panic(err)
			}

			// Compute the innovation: mcEst is used as the truth/sensor report,
			// and est is what we're computing from the PurePrediction KF.
			var innovation mat64.Vector
			innovation.SubVec(mcTruth.Measurement(), est.Measurement())

			if withNEES {
				if NEESsamples[k] == nil {
					NEESsamples[k] = make([]float64, numRuns)
				}
				var mkp1Plus, mkp1Plus0 mat64.Vector
				vest := est.(VanillaEstimate)
				mkp1Plus0.MulVec(vest.Gain(), &innovation)
				mkp1Plus.AddVec(vest.State(), &mkp1Plus0)

				// Recompute the Pkp1Plus
				//Pkp1_plus = (eye(2) - Kkp1*H)*Pkp1_minus; %compute update to covar
				var Pkp1Plus, KH, PInv mat64.Dense
				KH.Mul(vest.Gain(), kf.H)
				rows, _ := KH.Dims()
				KH.Sub(Identity(rows), &KH)
				Pkp1Plus.Mul(&KH, vest.Covariance())

				if ierr := PInv.Inverse(&Pkp1Plus); ierr != nil {
					fmt.Printf("covariance might be singular: %s\nP=%v\n", ierr, mat64.Formatted(est.Covariance(), mat64.Prefix("  ")))
				}
				var nees, nees0, nees1 mat64.Vector
				nees0.SubVec(mcTruth.State(), &mkp1Plus)
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
				Pyy0.Mul(est.Covariance(), kf.H.T())
				Pyy.Mul(kf.H, &Pyy0)
				Pyy.Add(&Pyy, kf.Noise.MeasurementMatrix())
				// This corresponds to the pure prediction: H*Pkp1_minus*H' + Rtrue;
				// Which we can find as the covariance of the MC run estimate.
				if ierr := PyyInv.Inverse(&Pyy); ierr != nil {
					fmt.Printf("Pyy might be singular: %s\n", ierr)
				}
				var nis, nis0 mat64.Vector
				nis0.MulVec(&PyyInv, &innovation)
				nis.MulVec(innovation.T(), &nis0)
				NISsamples[k][rNo] = nis.At(0, 0) // Will be just be a scalar.
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