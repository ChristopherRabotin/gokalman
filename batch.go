package gokalman

import "github.com/gonum/matrix/mat64"

type measurementInfo struct {
	RealObs        *mat64.Vector
	ComputedObs    *mat64.Vector
	ObservationDev *mat64.Vector
	Φ, H           *mat64.Dense
}

// BatchKF defines a vanilla kalman filter. Use NewVanilla to initialize.
type BatchKF struct {
	Λ            *mat64.Dense
	N            *mat64.Vector
	Measurements []measurementInfo
	noise        Noise
	locked       bool // Locks the KF to prevent adding more measurements when iterating
	step         int
}

// NewBatchKF returns a new hybrid Kalman Filter which can be used both as a CKF and EKF.
// Warning: there is a failsafe preventing any update prior to updating the matrices.
// Usage:
// ```
/// kf.Prepare(Φ, Htilde)
//  estimate, err := kf.Update(realObs, computedObs)
/// ```
// Parameters:
// - x0: initial state estimate
// - P0: initial covariance symmetric matrix
// - noise: Noise
// - measSize: number of rows of the measurement vector (not actually important)
func NewBatchKF(numMeasurements int, noise Noise) *BatchKF {
	meas := make([]measurementInfo, numMeasurements)
	// Note that we will create the Λ matrix and N vector on the first call to SetNextMeasurement.
	return &BatchKF{nil, nil, meas, noise, false, 0}
}

// SetNextMeasurement sets the next sequential measurement to the list of measurements to be taken into account for the filter.
func (kf *BatchKF) SetNextMeasurement(realObs, computedObs *mat64.Vector, Φ, H *mat64.Dense) {
	if kf.N == nil || kf.Λ == nil {
		// There is no reason for both to *not* happen at the same time, but whatevs
		_, cH := H.Dims()
		kf.N = mat64.NewVector(cH, nil)
		kf.Λ = mat64.NewDense(cH, cH, nil)
	}
	// And compute the current Λ and N.
	var HtR, HtRH mat64.Dense
	HtR.Mul(H.T(), kf.noise.MeasurementMatrix())
	HtRH.Mul(&HtR, H)
	kf.Λ.Add(kf.Λ, &HtRH)
	// Compute observation deviation y
	var y, HtRY mat64.Vector
	y.SubVec(realObs, computedObs)
	// Store the measurement
	kf.Measurements[kf.step] = measurementInfo{realObs, computedObs, &y, Φ, H}
	HtRY.MulVec(&HtR, &y)
	kf.N.AddVec(kf.N, &HtRY)
	kf.step++
}

// Solve will solve the Batch Kalman filter once and return xHat0 and P0, or an error
func (kf *BatchKF) Solve() (xHat0 *mat64.Vector, P0 *mat64.SymDense, err error) {
	var Λinv mat64.Dense
	if err = Λinv.Inverse(kf.Λ); err != nil {
		return nil, nil, err
	}
	// Make Λinv symmetric and store in P0
	P0, err = AsSymDense(&Λinv)
	if err != nil {
		return nil, nil, err
	}
	// Compute xHat0
	rΛ, _ := kf.Λ.Dims()
	xHat0 = mat64.NewVector(rΛ, nil)
	xHat0.MulVec(P0, kf.N)
	return xHat0, P0, nil
}
