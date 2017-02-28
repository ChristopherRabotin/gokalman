package gokalman

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewHybridCKF returns a new hybrid classical KF.
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
func NewHybridCKF(x0 *mat64.Vector, P0 mat64.Symmetric, noise Noise, measSize int) (*HybridCKF, *HybridCKFEstimate, error) {
	// Let's check the dimensions of everything here to return an error ASAP.
	if err := checkMatDims(x0, P0, "x0", "Covar0", rows2cols); err != nil {
		return nil, nil, err
	}

	// Populate with the initial values.
	cr, _ := P0.Dims()
	predCovar := mat64.NewSymDense(cr, nil)
	est0 := HybridCKFEstimate{x0, mat64.NewVector(measSize, nil), mat64.NewVector(measSize, nil), mat64.NewVector(measSize, nil), P0, predCovar, nil}
	return &HybridCKF{nil, nil, noise, est0, true, 0}, &est0, nil
}

// HybridCKF defines a vanilla kalman filter. Use NewVanilla to initialize.
type HybridCKF struct {
	Φ, Htilde *mat64.Dense
	Noise     Noise
	prevEst   HybridCKFEstimate
	locked    bool // Locks the KF to ensure Prepare is called.
	step      int
}

func (kf *HybridCKF) String() string {
	return fmt.Sprintf("HybridCKF [k=%d]\n%s", kf.step, kf.Noise)
}

// SetNoise updates the Noise.
func (kf *HybridCKF) SetNoise(n Noise) {
	kf.Noise = n
}

// GetNoise updates the F matrix.
func (kf *HybridCKF) GetNoise() Noise {
	return kf.Noise
}

// Prepare unlocks the KF ready for the next Update call.
func (kf *HybridCKF) Prepare(Φ, Htilde *mat64.Dense) {
	kf.Φ = Φ
	kf.Htilde = Htilde
	kf.locked = false
}

// Update implements the HybridKalmanFilter interface to compute an update.
// Will return an error if the KF is locked (call Prepare to unlock)
func (kf *HybridCKF) Update(realObservation, computedObservation *mat64.Vector) (est *HybridCKFEstimate, err error) {
	if kf.locked {
		return nil, errors.New("kf is locked (call Prepare() first)")
	}
	if err = checkMatDims(realObservation, computedObservation, "real observation", "computed observation", rowsAndcols); err != nil {
		return nil, err
	}

	// Prediction step.
	var xBar mat64.Vector
	xBar.MulVec(kf.Φ, kf.prevEst.State())
	// XXX: No future proofing so I'm not using the process noise just yet.
	// PBar
	var PBar, ΦP mat64.Dense
	ΦP.Mul(kf.Φ, kf.prevEst.Covariance())
	PBar.Mul(&ΦP, kf.Φ.T())
	//XXX: Item (may be: PBar.Add(&PBar, kf.Noise.ProcessMatrix()) ?)

	// Compute observation deviation y
	var y mat64.Vector
	y.SubVec(realObservation, computedObservation)
	// Kalman gain
	var PHt, HPHt, Kkp1 mat64.Dense
	PHt.Mul(&PBar, kf.Htilde.T())
	HPHt.Mul(kf.Htilde, &PHt)
	HPHt.Add(&HPHt, kf.Noise.MeasurementMatrix())
	if ierr := HPHt.Inverse(&HPHt); ierr != nil {
		return nil, fmt.Errorf("could not invert `H*P_kp1_minus*H' + R` at k=%d: %s", kf.step, ierr)
	}
	Kkp1.Mul(&PHt, &HPHt)

	// Measurement update
	var innov, Hx, xHat mat64.Vector
	Hx.MulVec(kf.Htilde, &xBar) // Predicted measurement
	innov.SubVec(&y, &Hx)       // Innovation vector
	// XXX: Does not support scalar measurements.
	xHat.MulVec(&Kkp1, &innov)
	xHat.AddVec(&xBar, &xHat)

	var P, Ptmp1, IKH, KR, KRKt mat64.Dense
	IKH.Mul(&Kkp1, kf.Htilde)
	n, _ := IKH.Dims()
	IKH.Sub(Identity(n), &IKH)
	Ptmp1.Mul(&IKH, &PBar)
	P.Mul(&Ptmp1, IKH.T())
	KR.Mul(&Kkp1, kf.Noise.MeasurementMatrix())
	KRKt.Mul(&KR, Kkp1.T())
	P.Add(&P, &KRKt)

	PBarSym, err := AsSymDense(&PBar)
	if err != nil {
		return nil, err
	}

	PSym, err := AsSymDense(&P)
	if err != nil {
		return nil, err
	}
	est = &HybridCKFEstimate{&xHat, realObservation, &innov, &y, PSym, PBarSym, &Kkp1}
	kf.prevEst = *est
	kf.step++
	kf.locked = true
	return
}

// HybridCKFEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type HybridCKFEstimate struct {
	state, meas, innov, Δobs *mat64.Vector
	covar, predCovar         mat64.Symmetric
	gain                     mat64.Matrix
}

// IsWithinNσ returns whether the estimation is within the 2σ bounds.
func (e HybridCKFEstimate) IsWithinNσ(N float64) bool {
	for i := 0; i < e.state.Len(); i++ {
		nσ := N * math.Sqrt(e.covar.At(i, i))
		if e.state.At(i, 0) > nσ || e.state.At(i, 0) < -nσ {
			return false
		}
	}
	return true
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e HybridCKFEstimate) IsWithin2σ() bool {
	return e.IsWithinNσ(2)
}

// State implements the Estimate interface.
func (e HybridCKFEstimate) State() *mat64.Vector {
	return e.state
}

// Measurement implements the Estimate interface.
func (e HybridCKFEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Innovation implements the Estimate interface.
func (e HybridCKFEstimate) Innovation() *mat64.Vector {
	return e.innov
}

// ObservationDev returns the observation deviation.
func (e HybridCKFEstimate) ObservationDev() *mat64.Vector {
	return e.Δobs
}

// Covariance implements the Estimate interface.
func (e HybridCKFEstimate) Covariance() mat64.Symmetric {
	return e.covar
}

// PredCovariance implements the Estimate interface.
func (e HybridCKFEstimate) PredCovariance() mat64.Symmetric {
	return e.predCovar
}

// Gain the Estimate interface.
func (e HybridCKFEstimate) Gain() mat64.Matrix {
	return e.gain
}

func (e HybridCKFEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	innov := mat64.Formatted(e.Innovation(), mat64.Prefix("  "))
	predp := mat64.Formatted(e.PredCovariance(), mat64.Prefix("   "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\nP-=%v\ni=%v\n}", state, meas, covar, gain, predp, innov)
}
