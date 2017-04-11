package gokalman

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewHybridKF returns a new hybrid Kalman Filter which can be used both as a CKF and EKF.
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
func NewHybridKF(x0 *mat64.Vector, P0 mat64.Symmetric, noise Noise, measSize int) (*HybridKF, *HybridKFEstimate, error) {
	// Let's check the dimensions of everything here to return an error ASAP.
	if err := checkMatDims(x0, P0, "x0", "Covar0", rows2cols); err != nil {
		return nil, nil, err
	}

	// Populate with the initial values.
	cr, _ := P0.Dims()
	predCovar := mat64.NewSymDense(cr, nil)
	est0 := HybridKFEstimate{nil, nil, x0, mat64.NewVector(measSize, nil), mat64.NewVector(measSize, nil), mat64.NewVector(measSize, nil), P0, predCovar, nil}
	return &HybridKF{nil, nil, nil, noise, est0, false, true, false, measSize, 0}, &est0, nil
}

// HybridKF defines a hybrid kalman filter for non-linear dynamical systems. Use NewHybridKF to initialize.
type HybridKF struct {
	Φ, Htilde, Γ *mat64.Dense
	Noise        Noise
	prevEst      HybridKFEstimate
	ekfMode      bool // Allows switching between CKF and EKF.
	locked       bool // Locks the KF to ensure Prepare is called.
	sncEnabled   bool // Stores whether we should enable or disable the state noise compensation.
	measSize     int  // Stores the measurement vector size, needed only for Predict()
	step         int
}

// EKFEnabled returns whether the KF is in EKF mode.
func (kf *HybridKF) EKFEnabled() bool {
	return kf.ekfMode
}

// EnableEKF switches this to an EKF mode.
func (kf *HybridKF) EnableEKF() {
	kf.ekfMode = true
}

// DisableEKF switches this back to a CKF mode.
func (kf *HybridKF) DisableEKF() {
	kf.ekfMode = false
}

func (kf *HybridKF) String() string {
	return fmt.Sprintf("HybridKF [k=%d]\n%s", kf.step, kf.Noise)
}

// SetNoise updates the Noise.
func (kf *HybridKF) SetNoise(n Noise) {
	kf.Noise = n
}

// GetNoise updates the F matrix.
func (kf *HybridKF) GetNoise() Noise {
	return kf.Noise
}

// Prepare unlocks the KF ready for the next Update call.
func (kf *HybridKF) Prepare(Φ, Htilde *mat64.Dense) {
	kf.Φ = Φ
	kf.Htilde = Htilde
	kf.locked = false
}

// PreparePNT prepares the process noise transition matrix and enabled the SNC
// for the next update. WARNING: If not called, the SNC *will not* be included.
func (kf *HybridKF) PreparePNT(Γ *mat64.Dense) {
	kf.Γ = Γ
	kf.sncEnabled = true
}

// Update computes a full time and measurement update.
// Will return an error if the KF is locked (call Prepare to unlock).
func (kf *HybridKF) Update(realObservation, computedObservation *mat64.Vector) (est *HybridKFEstimate, err error) {
	return kf.fullUpdate(false, realObservation, computedObservation)
}

// Predict computes only the time update (or prediction).
// Will return an error if the KF is locked (call Prepare to unlock).
func (kf *HybridKF) Predict() (est *HybridKFEstimate, err error) {
	return kf.fullUpdate(true, nil, nil)
}

// fullUpdate performs all the steps of an update and allows to stop right after the pure prediction (or time update) step.
func (kf *HybridKF) fullUpdate(purePrediction bool, realObservation, computedObservation *mat64.Vector) (est *HybridKFEstimate, err error) {
	if kf.locked {
		return nil, errors.New("kf is locked (call Prepare() first)")
	}
	if !purePrediction {
		if err = checkMatDims(realObservation, computedObservation, "real observation", "computed observation", rowsAndcols); err != nil {
			return nil, err
		}
	}
	// PBar
	var PBar, ΦP mat64.Dense
	ΦP.Mul(kf.Φ, kf.prevEst.Covariance())
	PBar.Mul(&ΦP, kf.Φ.T())
	if kf.sncEnabled {
		// Add the process noise
		var ΓQΓt, ΓQ mat64.Dense
		ΓQ.Mul(kf.Γ, kf.Noise.ProcessMatrix())
		ΓQΓt.Mul(&ΓQ, kf.Γ.T())
		PBar.Add(&PBar, &ΓQΓt)
	}

	if purePrediction {
		var xBar mat64.Vector
		if kf.ekfMode {
			xBar = *mat64.NewVector(6, nil)
		} else {
			xBar.MulVec(kf.Φ, kf.prevEst.State())
		}
		// Time update completed.
		PBarSym, symerr := AsSymDense(&PBar)
		if symerr != nil {
			return nil, symerr
		}
		est = &HybridKFEstimate{kf.Φ, kf.Γ, &xBar, mat64.NewVector(kf.measSize, nil), mat64.NewVector(kf.measSize, nil), mat64.NewVector(kf.measSize, nil), PBarSym, PBarSym, mat64.NewDense(1, 1, nil)}
		kf.prevEst = *est
		kf.step++
		kf.sncEnabled = false
		kf.locked = true
		return
	}

	// Kalman gain
	var PHt, HPHt, K mat64.Dense
	PHt.Mul(&PBar, kf.Htilde.T())
	HPHt.Mul(kf.Htilde, &PHt)
	HPHt.Add(&HPHt, kf.Noise.MeasurementMatrix())
	if ierr := HPHt.Inverse(&HPHt); ierr != nil {
		return nil, fmt.Errorf("could not invert `H*P_kp1_minus*H' + R` at k=%d: %s", kf.step, ierr)
	}
	K.Mul(&PHt, &HPHt)

	// Compute observation deviation y
	var y mat64.Vector
	y.SubVec(realObservation, computedObservation)

	var innov, xHat mat64.Vector
	if kf.ekfMode {
		xHat.MulVec(&K, &y)
	} else {
		// Prediction step.
		var xBar mat64.Vector
		xBar.MulVec(kf.Φ, kf.prevEst.State())
		// Measurement update
		var Hx mat64.Vector
		Hx.MulVec(kf.Htilde, &xBar) // Predicted measurement
		innov.SubVec(&y, &Hx)       // Innovation vector
		// XXX: Does not support scalar measurements.
		xHat.MulVec(&K, &innov)
		xHat.AddVec(&xBar, &xHat)
	}
	var P, Ptmp1, IKH, KR, KRKt mat64.Dense
	IKH.Mul(&K, kf.Htilde)
	n, _ := IKH.Dims()
	IKH.Sub(Identity(n), &IKH)
	Ptmp1.Mul(&IKH, &PBar)
	P.Mul(&Ptmp1, IKH.T())
	KR.Mul(&K, kf.Noise.MeasurementMatrix())
	KRKt.Mul(&KR, K.T())
	P.Add(&P, &KRKt)

	PBarSym, err := AsSymDense(&PBar)
	if err != nil {
		return nil, err
	}

	PSym, err := AsSymDense(&P)
	if err != nil {
		return nil, err
	}
	Φ := *mat64.DenseCopyOf(kf.Φ)
	var Γ *mat64.Dense
	if kf.Γ != nil {
		Γ = mat64.DenseCopyOf(kf.Γ)
	}
	est = &HybridKFEstimate{&Φ, Γ, &xHat, realObservation, &innov, &y, PSym, PBarSym, &K}
	kf.prevEst = *est
	kf.step++
	kf.sncEnabled = false
	kf.locked = true
	return
}

// SmoothAll will smooth all the previous estimates using the provided data. Returns the smoothed estimates.
// Will return an error if there are more estimates than there should be.
// WARNING: overwrites the provided array of estimates.
func (kf *HybridKF) SmoothAll(estimates []*HybridKFEstimate) (err error) {
	if len(estimates) != kf.step {
		return errors.New("incorrect number of estimates provided")
	}
	l := len(estimates) - 1
	for k := l - 1; k >= 0; k-- {
		estimateKp1 := estimates[k+1]
		if estimateKp1.Γ == nil {
			// SNC was not enabled for this estimate.
			var S, SP, SPSt mat64.Dense
			if ierr := S.Inverse(estimateKp1.Φ); ierr != nil {
				return errors.New("provided STM Φ is not invertible")
			}
			SP.Mul(&S, estimateKp1.Covariance())
			SPSt.Mul(&SP, S.T())
			var xHat mat64.Vector
			xHat.MulVec(&S, estimateKp1.State())
			Pkl, serr := AsSymDense(&SPSt)
			if serr != nil {
				err = serr
				return
			}
			estimates[k].state = &xHat
			estimates[k].covar = Pkl
		} else {
			panic("not yet implemented")
		}
	}
	return
}

// HybridKFEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type HybridKFEstimate struct {
	Φ, Γ                     *mat64.Dense // Used for smoothing
	state, meas, innov, Δobs *mat64.Vector
	covar, predCovar         mat64.Symmetric
	gain                     mat64.Matrix
}

// IsWithinNσ returns whether the estimation is within the 2σ bounds.
func (e HybridKFEstimate) IsWithinNσ(N float64) bool {
	for i := 0; i < e.state.Len(); i++ {
		nσ := N * math.Sqrt(e.covar.At(i, i))
		if e.state.At(i, 0) > nσ || e.state.At(i, 0) < -nσ {
			return false
		}
	}
	return true
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e HybridKFEstimate) IsWithin2σ() bool {
	return e.IsWithinNσ(2)
}

// State implements the Estimate interface.
func (e HybridKFEstimate) State() *mat64.Vector {
	return e.state
}

// Measurement implements the Estimate interface.
func (e HybridKFEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Innovation implements the Estimate interface.
func (e HybridKFEstimate) Innovation() *mat64.Vector {
	return e.innov
}

// ObservationDev returns the observation deviation.
func (e HybridKFEstimate) ObservationDev() *mat64.Vector {
	return e.Δobs
}

// Covariance implements the Estimate interface.
func (e HybridKFEstimate) Covariance() mat64.Symmetric {
	return e.covar
}

// PredCovariance implements the Estimate interface.
func (e HybridKFEstimate) PredCovariance() mat64.Symmetric {
	return e.predCovar
}

// Gain the Estimate interface.
func (e HybridKFEstimate) Gain() mat64.Matrix {
	return e.gain
}

func (e HybridKFEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	innov := mat64.Formatted(e.Innovation(), mat64.Prefix("  "))
	predp := mat64.Formatted(e.PredCovariance(), mat64.Prefix("   "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\nP-=%v\ni=%v\n}", state, meas, covar, gain, predp, innov)
}
