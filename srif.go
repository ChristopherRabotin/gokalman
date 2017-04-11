package gokalman

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewSquareRootInformation returns a new Square Root Information Filter.
// It uses the algorithms from "Statistical Orbit determination" by Tapley, Schutz & Born.
// Set upperTriangular to `false` to use the lower triangular of the Cholesky decomposition instead of the upper.
func NewSquareRootInformation(x0 *mat64.Vector, P0 mat64.Symmetric, measSize int, upperTriangular bool) (*SRIF, *SRIFEstimate, error) {
	// Check the dimensions of each matrix to avoid errors.
	if err := checkMatDims(x0, P0, "x0", "P0", rows2cols); err != nil {
		return nil, nil, err
	}

	// Get the initial information matrix, I0.
	// NOTE: P0 is always diagonal, so we can just invert each component.
	r, _ := P0.Dims()
	I0 := mat64.NewSymDense(r, nil)
	for i := 0; i < r; i++ {
		I0.SetSym(i, i, 1/P0.At(i, i))
	}
	// Compute the cholesky factorization of the information matrix.
	var R0chol mat64.Cholesky
	R0chol.Factorize(I0)
	var R0tri mat64.TriDense
	if upperTriangular {
		R0tri.UFromCholesky(&R0chol)
	} else {
		R0tri.LFromCholesky(&R0chol)
	}
	var R0 mat64.Dense
	R0.Clone(&R0tri)
	b0 := mat64.NewVector(r, nil)
	b0.MulVec(&R0, x0)

	// Populate with the initial values.
	est0 := NewSRIFEstimate(nil, b0, nil, nil, nil, &R0, &R0)
	return &SRIF{nil, nil, est0, true, measSize, 0}, &est0, nil
}

// SRIF defines a square root information filter for non-linear dynamical systems. Use NewSquareRootInformation to initialize.
type SRIF struct {
	Φ, Htilde *mat64.Dense
	prevEst   SRIFEstimate
	locked    bool // Locks the KF to ensure Prepare is called.
	measSize  int  // Stores the measurement vector size, needed only for Predict()
	step      int
}

// Prepare unlocks the KF ready for the next Update call.
func (kf *SRIF) Prepare(Φ, Htilde *mat64.Dense) {
	kf.Φ = Φ
	kf.Htilde = Htilde
	kf.locked = false
}

// Update computes a full time and measurement update.
// Will return an error if the KF is locked (call Prepare to unlock).
func (kf *SRIF) Update(realObservation, computedObservation *mat64.Vector) (est *SRIFEstimate, err error) {
	return kf.fullUpdate(false, realObservation, computedObservation)
}

// Predict computes only the time update (or prediction).
// Will return an error if the KF is locked (call Prepare to unlock).
func (kf *SRIF) Predict() (est *SRIFEstimate, err error) {
	return kf.fullUpdate(true, nil, nil)
}

// fullUpdate performs all the steps of an update and allows to stop right after the pure prediction (or time update) step.
func (kf *SRIF) fullUpdate(purePrediction bool, realObservation, computedObservation *mat64.Vector) (est *SRIFEstimate, err error) {
	if kf.locked {
		return nil, errors.New("kf is locked (call Prepare() first)")
	}
	if !purePrediction {
		if err = checkMatDims(realObservation, computedObservation, "real observation", "computed observation", rowsAndcols); err != nil {
			return nil, err
		}
	}
	// RBar
	var RBar, invΦ mat64.Dense
	if ierr := invΦ.Inverse(kf.Φ); ierr != nil {
		return nil, fmt.Errorf("could not invert `Φ` at k=%d: %s", kf.step, ierr)
	}
	RBar.Mul(kf.prevEst.matR0, &invΦ)

	var xBar, bBar mat64.Vector
	xBar.MulVec(kf.Φ, kf.prevEst.State())
	bBar.MulVec(&RBar, &xBar)

	if purePrediction {
		kf.prevEst = NewSRIFEstimate(kf.Φ, &bBar, mat64.NewVector(kf.measSize, nil), mat64.NewVector(kf.measSize, nil), mat64.NewVector(kf.measSize, nil), &RBar, &RBar)
		kf.step++
		kf.locked = true
		return
	}
	/*
		// Kalman gain
		var PHt, HPHt, K mat64.Dense
		PHt.Mul(&PBar, kf.Htilde.T())
		HPHt.Mul(kf.Htilde, &PHt)
		if ierr := HPHt.Inverse(&HPHt); ierr != nil {
			return nil, fmt.Errorf("could not invert `H*P_kp1_minus*H' + R` at k=%d: %s", kf.step, ierr)
		}
		K.Mul(&PHt, &HPHt)

		// Compute observation deviation y
		var y mat64.Vector
		y.SubVec(realObservation, computedObservation)

		var innov, xHat mat64.Vector

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

		var P, Ptmp1, IKH, KR, KRKt mat64.Dense
		IKH.Mul(&K, kf.Htilde)
		n, _ := IKH.Dims()
		IKH.Sub(Identity(n), &IKH)
		Ptmp1.Mul(&IKH, &PBar)
		P.Mul(&Ptmp1, IKH.T())
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
		est = NewSRIFEstimate(Φ, xHat, &y, innov, realObservation, R0, predR0)
		kf.prevEst = *est
		kf.step++
		kf.locked = true
	*/
	return
}

// SRIFEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type SRIFEstimate struct {
	Φ                            *mat64.Dense // Used for smoothing
	sqinfoState, meas            *mat64.Vector
	innov, Δobs, cachedState     *mat64.Vector
	matR0, predMatR0             *mat64.Dense
	cachedCovar, predCachedCovar mat64.Symmetric
}

// IsWithinNσ returns whether the estimation is within the 2σ bounds.
func (e SRIFEstimate) IsWithinNσ(N float64) bool {
	state := e.State()
	covar := e.Covariance()
	for i := 0; i < state.Len(); i++ {
		nσ := N * math.Sqrt(covar.At(i, i))
		if state.At(i, 0) > nσ || state.At(i, 0) < -nσ {
			return false
		}
	}
	return true
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e SRIFEstimate) IsWithin2σ() bool {
	return e.IsWithinNσ(2)
}

// State implements the Estimate interface.
func (e SRIFEstimate) State() *mat64.Vector {
	if e.cachedState == nil {
		rState, _ := e.sqinfoState.Dims()
		e.cachedState = mat64.NewVector(rState, nil)
		e.cachedState.MulVec(e.Covariance(), e.sqinfoState)
	}
	return e.cachedState
}

// Measurement implements the Estimate interface.
func (e SRIFEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Innovation implements the Estimate interface.
func (e SRIFEstimate) Innovation() *mat64.Vector {
	return e.innov
}

// ObservationDev returns the observation deviation.
func (e SRIFEstimate) ObservationDev() *mat64.Vector {
	return e.Δobs
}

// Covariance implements the Estimate interface.
func (e SRIFEstimate) Covariance() mat64.Symmetric {
	if e.cachedCovar == nil {
		var invCovar mat64.Dense
		invCovar.Mul(e.matR0, e.matR0.T())
		var tmpCovar mat64.Dense
		err := tmpCovar.Inverse(&invCovar)
		if err != nil {
			fmt.Printf("gokalman: SRIF: R0 is not (yet) invertible: %s\n", err)
			return e.cachedCovar
		}
		cachedCovar, _ := AsSymDense(&tmpCovar)
		e.cachedCovar = cachedCovar
	}
	return e.cachedCovar
}

// PredCovariance implements the Estimate interface.
func (e SRIFEstimate) PredCovariance() mat64.Symmetric {
	if e.predCachedCovar == nil {
		var invPredCovar mat64.Dense
		invPredCovar.Mul(e.predMatR0, e.predMatR0.T())
		var tmpPredCovar mat64.Dense
		err := tmpPredCovar.Inverse(&invPredCovar)
		if err != nil {
			fmt.Printf("gokalman: SRIF: prediction R0 is not (yet) invertible: %s\n", err)
			return e.cachedCovar
		}
		predCachedCovar, _ := AsSymDense(&tmpPredCovar)
		e.predCachedCovar = predCachedCovar
	}
	return e.predCachedCovar
}

func (e SRIFEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	innov := mat64.Formatted(e.Innovation(), mat64.Prefix("  "))
	predp := mat64.Formatted(e.PredCovariance(), mat64.Prefix("   "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nP-=%v\ni=%v\n}", state, meas, covar, predp, innov)
}

// NewSRIFEstimate initializes a new SRIFEstimate.
// NOTE: R0 and predR0 are mat64.Dense for simplicity of implementation, but they should be symmetric.
func NewSRIFEstimate(Φ *mat64.Dense, sqinfoState, meas, innov, Δobs *mat64.Vector, R0, predR0 *mat64.Dense) SRIFEstimate {
	return SRIFEstimate{Φ, sqinfoState, meas, innov, Δobs, nil, R0, predR0, nil, nil}
}
