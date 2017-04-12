package gokalman

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewSRIF returns a new Square Root Information Filter.
// It uses the algorithms from "Statistical Orbit determination" by Tapley, Schutz & Born.
// Set upperTriangular to `false` to use the lower triangular of the Cholesky decomposition instead of the upper.
func NewSRIF(x0 *mat64.Vector, P0 mat64.Symmetric, measSize int, upperTriangular bool, n Noise) (*SRIF, *SRIFEstimate, error) {
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

	// Compute the square root of the measurement noise.
	var sqrtRchol mat64.Cholesky
	sqrtRchol.Factorize(n.MeasurementMatrix())
	var sqrtMeasNoise mat64.TriDense
	sqrtMeasNoise.LFromCholesky(&sqrtRchol)
	var sqrtInvNoise mat64.Dense
	if err := sqrtInvNoise.Inverse(&sqrtMeasNoise); err != nil {
		return nil, nil, err
	}
	// Populate with the initial values.
	est0 := NewSRIFEstimate(nil, b0, nil, nil, &R0, &R0)
	return &SRIF{nil, nil, &sqrtMeasNoise, est0, true, measSize, 0}, &est0, nil
}

// SRIF defines a square root information filter for non-linear dynamical systems. Use NewSquareRootInformation to initialize.
type SRIF struct {
	Φ, Htilde    *mat64.Dense
	sqrtInvNoise mat64.Matrix
	prevEst      SRIFEstimate
	locked       bool // Locks the KF to ensure Prepare is called.
	measSize     int  // Stores the measurement vector size, needed only for Predict()
	step         int
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
		kf.prevEst = NewSRIFEstimate(kf.Φ, &bBar, mat64.NewVector(kf.measSize, nil), mat64.NewVector(kf.measSize, nil), &RBar, &RBar)
		kf.step++
		kf.locked = true
		return
	}
	// Compute observation deviation y
	var y mat64.Vector
	y.SubVec(realObservation, computedObservation)
	// Whiten the H and y
	var Htilde mat64.Dense
	Htilde.Mul(kf.sqrtInvNoise, kf.Htilde)
	y.MulVec(kf.sqrtInvNoise, &y)

	Rk, bk, _, err := HouseholderSRIF(&RBar, &Htilde, &bBar, &y)
	if err != nil {
		return nil, err
	}
	kf.prevEst = NewSRIFEstimate(kf.Φ, bk, realObservation, &y, Rk, &RBar)
	kf.step++
	kf.locked = true
	return
}

// SRIFEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type SRIFEstimate struct {
	Φ                            *mat64.Dense // Used for smoothing
	sqinfoState, meas            *mat64.Vector
	Δobs, cachedState            *mat64.Vector
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
	predp := mat64.Formatted(e.PredCovariance(), mat64.Prefix("   "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nP-=%v\n}", state, meas, covar, predp)
}

// NewSRIFEstimate initializes a new SRIFEstimate.
// NOTE: R0 and predR0 are mat64.Dense for simplicity of implementation, but they should be symmetric.
func NewSRIFEstimate(Φ *mat64.Dense, sqinfoState, meas, Δobs *mat64.Vector, R0, predR0 *mat64.Dense) SRIFEstimate {
	return SRIFEstimate{Φ, sqinfoState, meas, Δobs, nil, R0, predR0, nil, nil}
}

// HouseholderSRIF prepare the matrix and performs the Householder transformation.
func HouseholderSRIF(R, H *mat64.Dense, b, y *mat64.Vector) (*mat64.Dense, *mat64.Vector, *mat64.Vector, error) {
	if err := checkMatDims(R, H, "R", "H", cols2cols); err != nil {
		return nil, nil, nil, err
	}
	if err := checkMatDims(R, b, "R", "b", rows2rows); err != nil {
		return nil, nil, nil, err
	}
	if err := checkMatDims(H, y, "H", "y", rows2rows); err != nil {
		return nil, nil, nil, err
	}
	n, _ := b.Dims()
	m, _ := y.Dims()
	A0 := mat64.NewDense(m+n, n, nil)
	A0.Stack(R, H)
	col := mat64.NewVector(m+n, nil)
	for i := 0; i < m+n; i++ {
		if i < n {
			col.SetVec(i, b.At(i, 0))
		} else {
			col.SetVec(i, y.At(i-n, 0))
		}
	}
	A := mat64.NewDense(m+n, n+1, nil)
	A.Augment(A0, col)

	HouseholderTransf(A, n, m)

	// Extract the data.
	Rk := A.Slice(0, n, 0, n)
	bkMat := A.Slice(n, n+1, 0, n)
	bk := mat64.NewVector(n, nil)
	for i := 0; i < n; i++ {
		bk.SetVec(i, bkMat.At(i, 0))
	}
	ekMat := A.Slice(n, n+1, n, m)
	ek := mat64.NewVector(m-n, nil)
	for i := 0; i < n; i++ {
		ek.SetVec(i, ekMat.At(i, 0))
	}

	return Rk.(*mat64.Dense), bk, ek, nil
}
