package gokalman

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewSquareRootInformation returns a new Square Root Information Filter.
// It uses the algorithms from "Statistical Orbit determination" by Tapley, Schutz & Born.
// It also uses the Householder algorithm.
func NewSquareRootInformation(x0 *mat64.Vector, P0 mat64.Symmetric, noise Noise, measSize int) (*SRIF, *SRIFEstimate, error) {
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
	R0tri.LFromCholesky(&R0chol)
	var R0dense mat64.Dense
	R0dense.Clone(&R0tri)
	R0, _ := AsSymDense(&R0dense)

	b0 := mat64.NewVector(r, nil)
	b0.MulVec(&R0dense, x0)

	// Populate with the initial values.
	est0 := NewSRIFEstimate(nil, nil, b0, nil, nil, nil, R0, R0)
	return &SRIF{nil, nil, nil, est0, true, measSize, 0}, &est0, nil
}

// SRIF defines a square root information filter for non-linear dynamical systems. Use NewSquareRootInformation to initialize.
type SRIF struct {
	Φ, Htilde, Γ *mat64.Dense
	prevEst      SRIFEstimate
	locked       bool // Locks the KF to ensure Prepare is called.
	measSize     int  // Stores the measurement vector size, needed only for Predict()
	step         int
}

// SRIFEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type SRIFEstimate struct {
	Φ, Γ                         *mat64.Dense // Used for smoothing
	sqinfoState, meas            *mat64.Vector
	innov, Δobs, cachedState     *mat64.Vector
	matR0, predMatR0             mat64.Symmetric
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
func NewSRIFEstimate(Φ, Γ *mat64.Dense, sqinfoState, meas, innov, Δobs *mat64.Vector, R0, predR0 mat64.Symmetric) SRIFEstimate {
	return SRIFEstimate{Φ, Γ, sqinfoState, meas, innov, Δobs, nil, R0, predR0, nil, nil}
}
