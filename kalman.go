package gokalman

import "github.com/gonum/matrix/mat64"

// FilterType allows for quick comparison of filters.
type FilterType uint8

const (
	// CKFType definition would be a tautology
	CKFType FilterType = iota + 1
	// EKFType definition would be a tautology
	EKFType
	// UKFType definition would be a tautology
	UKFType
	// SRIFType definition would be a tautology
	SRIFType
)

// LDKF defines a linear dynamics Kalman Filter.
type LDKF interface {
	Update(measurement, control *mat64.Vector) (Estimate, error)
	GetNoise() Noise
	GetStateTransition() mat64.Matrix
	GetInputControl() mat64.Matrix
	GetMeasurementMatrix() mat64.Matrix
	SetStateTransition(mat64.Matrix)
	SetInputControl(mat64.Matrix)
	SetMeasurementMatrix(mat64.Matrix)
	SetNoise(Noise)
	Reset()
	String() string
}

// NLDKF defines a non-linear dynamics Kalman Filter.
// Operates and is architectured slightly differently than LDKF.
type NLDKF interface {
	Prepare(Φ, Htilde *mat64.Dense)
	Predict() (est Estimate, err error)
	Update(realObservation, computedObservation *mat64.Vector) (est Estimate, err error)
	EKFEnabled() bool
	EnableEKF()
	DisableEKF()
	PreparePNT(Γ *mat64.Dense)
	SetNoise(n Noise)
}

// Estimate is returned from Update() in any KF.
// This allows to avoid some computations in other filters, e.g. in the Information filter.
type Estimate interface {
	IsWithinNσ(N float64) bool       // IsWithinNσ returns whether the estimation is within the N*σ bounds.
	State() *mat64.Vector            // Returns \hat{x}_{k+1}^{+}
	Measurement() *mat64.Vector      // Returns \hat{y}_{k}^{+}
	Innovation() *mat64.Vector       // Returns y_{k} - H*\hat{x}_{k+1}^{-}
	Covariance() mat64.Symmetric     // Return P_{k+1}^{+}
	PredCovariance() mat64.Symmetric // Return P_{k+1}^{-}
	String() string                  // Must implement the stringer interface.
}
