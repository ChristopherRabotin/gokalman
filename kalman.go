package gokalman

import "github.com/gonum/matrix/mat64"

// KalmanFilter defines a general KF.
type KalmanFilter interface {
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

// HybridKalmanFilter defines a hybrid continuous/discrete KF.
// Usage example: statistical orbit determination.
type HybridKalmanFilter interface {
	Prepare(Φ, Htilde *mat64.Dense)
	Update(realObservation, computedObservation *mat64.Vector) (*HybridCKFEstimate, error)
	GetNoise() Noise
	SetNoise(Noise)
	String() string
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
