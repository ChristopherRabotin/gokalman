package gokalman

import "github.com/gonum/matrix/mat64"

// KalmanFilter defines a general KF.
type KalmanFilter interface {
	Update(measurement, control *mat64.Vector) (Estimate, error)
	SetF(mat64.Matrix)
	SetG(mat64.Matrix)
	SetH(mat64.Matrix)
	SetNoise(Noise)
	String() string
}

// Estimate is returned from Update() in any KF.
// This allows to avoid some computations in other filters, e.g. in the Information filter.
type Estimate interface {
	IsWithin2σ() bool            // IsWithin2σ returns whether the estimation is within the 2σ bounds.
	State() *mat64.Vector        // Returns \hat{x}_{k+1}^{+}
	Measurement() *mat64.Vector  // Returns \hat{y}_{k}^{+}
	Covariance() mat64.Symmetric // Return P_{k+1}^{+}
	String() string              // Must implement the stringer interface.
}
