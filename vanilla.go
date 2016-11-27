package gokalman

import "github.com/gonum/matrix/mat64"

// NewVanilla returns a new Vanilla KF. To get the next estimate, simply push to
// the MeasChan the next measurement and read from StateEst and MeasEst to get
// the next state estimate (\hat{x}_{k+1}^{+}) and next measurement estimate (\hat{y}_{k+1}).
// The Covar channel stores the next covariance of the system (P_{k+1}^{+}).
// Parameters:
// - x0: initial state
// - Covar0: initial covariance matrix
// - u0: initial control vector
// - F: state update matrix
// - G: control matrix (if all zeros, then control vector will not be used)
// - H: measurement update matrix
// - Γ: process noise input matrix
// - Q: process noise multiplier matrix
// - R: measurement noise matrix
func NewVanilla(x0 mat64.Vector, Covar0 mat64.Symmetric, u0 mat64.Vector, F, G, H, Γ, Q, R mat64.Matrix) *Vanilla {
	// Let's check the dimensions of everything here to panic ASAP.
	if err := checkMatDims(&x0, Covar0, "x0", "Covar0", rows2cols); err != nil {
		panic(err)
	}
	if err := checkMatDims(F, Covar0, "F", "Covar0", rows2cols); err != nil {
		panic(err)
	}
	if err := checkMatDims(Covar0, F.T(), "Covar0", "F^T", cols2rows); err != nil {
		panic(err)
	}
	if err := checkMatDims(H, &x0, "H", "x0", cols2rows); err != nil {
		panic(err)
	}
	if err := checkMatDims(Γ, Q, "Γ", "Q", rows2cols); err != nil {
		panic(err)
	}
	if err := checkMatDims(Q, Γ.T(), "Q", "Γ^T", cols2rows); err != nil {
		panic(err)
	}

	return nil
}

// Vanilla defines a vanilla kalman filter.
type Vanilla struct {
	StateEst chan (mat64.Vector) // state estimate channel
	MeasEst  chan (mat64.Vector) // measurement estimate channel
	MeasChan chan (mat64.Vector) // measurement channel.
	CtrlChan chan (mat64.Vector) // control channel.
	Covar    chan (mat64.Symmetric)
	F        mat64.Matrix
	G        mat64.Matrix
	H        mat64.Matrix
	Γ        mat64.Matrix
	Q        mat64.Matrix
	needCtrl bool
}
