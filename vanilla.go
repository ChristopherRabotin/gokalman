package gokalman

import "github.com/gonum/matrix/mat64"

// NewVanilla returns a new Vanilla KF. To get the next estimate, simply push to
// the MeasChan the next measurement and read from StateEst and MeasEst to get
// the next state estimate (\hat{x}_{k+1}^{+}) and next measurement estimate (\hat{y}_{k+1}).
// The Covar channel stores the next covariance of the system (P_{k+1}^{+}).
// Parameters:
// - x0: initial state
// - Covar0: initial covariance matrix
// - F: state update matrix
// - G: control matrix
// - H: measurement update matrix
// - Γ: process noise input matrix
// - Q: process noise multiplier matrix
// - R: measurement noise matrix
func NewVanilla(x0 mat64.Vector, Covar0 mat64.Symmetric, F, G, H, Γ, Q, R mat64.Matrix) *Vanilla {
	/*n, _ := x0.Dims()
	if nP, _ := Covar0.Dims(); nP != n {
		panic(mat64.)
	}*/
	return nil
}

// Vanilla defines a vanilla kalman filter.
type Vanilla struct {
	StateEst chan (mat64.Vector) // state estimate channel
	MeasEst  chan (mat64.Vector) // measurement estimate channel
	MeasChan chan (mat64.Vector) // measurement channel.
	Covar    chan (mat64.Symmetric)
	F        mat64.Matrix
	G        mat64.Matrix
	H        mat64.Matrix
	Γ        mat64.Matrix
	Q        mat64.Matrix
}
