package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

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
	var Qprime mat64.Matrix
	if !IsNil(Γ) {
		if err := checkMatDims(Γ, Q, "Γ", "Q", rows2cols); err != nil {
			panic(err)
		}
		if err := checkMatDims(Q, Γ.T(), "Q", "Γ^T", cols2rows); err != nil {
			panic(err)
		}
		var QΓt, ΓQΓt mat64.Dense
		QΓt.Mul(Q, Γ.T())
		ΓQΓt.Mul(Γ, &QΓt)
		Qprime = &ΓQΓt
	} else {
		Qprime = Q
	}

	buffSize := 100
	stateEst := make(chan (mat64.Vector), buffSize)
	measEast := make(chan (mat64.Vector), buffSize)
	measChan := make(chan (mat64.Vector), buffSize)
	covarChan := make(chan (mat64.Symmetric), buffSize)
	var ctrlChan chan (mat64.Vector)

	needCtrl := !IsNil(G)
	if needCtrl {
		ctrlChan = make(chan (mat64.Vector), buffSize)
		ctrlChan <- u0
	}

	// Populate with the initial values.
	stateEst <- x0
	covarChan <- Covar0

	return &Vanilla{stateEst, measEast, measChan, ctrlChan, covarChan, F, G, H, Qprime, R, needCtrl, x0, Covar0}
}

// Vanilla defines a vanilla kalman filter. Use NewVanilla to initialize.
type Vanilla struct {
	StateEst  chan (mat64.Vector) // state estimate channel
	MeasEst   chan (mat64.Vector) // measurement estimate channel
	MeasChan  chan (mat64.Vector) // measurement channel
	CtrlChan  chan (mat64.Vector) // control channel
	CovarChan chan (mat64.Symmetric)
	F         mat64.Matrix
	G         mat64.Matrix
	H         mat64.Matrix
	Q         mat64.Matrix // Stores Γ*Q*Γ^T if Γ was provided.
	R         mat64.Matrix
	needCtrl  bool
	prevState mat64.Vector    // local copy of the previous \hat{x}^{+}
	prevCovar mat64.Symmetric // local copy of the previous covar
}

// Estimate is a blocking function which waits on a measurement and (eventually)
// a control vector. It will return as soon as the measurement channel MeasChan
// is closed.
func (kf *Vanilla) Estimate() {
	// Prediction step.
	// \hat{x}_{k+1}^{-}
	var xKp1Minus, xKp1Minus1, xKp1Minus2 mat64.Vector
	xKp1Minus1.MulVec(kf.F, &kf.prevState)
	if kf.needCtrl {
		ctrlU := <-kf.CtrlChan
		xKp1Minus2.MulVec(kf.G, &ctrlU)
		xKp1Minus.AddVec(&xKp1Minus1, &xKp1Minus2)
	} else {
		xKp1Minus = xKp1Minus1
	}

	// P_{k+1}^{-}
	var Pkp1Minus, PFt, FPFt mat64.Dense
	PFt.Mul(kf.prevCovar, kf.F.T())
	FPFt.Mul(kf.F, &PFt)
	Pkp1Minus.Add(&FPFt, kf.Q)

	//TODO: \hat{y}_{k+1}

	// Kalman gain
	// Kkp1 = P_kp1_minus*H'*inv(H*P_kp1_minus*H'+[Ra 0; 0 Rp]);
	var PHt, HPHt, Kkp1 mat64.Dense
	PHt.Mul(&Pkp1Minus, kf.H.T())
	HPHt.Mul(kf.H, &PHt)
	HPHt.Add(&HPHt, kf.R)
	if err := HPHt.Inverse(&HPHt); err != nil {
		panic(fmt.Errorf("could not invert `H*P_kp1_minus*H' + R`: %s", err))
	}
	Kkp1.Mul(&PHt, &HPHt)

	// Measurement update
	ykp1, more := <-kf.MeasChan
	if !more {
		// Channel was closed, let's stop estimating.
		return
	}
	var xkp1Plus, xkp1Plus1 mat64.Vector
	xkp1Plus1.MulVec(kf.H, &xKp1Minus)
	xkp1Plus1.ScaleVec(-1.0, &xkp1Plus1)
	xkp1Plus1.AddVec(&ykp1, &xkp1Plus1)
	xkp1Plus1.MulVec(&Kkp1, &xkp1Plus1)
	xkp1Plus.AddVec(&xKp1Minus, &xkp1Plus1)

	// Pa_kp1_plus = (eye(4) - Kkp1*H)*P_kp1_minus;
	var Pkp1Plus, Kkp1H mat64.Dense
	//var Pkp1PlusSym mat64.SymDense
	Kkp1H.Mul(&Kkp1, kf.H)
	Kkp1H.Scale(-1.0, &Kkp1H)
	n, _ := Kkp1H.Dims()
	id := Identity(n)
	Kkp1H.Add(id, &Kkp1H)
	Pkp1Plus.Mul(&Kkp1H, &Pkp1Minus)
	Pkp1PlusSym, err := AsSymDense(&Pkp1Plus)
	if err != nil {
		panic(err)
	}

	// Push onto channels
	kf.StateEst <- xkp1Plus
	kf.CovarChan <- Pkp1PlusSym
}

// Stop closes all channels
func (kf *Vanilla) Stop() {
	close(kf.StateEst)
	close(kf.CovarChan)
	close(kf.MeasChan)
	close(kf.MeasEst)
	if kf.needCtrl {
		close(kf.CtrlChan)
	}
}
