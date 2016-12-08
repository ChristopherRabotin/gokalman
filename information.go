package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// NewInformation returns a new Vanilla KF. To get the next estimate, simply push to
// the MeasChan the next measurement and read from StateEst and MeasEst to get
// the next state estimate (\hat{x}_{k+1}^{+}) and next measurement estimate (\hat{y}_{k+1}).
// The Covar channel stores the next covariance of the system (P_{k+1}^{+}).
// Parameters:
// - x0: initial state
// - Covar0: initial covariance matrix
// - F: state update matrix
// - G: control matrix (if all zeros, then control vector will not be used)
// - H: measurement update matrix
// - n: Noise
func NewInformation(x0 *mat64.Vector, Covar0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*Information, error) {
	// Let's check the dimensions of everything here to panic ASAP.
	if err := checkMatDims(x0, Covar0, "x0", "Covar0", rows2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(F, Covar0, "F", "Covar0", rows2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(H, x0, "H", "x0", cols2rows); err != nil {
		return nil, err
	}

	// Populate with the initial values.
	rowsH, _ := H.Dims()
	est0 := NewInformationEstimate(x0, mat64.NewVector(rowsH, nil), Covar0)

	var Finv mat64.Dense
	if err := Finv.Inverse(mat64.DenseCopyOf(F)); err != nil {
		panic(fmt.Errorf("F not invertible: %s", err))
	}

	var Qinv mat64.Dense
	if err := Qinv.Inverse(mat64.DenseCopyOf(noise.ProcessMatrix())); err != nil {
		panic(fmt.Errorf("Q not invertible: %s", err))
	}
	var Rinv mat64.Dense
	if err := Rinv.Inverse(mat64.DenseCopyOf(noise.MeasurementMatrix())); err != nil {
		panic(fmt.Errorf("R not invertible: %s", err))
	}

	return &Information{&Finv, G, H, &Qinv, &Rinv, noise, !IsNil(G), *est0, 0}, nil
}

// Information defines a vanilla kalman filter. Use NewVanilla to initialize.
type Information struct {
	Finv     mat64.Matrix
	G        mat64.Matrix
	H        mat64.Matrix
	Qinv     mat64.Matrix
	Rinv     mat64.Matrix
	Noise    Noise
	needCtrl bool
	prevEst  InformationEstimate
	step     int
}

// Update implements the KalmanFilter interface.
func (kf *Information) Update(measurement, control *mat64.Vector) (est Estimate, err error) {
	if err = checkMatDims(control, kf.G, "control (u)", "G", rows2cols); kf.needCtrl && err != nil {
		return nil, err
	}

	if err = checkMatDims(measurement, kf.H, "measurement (y)", "H", rows2rows); err != nil {
		return nil, err
	}

	// zMat computation
	var zMat mat64.Dense
	zMat.Mul(kf.Finv, kf.prevEst.infoState)
	zMat.Mul(kf.Finv, &zMat)

	// Prediction step.
	// \hat{i}_{k+1}^{-}
	var zkzkqi mat64.Dense
	/*
		zr, zc := zMat.Dims()
		qr, qc := kf.Qinv.Dims()
		fmt.Printf("%d!=%d? \t %d!=%d?", zr, qr, zc, qc)
	*/
	zkzkqi.Add(&zMat, kf.Qinv)
	zkzkqi.Mul(&zMat, &zkzkqi)
	rzk, _ := zkzkqi.Dims()
	var iKp1Minus, iKp1Minus1 mat64.Vector
	iKp1Minus.MulVec(kf.Finv.T(), kf.prevEst.infoState)
	if kf.needCtrl {
		iKp1Minus1.MulVec(kf.G, control)
		iKp1Minus1.MulVec(&zMat, &iKp1Minus1)
		iKp1Minus.AddVec(&iKp1Minus, &iKp1Minus1)
	}
	var iKp1MinusM mat64.Dense
	iKp1MinusM.Add(Identity(rzk), &zkzkqi)
	iKp1Minus.MulVec(&iKp1MinusM, &iKp1Minus)

	// I_{k+1}^{-}
	var Ikp1Minus mat64.Dense
	Ikp1Minus.Mul(&zkzkqi, &zMat)
	Ikp1Minus.Scale(-1.0, &Ikp1Minus)
	Ikp1Minus.Add(&zMat, &Ikp1Minus)

	// TODO: Compute estimated measurement update \hat{y}_{k}
	/*var ykHat mat64.Vector
	ykHat.MulVec(kf.H, kf.prevEst.State())
	ykHat.AddVec(&ykHat, kf.Noise.Measurement(kf.step))*/

	/*
		// Kalman gain -- none in IF?!
		// Kkp1 = P_kp1_minus*H'*inv(H*P_kp1_minus*H'+[Ra 0; 0 Rp]);
		var PHt, HPHt, Kkp1 mat64.Dense
		PHt.Mul(&Pkp1Minus, kf.H.T())
		HPHt.Mul(kf.H, &PHt)
		HPHt.Add(&HPHt, kf.Noise.MeasurementMatrix())
		if ierr := HPHt.Inverse(&HPHt); ierr != nil {
			panic(fmt.Errorf("could not invert `H*P_kp1_minus*H' + R`: %s", ierr))
		}
		Kkp1.Mul(&PHt, &HPHt)
	*/

	// Measurement update
	var ikp1Plus mat64.Vector
	ikp1Plus.MulVec(kf.Rinv, measurement)
	ikp1Plus.MulVec(kf.H.T(), &ikp1Plus)
	ikp1Plus.AddVec(&ikp1Plus, &iKp1Minus)

	// I_{k+1}^{+}
	var Ikp1Plus mat64.Dense
	Ikp1Plus.Mul(kf.Rinv, kf.H)
	Ikp1Plus.Mul(kf.H.T(), &Ikp1Plus)
	Ikp1Plus.Add(kf.prevEst.infoMat, &Ikp1Plus)

	Ikp1PlusSym, err := AsSymDense(&Ikp1Plus)
	if err != nil {
		panic(err)
	}

	est = NewInformationEstimate(&ikp1Plus, measurement, Ikp1PlusSym)
	kf.prevEst = est.(InformationEstimate)
	kf.step++
	return
}

// InformationEstimate is the output of each update state of the Information KF.
// It implements the Estimate interface.
type InformationEstimate struct {
	infoState, meas *mat64.Vector
	infoMat         mat64.Symmetric
	cachedState     *mat64.Vector
	cachedCovar     mat64.Symmetric
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e InformationEstimate) IsWithin2σ() bool {
	state := e.State()
	covar := e.Covariance()
	for i := 0; i < state.Len(); i++ {
		if state.At(i, 0) > covar.At(i, i) || state.At(i, 0) < -covar.At(i, i) {
			return false
		}
	}
	return true
}

// State implements the Estimate interface.
func (e InformationEstimate) State() *mat64.Vector {
	if e.cachedState == nil {
		e.cachedState.MulVec(e.Covariance(), e.infoState)
	}
	return e.cachedState
}

// Measurement implements the Estimate interface.
func (e InformationEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Covariance implements the Estimate interface.
func (e InformationEstimate) Covariance() mat64.Symmetric {
	if e.cachedCovar == nil {
		infoMat := mat64.DenseCopyOf(e.infoMat)
		var tmpCovar mat64.Dense
		err := tmpCovar.Inverse(infoMat)
		if err != nil {
			panic(fmt.Errorf("information matrix is not invertible: %s", err))
		}
		cachedCovar, err := AsSymDense(&tmpCovar)
		if err != nil {
			panic(fmt.Errorf("covariance matrix %s", err))
		}
		e.cachedCovar = cachedCovar
	}
	return e.cachedCovar
}

// Gain the Estimate interface.
func (e InformationEstimate) Gain() mat64.Matrix {
	return nil
}

func (e InformationEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\n}", state, meas, covar, gain)
}

// NewInformationEstimate initializes a new InformationEstimate.
func NewInformationEstimate(infoState, meas *mat64.Vector, infoMat mat64.Symmetric) *InformationEstimate {
	return &InformationEstimate{infoState, meas, infoMat, nil, nil}
}
