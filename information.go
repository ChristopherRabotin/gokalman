package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// NewInformation returns a new Information KF. To get the next estimate, call
// Update() with the next measurement and the control vector. This will return a
// new InformationEstimate which contains everything of this step and an error if any.
// Parameters:
// - i0: initial information state (usually a zero vector)
// - I0: initial information matrix (usually a zero matrix)
// - F: state update matrix
// - G: control matrix (if all zeros, then control vector will not be used)
// - H: measurement update matrix
// - noise: Noise
func NewInformation(i0 *mat64.Vector, I0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*Information, error) {
	// Let's check the dimensions of everything here to panic ASAP.
	if err := checkMatDims(i0, I0, "x0", "Covar0", rows2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(F, I0, "F", "Covar0", rows2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(H, i0, "H", "x0", cols2rows); err != nil {
		return nil, err
	}

	// Populate with the initial values.
	rowsH, _ := H.Dims()
	est0 := NewInformationEstimate(i0, mat64.NewVector(rowsH, nil), I0)

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

	return &Information{&Finv, G, H, &Qinv, &Rinv, noise, !IsNil(G), est0, 0}, nil
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

func (kf *Information) String() string {
	return fmt.Sprintf("inv(F)=%v\nG=%v\nH=%v\n%s", mat64.Formatted(kf.Finv, mat64.Prefix("      ")), mat64.Formatted(kf.G, mat64.Prefix("  ")), mat64.Formatted(kf.H, mat64.Prefix("  ")), kf.Noise)
}

// SetF updates the F matrix.
func (kf *Information) SetF(F mat64.Matrix) {
	var Finv mat64.Dense
	if err := Finv.Inverse(mat64.DenseCopyOf(F)); err != nil {
		panic(fmt.Errorf("F not invertible: %s", err))
	}
	kf.Finv = &Finv
}

// SetG updates the F matrix.
func (kf *Information) SetG(G mat64.Matrix) {
	kf.G = G
}

// SetH updates the F matrix.
func (kf *Information) SetH(H mat64.Matrix) {
	kf.H = H
}

// SetNoise updates the Noise.
func (kf *Information) SetNoise(n Noise) {
	kf.Noise = n
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
	zMat.Mul(kf.Finv, kf.prevEst.infoMat)
	zMat.Mul(kf.Finv.T(), &zMat)

	// Prediction step.
	// \hat{i}_{k+1}^{-}
	var zkzkqi mat64.Dense

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
	var HTR mat64.Dense
	if rR, cR := kf.Rinv.Dims(); rR == 1 && cR == 1 {
		// Rinv is a scalar and mat64 won't be happy.
		HTR.Scale(kf.Rinv.At(0, 0), kf.H.T())
	} else {
		HTR.Mul(kf.H.T(), kf.Rinv)
	}

	var ikp1Plus mat64.Vector
	ikp1Plus.MulVec(&HTR, measurement)
	ikp1Plus.AddVec(&ikp1Plus, &iKp1Minus)

	// I_{k+1}^{+}
	var Ikp1Plus mat64.Dense
	Ikp1Plus.Mul(&HTR, kf.H)
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
		rState, _ := e.infoState.Dims()
		e.cachedState = mat64.NewVector(rState, nil)
		e.cachedState.MulVec(e.Covariance(), e.infoState)
	}
	return e.cachedState
}

// Measurement implements the Estimate interface.
func (e InformationEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Covariance implements the Estimate interface.
// *NOTE:* With the IF, one cannot view the covariance matrix until there is enough information.
func (e InformationEstimate) Covariance() mat64.Symmetric {
	if e.cachedCovar == nil {
		rCovar, _ := e.infoMat.Dims()
		e.cachedCovar = mat64.NewSymDense(rCovar, nil)
		infoMat := mat64.DenseCopyOf(e.infoMat)
		var tmpCovar mat64.Dense
		err := tmpCovar.Inverse(infoMat)
		if err != nil {
			fmt.Printf("gokalman: InformationEstimate: information matrix is not (yet) invertible: %s\n", err)
		} else {
			cachedCovar, err := AsSymDense(&tmpCovar)
			if err != nil {
				fmt.Printf("gokalman: InformationEstimate: covariance matrix: %s\n", err)
			} else {
				e.cachedCovar = cachedCovar
			}
		}
	}
	return e.cachedCovar
}

// Gain the Estimate interface. Note that there is no Gain in the Information filter.
func (e InformationEstimate) Gain() mat64.Matrix {
	return mat64.NewDense(1, 1, nil)
}

func (e InformationEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\n}", state, meas, covar, gain)
}

// NewInformationEstimate initializes a new InformationEstimate.
func NewInformationEstimate(infoState, meas *mat64.Vector, infoMat mat64.Symmetric) InformationEstimate {
	return InformationEstimate{infoState, meas, infoMat, nil, nil}
}
