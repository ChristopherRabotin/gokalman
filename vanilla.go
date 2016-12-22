package gokalman

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewVanilla returns a new Vanilla KF. To get the next estimate, simply push to
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
func NewVanilla(x0 *mat64.Vector, Covar0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*Vanilla, *VanillaEstimate, error) {
	// Let's check the dimensions of everything here to panic ASAP.
	if err := checkMatDims(x0, Covar0, "x0", "Covar0", rows2cols); err != nil {
		return nil, nil, err
	}
	if err := checkMatDims(F, Covar0, "F", "Covar0", rows2cols); err != nil {
		return nil, nil, err
	}
	if err := checkMatDims(H, x0, "H", "x0", cols2rows); err != nil {
		return nil, nil, err
	}

	// Populate with the initial values.
	rowsH, _ := H.Dims()
	cr, _ := Covar0.Dims()
	predCovar := mat64.NewSymDense(cr, nil)
	est0 := VanillaEstimate{x0, mat64.NewVector(rowsH, nil), mat64.NewVector(rowsH, nil), Covar0, predCovar, nil}

	return &Vanilla{F, G, H, noise, !IsNil(G), est0, 0, false}, &est0, nil
}

// NewPurePredictorVanilla returns a new Vanilla KF which only does prediction.
func NewPurePredictorVanilla(x0 *mat64.Vector, Covar0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*Vanilla, *VanillaEstimate, error) {
	// Let's check the dimensions of everything here to panic ASAP.
	if err := checkMatDims(x0, Covar0, "x0", "Covar0", rows2cols); err != nil {
		return nil, nil, err
	}
	if err := checkMatDims(F, Covar0, "F", "Covar0", rows2cols); err != nil {
		return nil, nil, err
	}
	if err := checkMatDims(H, x0, "H", "x0", cols2rows); err != nil {
		return nil, nil, err
	}

	// Populate with the initial values.
	rowsH, _ := H.Dims()
	cr, _ := Covar0.Dims()
	predCovar := mat64.NewSymDense(cr, nil)
	est0 := VanillaEstimate{x0, mat64.NewVector(rowsH, nil), mat64.NewVector(rowsH, nil), Covar0, predCovar, nil}

	return &Vanilla{F, G, H, noise, !IsNil(G), est0, 0, true}, &est0, nil
}

// Vanilla defines a vanilla kalman filter. Use NewVanilla to initialize.
type Vanilla struct {
	F              mat64.Matrix
	G              mat64.Matrix
	H              mat64.Matrix
	Noise          Noise
	needCtrl       bool
	prevEst        VanillaEstimate
	step           int
	predictionOnly bool
}

func (kf *Vanilla) String() string {
	return fmt.Sprintf("F=%v\nG=%v\nH=%v\n%s", mat64.Formatted(kf.F, mat64.Prefix("  ")), mat64.Formatted(kf.G, mat64.Prefix("  ")), mat64.Formatted(kf.H, mat64.Prefix("  ")), kf.Noise)
}

// GetStateTransition returns the F matrix.
func (kf *Vanilla) GetStateTransition() mat64.Matrix {
	return kf.F
}

// GetInputControl returns the G matrix.
func (kf *Vanilla) GetInputControl() mat64.Matrix {
	return kf.G
}

// GetMeasurementMatrix returns the H matrix.
func (kf *Vanilla) GetMeasurementMatrix() mat64.Matrix {
	return kf.H
}

// SetStateTransition updates the F matrix.
func (kf *Vanilla) SetStateTransition(F mat64.Matrix) {
	kf.F = F
}

// SetInputControl updates the F matrix.
func (kf *Vanilla) SetInputControl(G mat64.Matrix) {
	kf.G = G
}

// SetMeasurementMatrix updates the H matrix.
func (kf *Vanilla) SetMeasurementMatrix(H mat64.Matrix) {
	kf.H = H
}

// SetNoise updates the Noise.
func (kf *Vanilla) SetNoise(n Noise) {
	kf.Noise = n
}

// GetNoise updates the F matrix.
func (kf *Vanilla) GetNoise() Noise {
	return kf.Noise
}

// Update implements the KalmanFilter interface.
func (kf *Vanilla) Update(measurement, control *mat64.Vector) (est Estimate, err error) {
	if err = checkMatDims(control, kf.G, "control (u)", "G", rows2cols); kf.needCtrl && err != nil {
		return nil, err
	}

	if err = checkMatDims(measurement, kf.H, "measurement (y)", "H", rows2rows); err != nil {
		return nil, err
	}

	// Prediction step.
	var xKp1Minus, xKp1Minus1, xKp1Minus2 mat64.Vector
	xKp1Minus1.MulVec(kf.F, kf.prevEst.State())
	if kf.needCtrl {
		xKp1Minus2.MulVec(kf.G, control)
		xKp1Minus.AddVec(&xKp1Minus1, &xKp1Minus2)
	} else {
		xKp1Minus = xKp1Minus1
	}

	// P_{k+1}^{-}
	var Pkp1Minus, FP, FPFt mat64.Dense
	FP.Mul(kf.F, kf.prevEst.Covariance())
	FPFt.Mul(&FP, kf.F.T())
	Pkp1Minus.Add(&FPFt, kf.Noise.ProcessMatrix())

	// Compute estimated measurement update \hat{y}_{k}
	var ykHat mat64.Vector
	ykHat.MulVec(kf.H, kf.prevEst.State())
	ykHat.AddVec(&ykHat, kf.Noise.Measurement(kf.step))

	// Kalman gain
	var PHt, HPHt, Kkp1 mat64.Dense
	PHt.Mul(&Pkp1Minus, kf.H.T())
	HPHt.Mul(kf.H, &PHt)
	HPHt.Add(&HPHt, kf.Noise.MeasurementMatrix())
	if ierr := HPHt.Inverse(&HPHt); ierr != nil {
		panic(fmt.Errorf("could not invert `H*P_kp1_minus*H' + R`: %s", ierr))
	}
	Kkp1.Mul(&PHt, &HPHt)

	if kf.predictionOnly {
		// Note that in the case of a pure prediction, we set the prediction
		// covariance and the covariance to Pkp1Minus.
		Pkp1MinusSym, _ := AsSymDense(&Pkp1Minus)
		rowsH, _ := kf.H.Dims()
		est = VanillaEstimate{&xKp1Minus, &ykHat, mat64.NewVector(rowsH, nil), Pkp1MinusSym, Pkp1MinusSym, &Kkp1}
		kf.prevEst = est.(VanillaEstimate)
		kf.step++
		return
	}

	// Measurement update
	var innov, xkp1Plus, xkp1Plus1, xkp1Plus2 mat64.Vector
	xkp1Plus1.MulVec(kf.H, &xKp1Minus)    // Predicted measurement
	innov.SubVec(measurement, &xkp1Plus1) // Innovation vector
	if rX, _ := innov.Dims(); rX == 1 {
		// xkp1Plus1 is a scalar and mat64 won't be happy, so fiddle around to get a vector.
		var sKkp1 mat64.Dense
		sKkp1.Scale(innov.At(0, 0), &Kkp1)
		rGain, _ := sKkp1.Dims()
		xkp1Plus2.AddVec(sKkp1.ColView(0), mat64.NewVector(rGain, nil))
	} else {
		xkp1Plus2.MulVec(&Kkp1, &innov)
	}
	xkp1Plus.AddVec(&xKp1Minus, &xkp1Plus2)
	xkp1Plus.AddVec(&xkp1Plus, kf.Noise.Process(kf.step))

	var Pkp1Plus, Pkp1Plus1, Kkp1H, Kkp1R, Kkp1RKkp1 mat64.Dense
	Kkp1H.Mul(&Kkp1, kf.H)
	n, _ := Kkp1H.Dims()
	Kkp1H.Sub(Identity(n), &Kkp1H)
	Pkp1Plus1.Mul(&Kkp1H, &Pkp1Minus)
	Pkp1Plus.Mul(&Pkp1Plus1, Kkp1H.T())
	Kkp1R.Mul(&Kkp1, kf.Noise.MeasurementMatrix())
	Kkp1RKkp1.Mul(&Kkp1R, Kkp1.T())
	Pkp1Plus.Add(&Pkp1Plus, &Kkp1RKkp1)

	Pkp1MinusSym, err := AsSymDense(&Pkp1Minus)
	if err != nil {
		return nil, err
	}

	Pkp1PlusSym, err := AsSymDense(&Pkp1Plus)
	if err != nil {
		return nil, err
	}
	est = VanillaEstimate{&xkp1Plus, &ykHat, &innov, Pkp1PlusSym, Pkp1MinusSym, &Kkp1}
	kf.prevEst = est.(VanillaEstimate)
	kf.step++
	return
}

// VanillaEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type VanillaEstimate struct {
	state, meas, innovation *mat64.Vector
	covar, predCovar        mat64.Symmetric
	gain                    mat64.Matrix
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e VanillaEstimate) IsWithin2σ() bool {
	for i := 0; i < e.state.Len(); i++ {
		twoσ := 2 * math.Sqrt(e.covar.At(i, i))
		if e.state.At(i, 0) > twoσ || e.state.At(i, 0) < -twoσ {
			return false
		}
	}
	return true
}

// State implements the Estimate interface.
func (e VanillaEstimate) State() *mat64.Vector {
	return e.state
}

// Measurement implements the Estimate interface.
func (e VanillaEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Innovation implements the Estimate interface.
func (e VanillaEstimate) Innovation() *mat64.Vector {
	return e.innovation
}

// Covariance implements the Estimate interface.
func (e VanillaEstimate) Covariance() mat64.Symmetric {
	return e.covar
}

// PredCovariance implements the Estimate interface.
func (e VanillaEstimate) PredCovariance() mat64.Symmetric {
	return e.predCovar
}

// Gain the Estimate interface.
func (e VanillaEstimate) Gain() mat64.Matrix {
	return e.gain
}

func (e VanillaEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	innov := mat64.Formatted(e.Innovation(), mat64.Prefix("  "))
	predp := mat64.Formatted(e.PredCovariance(), mat64.Prefix("  "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\nP-=%v\ni=%v\n}", state, meas, covar, gain, predp, innov)
}
