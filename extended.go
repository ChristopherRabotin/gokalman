package gokalman

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// WARNING: This is an extended Kalman filter designed for an orbital determination course
// and the implementation is specifically that of "Tapley, Schutz and Born". It may not work
// in other cases.

// NewExtended returns a new Extended KF. To get the next estimate, simply push to
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
func NewExtended(x0 *mat64.Vector, Covar0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*Extended, *ExtendedEstimate, error) {
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
	est0 := ExtendedEstimate{x0, mat64.NewVector(rowsH, nil), mat64.NewVector(rowsH, nil), Covar0, predCovar, nil}

	return &Extended{F, G, H, noise, !IsNil(G), est0, est0, 0}, &est0, nil
}

// Extended defines a vanilla kalman filter. Use NewExtended to initialize.
type Extended struct {
	F                mat64.Matrix
	G                mat64.Matrix
	H                mat64.Matrix
	Noise            Noise
	needCtrl         bool
	prevEst, initEst ExtendedEstimate
	step             int
}

func (kf *Extended) String() string {
	return fmt.Sprintf("F=%v\nG=%v\nH=%v\n%s", mat64.Formatted(kf.F, mat64.Prefix("  ")), mat64.Formatted(kf.G, mat64.Prefix("  ")), mat64.Formatted(kf.H, mat64.Prefix("  ")), kf.Noise)
}

// GetStateTransition returns the F matrix.
func (kf *Extended) GetStateTransition() mat64.Matrix {
	return kf.F
}

// GetInputControl returns the G matrix.
func (kf *Extended) GetInputControl() mat64.Matrix {
	return kf.G
}

// GetMeasurementMatrix returns the H matrix.
func (kf *Extended) GetMeasurementMatrix() mat64.Matrix {
	return kf.H
}

// SetStateTransition updates the F matrix.
func (kf *Extended) SetStateTransition(F mat64.Matrix) {
	kf.F = F
}

// SetInputControl updates the F matrix.
func (kf *Extended) SetInputControl(G mat64.Matrix) {
	kf.G = G
}

// SetMeasurementMatrix updates the H matrix.
func (kf *Extended) SetMeasurementMatrix(H mat64.Matrix) {
	kf.H = H
}

// SetNoise updates the Noise.
func (kf *Extended) SetNoise(n Noise) {
	kf.Noise = n
}

// GetNoise updates the F matrix.
func (kf *Extended) GetNoise() Noise {
	return kf.Noise
}

// Reset reinitializes the KF with its initial estimate.
func (kf *Extended) Reset() {
	kf.prevEst = kf.initEst
	kf.step = 0
	kf.Noise.Reset()
}

// Update implements the KalmanFilter interface.
func (kf *Extended) Update(measurement, control *mat64.Vector) (est Estimate, err error) {
	if err = checkMatDims(control, kf.G, "control (u)", "G", rows2cols); kf.needCtrl && err != nil {
		return nil, err
	}

	if err = checkMatDims(measurement, kf.H, "measurement (y)", "H", rows2rows); err != nil {
		return nil, err
	}

	// Kalman gain
	var PHt, HPHt, Kkp1 mat64.Dense
	PHt.Mul(kf.prevEst.Covariance(), kf.H.T())
	HPHt.Mul(kf.H, &PHt)
	HPHt.Add(&HPHt, kf.Noise.MeasurementMatrix())
	if ierr := HPHt.Inverse(&HPHt); ierr != nil {
		//panic(fmt.Errorf("could not invert `H*P_kp1_minus*H' + R`: %s", ierr))
		return nil, fmt.Errorf("could not invert `H*P_kp1_minus*H' + R`: %s", ierr)
	}
	Kkp1.Mul(&PHt, &HPHt)

	// Measurement update
	var xkp1Plus mat64.Vector
	xkp1Plus.MulVec(&Kkp1, measurement)

	var Pkp1Plus, Pkp1Plus1, Kkp1H, Kkp1R, Kkp1RKkp1 mat64.Dense
	Kkp1H.Mul(&Kkp1, kf.H)
	n, _ := Kkp1H.Dims()
	Kkp1H.Sub(Identity(n), &Kkp1H)
	Pkp1Plus1.Mul(&Kkp1H, kf.prevEst.Covariance())
	Pkp1Plus.Mul(&Pkp1Plus1, Kkp1H.T())
	Kkp1R.Mul(&Kkp1, kf.Noise.MeasurementMatrix())
	Kkp1RKkp1.Mul(&Kkp1R, Kkp1.T())
	Pkp1Plus.Add(&Pkp1Plus, &Kkp1RKkp1)

	Pkp1PlusSym, err := AsSymDense(&Pkp1Plus)
	if err != nil {
		return nil, err
	}
	est = ExtendedEstimate{&xkp1Plus, mat64.NewVector(2, nil), mat64.NewVector(2, nil), Pkp1PlusSym, kf.prevEst.Covariance(), &Kkp1}
	kf.prevEst = est.(ExtendedEstimate)
	kf.step++
	return
}

// ExtendedEstimate is the output of each update state of the Extended KF.
// It implements the Estimate interface.
type ExtendedEstimate struct {
	state, meas, innovation *mat64.Vector
	covar, predCovar        mat64.Symmetric
	gain                    mat64.Matrix
}

// IsWithinNσ returns whether the estimation is within the 2σ bounds.
func (e ExtendedEstimate) IsWithinNσ(N float64) bool {
	for i := 0; i < e.state.Len(); i++ {
		twoσ := N * math.Sqrt(e.covar.At(i, i))
		if e.state.At(i, 0) > twoσ || e.state.At(i, 0) < -twoσ {
			return false
		}
	}
	return true
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e ExtendedEstimate) IsWithin2σ() bool {
	return e.IsWithinNσ(2)
}

// State implements the Estimate interface.
func (e ExtendedEstimate) State() *mat64.Vector {
	return e.state
}

// Measurement implements the Estimate interface.
func (e ExtendedEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Innovation implements the Estimate interface.
func (e ExtendedEstimate) Innovation() *mat64.Vector {
	return e.innovation
}

// Covariance implements the Estimate interface.
func (e ExtendedEstimate) Covariance() mat64.Symmetric {
	return e.covar
}

// PredCovariance implements the Estimate interface.
func (e ExtendedEstimate) PredCovariance() mat64.Symmetric {
	return e.predCovar
}

// Gain the Estimate interface.
func (e ExtendedEstimate) Gain() mat64.Matrix {
	return e.gain
}

func (e ExtendedEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	innov := mat64.Formatted(e.Innovation(), mat64.Prefix("  "))
	predp := mat64.Formatted(e.PredCovariance(), mat64.Prefix("   "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\nP-=%v\ni=%v\n}", state, meas, covar, gain, predp, innov)
}
