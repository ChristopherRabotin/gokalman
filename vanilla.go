package gokalman

// TODO: Get rid of Q and R and instead one should be able to pass the exact w and v (as seen in class today).
// Also, if Q and R and provided, then you gonum/stat to generate noise, but that, in "reality" only happens when
// you are simulating the KF not the actual application of the KF.

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
// - F: state update matrix
// - G: control matrix (if all zeros, then control vector will not be used)
// - H: measurement update matrix
// - n: Noise
func NewVanilla(x0 *mat64.Vector, Covar0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*Vanilla, error) {
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
	est0 := VanillaEstimate{x0, mat64.NewVector(rowsH, nil), Covar0, nil}

	return &Vanilla{F, G, H, noise, !IsNil(G), est0, 0}, nil
}

// Vanilla defines a vanilla kalman filter. Use NewVanilla to initialize.
type Vanilla struct {
	F        mat64.Matrix
	G        mat64.Matrix
	H        mat64.Matrix
	Noise    Noise
	needCtrl bool
	prevEst  VanillaEstimate
	step     int
}

// Update implements the KalmanFilter interface.
func (kf *Vanilla) Update(measurement, control *mat64.Vector) (est Estimate, err error) {
	defer func() {
		if r := recover(); r != nil {
			var ok bool
			err, ok = r.(error)
			if !ok {
				err = fmt.Errorf("gokalman: vanilla: %v", r)
			}
		}
	}()

	if err = checkMatDims(control, kf.G, "control (u)", "G", rows2cols); err != nil {
		panic(err)
	}

	if err = checkMatDims(measurement, kf.H, "measurement (y)", "H", rows2rows); err != nil {
		panic(err)
	}

	// Prediction step.
	// \hat{x}_{k+1}^{-}
	var xKp1Minus, xKp1Minus1, xKp1Minus2 mat64.Vector
	xKp1Minus1.MulVec(kf.F, kf.prevEst.State())
	if kf.needCtrl {
		xKp1Minus2.MulVec(kf.G, control)
		xKp1Minus.AddVec(&xKp1Minus1, &xKp1Minus2)
	} else {
		xKp1Minus = xKp1Minus1
	}

	// P_{k+1}^{-}
	var Pkp1Minus, PFt, FPFt mat64.Dense
	PFt.Mul(kf.prevEst.Covariance(), kf.F.T())
	FPFt.Mul(kf.F, &PFt)
	Pkp1Minus.Add(&FPFt, kf.Noise.ProcessMatrix())

	// Compute estimated measurement update \hat{y}_{k}
	var ykHat mat64.Vector
	ykHat.MulVec(kf.H, kf.prevEst.State())
	ykHat.AddVec(&ykHat, kf.Noise.Measurement(kf.step))

	// Kalman gain
	// Kkp1 = P_kp1_minus*H'*inv(H*P_kp1_minus*H'+[Ra 0; 0 Rp]);
	var PHt, HPHt, Kkp1 mat64.Dense
	PHt.Mul(&Pkp1Minus, kf.H.T())
	HPHt.Mul(kf.H, &PHt)
	HPHt.Add(&HPHt, kf.Noise.MeasurementMatrix())
	if ierr := HPHt.Inverse(&HPHt); ierr != nil {
		panic(fmt.Errorf("could not invert `H*P_kp1_minus*H' + R`: %s", ierr))
	}
	Kkp1.Mul(&PHt, &HPHt)

	// Measurement update
	var xkp1Plus, xkp1Plus1 mat64.Vector
	xkp1Plus1.MulVec(kf.H, &xKp1Minus)
	xkp1Plus1.ScaleVec(-1.0, &xkp1Plus1)
	xkp1Plus1.AddVec(measurement, &xkp1Plus1)
	if rX, _ := xkp1Plus1.Dims(); rX == 1 {
		// xkp1Plus1 is a scalar and mat64 won't be happy.
		// The following line will panic if gain unexpectedly has more than one column.
		Kkp1.Scale(xkp1Plus1.At(0, 0), &Kkp1)
		rGain, _ := Kkp1.Dims()
		var xkp1Plus2 mat64.Vector
		xkp1Plus2.AddVec(Kkp1.ColView(0), mat64.NewVector(rGain, nil))
		xkp1Plus1 = xkp1Plus2
	} else {
		xkp1Plus1.MulVec(&Kkp1, &xkp1Plus1)
	}
	xkp1Plus.AddVec(&xKp1Minus, &xkp1Plus1)
	xkp1Plus.AddVec(&xKp1Minus, kf.Noise.Process(kf.step))

	// Pa_kp1_plus = (eye(4) - Kkp1*H)*P_kp1_minus;
	var Pkp1Plus, Kkp1H mat64.Dense
	//var Pkp1PlusSym mat64.SymDense
	Kkp1H.Mul(&Kkp1, kf.H)
	Kkp1H.Scale(-1.0, &Kkp1H)
	n, _ := Kkp1H.Dims()
	Kkp1H.Add(Identity(n), &Kkp1H)
	Pkp1Plus.Mul(&Kkp1H, &Pkp1Minus)
	Pkp1PlusSym, err := AsSymDense(&Pkp1Plus)
	if err != nil {
		panic(err)
	}
	est = VanillaEstimate{&xkp1Plus, &ykHat, Pkp1PlusSym, &Kkp1}
	kf.step++
	return
}

// VanillaEstimate is the output of each update state of the Vanilla KF.
// It implements the Estimate interface.
type VanillaEstimate struct {
	state, meas *mat64.Vector
	covar       mat64.Symmetric
	gain        mat64.Matrix
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e VanillaEstimate) IsWithin2σ() bool {
	for i := 0; i < e.state.Len(); i++ {
		if e.state.At(i, 0) > e.covar.At(i, i) || e.state.At(i, 0) < -1*e.covar.At(i, i) {
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

// Covariance implements the Estimate interface.
func (e VanillaEstimate) Covariance() mat64.Symmetric {
	return e.covar
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
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\n}", state, meas, covar, gain)
}
