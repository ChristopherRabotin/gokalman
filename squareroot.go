package gokalman

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// NewSquareRoot returns a new Square Root KF. To get the next estimate, push to
// the MeasChan the next measurement and read from StateEst and MeasEst to get
// the next state estimate (\hat{x}{k+1}^{+}) and next measurement estimate (\hat{y}{k+1}).
// The Covar channel stores the next covariance of the system (P_{k+1}^{+}).
// Parameters:
// - x0: initial state
// - Covar0: initial covariance matrix
// - F: state update matrix
// - G: control matrix (if all zeros, then control vector will not be used)
// - H: measurement update matrix
// - noise: Noise
func NewSquareRoot(x0 *mat64.Vector, P0 mat64.Symmetric, F, G, H mat64.Matrix, noise Noise) (*SquareRoot, error) {
	// Check the dimensions of each matrix to avoid errors.
	if err := checkMatDims(x0, P0, "x0", "P0", rows2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(F, P0, "F", "P0", rows2cols); err != nil {
		return nil, err
	}
	if err := checkMatDims(H, x0, "H", "x0", cols2rows); err != nil {
		return nil, err
	}

	// Get s0 from Covariance
	// Compute the cholesky factorization of the covariance.
	var sqrtP0 mat64.Cholesky
	sqrtP0.Factorize(P0)
	var stddevL mat64.TriDense
	stddevL.LFromCholesky(&sqrtP0)
	var stddev mat64.Dense
	stddev.Clone(&stddevL)

	// Populate with the initial values.
	rowsH, _ := H.Dims()
	est0 := NewSqrtEstimate(x0, mat64.NewVector(rowsH, nil), &stddev, nil)
	// Return the state and estimate to the SquareRoot structure.
	sqrt := SquareRoot{F, G, H, nil, nil, nil, !IsNil(G), est0, 0}
	sqrt.SetNoise(noise) // Computes the Cholesky decompositions of the noise.
	return &sqrt, nil
}

// SquareRoot defines a square root kalman filter. Use NewSqrt to initialize.
type SquareRoot struct {
	F            mat64.Matrix
	G            mat64.Matrix
	H            mat64.Matrix
	Noise        Noise
	sqrtQ, sqrtR mat64.Matrix
	needCtrl     bool
	prevEst      SquareRootEstimate
	step         int
}

// Prints the output.
func (kf *SquareRoot) String() string {
	return fmt.Sprintf("F=%v\nG=%v\nH=%v\n%s", mat64.Formatted(kf.F, mat64.Prefix(" ")), mat64.Formatted(kf.G, mat64.Prefix(" ")), mat64.Formatted(kf.H, mat64.Prefix(" ")), kf.Noise)
}

// SetF updates the F matrix.
func (kf *SquareRoot) SetF(F mat64.Matrix) {
	kf.F = F
}

// SetG updates the G matrix.
func (kf *SquareRoot) SetG(G mat64.Matrix) {
	kf.G = G
}

// SetH updates the H matrix.
func (kf *SquareRoot) SetH(H mat64.Matrix) {
	kf.H = H
}

// SetNoise updates the Noise.
func (kf *SquareRoot) SetNoise(n Noise) {
	// Compute the Cholesky of Q and R only once when the noise is set.
	var sqrtQchol mat64.Cholesky
	sqrtQchol.Factorize(n.ProcessMatrix())
	var sqrtQ mat64.TriDense
	sqrtQ.LFromCholesky(&sqrtQchol)

	var sqrtRchol mat64.Cholesky
	sqrtRchol.Factorize(n.MeasurementMatrix())
	var sqrtR mat64.TriDense
	sqrtR.LFromCholesky(&sqrtRchol)
	kf.Noise = n
	kf.sqrtQ = &sqrtQ
	kf.sqrtR = &sqrtR
}

// Update implements the KalmanFilter interface.
func (kf *SquareRoot) Update(measurement, control *mat64.Vector) (est Estimate, err error) {
	// Check for matrix dimensions errors.
	if err = checkMatDims(control, kf.G, "control (u)", "G", rows2cols); kf.needCtrl && err != nil {
		return nil, err
	}
	if err = checkMatDims(measurement, kf.H, "measurement (y)", "H", rows2rows); err != nil {
		return nil, err
	}

	// Prediction Step //
	// Get xKp1Minus
	var xKp1Minus, xKp1Minus1, xKp1Minus2 mat64.Vector
	xKp1Minus1.MulVec(kf.F, kf.prevEst.State())
	if kf.needCtrl {
		xKp1Minus2.MulVec(kf.G, control)
		xKp1Minus.AddVec(&xKp1Minus1, &xKp1Minus2)
	} else {
		xKp1Minus = xKp1Minus1
	}

	// Get sKplus
	//sKPlus := kf.prevEst.stddev

	// Get sKp1Minus

	// C Matrix
	nState, _ := kf.prevEst.state.Dims()
	cVals := make([]float64, 2*nState*nState, 2*nState*nState)
	var sTFT mat64.Dense
	sTFT.Mul(kf.prevEst.stddev.T(), kf.F.T())
	cValsPos := 0
	sTFTr, sTFTc := sTFT.Dims()
	for i := 0; i < sTFTr; i++ {
		for j := 0; j < sTFTc; j++ {
			cVals[cValsPos] = sTFT.At(i, j)
			cValsPos++
		}
	}
	// Now let's add the sqrtQ elements to the values for C
	sQr, sQc := kf.sqrtQ.Dims()
	for i := 0; i < sQr; i++ {
		for j := 0; j < sQc; j++ {
			cVals[cValsPos] = kf.sqrtQ.T().At(i, j)
			cValsPos++
		}
	}
	C := mat64.NewDense(2*nState, nState, cVals)
	var TcUc mat64.QR
	TcUc.Factorize(C)
	var Uc mat64.Dense
	Uc.RFromQR(&TcUc)

	// Get sKp1Minus from the top block of QR decomposition.
	//skR, skC := kf.prevEst.stddev.Dims()
	skR := nState
	skC := nState
	SKp1Minus := Uc.View(0, 0, skR, skC)

	// Delta Matrix

	// And now let's compute the two bottom blocks of the Delta matrix.
	var SKp1MinusTHT mat64.Dense
	SKp1MinusTHT.Mul(SKp1Minus.T(), kf.H.T())
	shR, shC := SKp1MinusTHT.Dims()
	sRr, sRc := kf.sqrtR.Dims()
	pMeas, _ := measurement.Dims()
	Δ := mat64.NewDense(nState+pMeas, nState+pMeas, nil)
	ΔrMax, ΔcMax := Δ.Dims()
	fmt.Printf("Δ(%d,%d) R'(%d,%d) SK(%d,%d) SH(%d,%d)\nR'=%v\nSK=%v\nSH=%v\n", ΔrMax, ΔcMax, sRr, sRc, skR, skC, shR, shC, mat64.Formatted(SKp1Minus.T(), mat64.Prefix("   ")),
		mat64.Formatted(kf.sqrtR.T(), mat64.Prefix("   ")), mat64.Formatted(&SKp1MinusTHT, mat64.Prefix("   ")))

	// Note that we populate by *column* for simpler logic.
	for Δc := 0; Δc < ΔcMax; Δc++ {
		for Δr := 0; Δr < ΔrMax; Δr++ {
			fmt.Printf("Δ(%d,%d)\n", Δr, Δc)
			if Δc < sRc {
				if Δr < sRr {
					// Still in the upper left, let's set this to the R.T()
					Δ.Set(Δr, Δc, kf.sqrtR.T().At(Δr, Δc))
				} else if Δc < shC {
					Δ.Set(Δr, Δc, SKp1MinusTHT.At(Δr-sRr, Δc))
				} else {
					Δ.Set(Δr, Δc, SKp1Minus.T().At(Δr-skC, Δc-shR))
				}
			} else if Δr < sRr {
				Δ.Set(Δr, Δc, 0)
			} else {
				Δ.Set(Δr, Δc, SKp1Minus.T().At(Δr-sRr, Δc-shC))
			}
		}
	}

	// Extract the UΔ matrix post QR decomposition.
	var TΔUΔ mat64.QR
	TΔUΔ.Factorize(Δ)
	var UΔ mat64.Dense
	UΔ.RFromQR(&TΔUΔ)

	// Extract Skp1Plus first
	UΔR, UΔC := UΔ.Dims()
	// Note that Skp1PlusT is transposed, hence the change of indices.
	Skp1PlusT := UΔ.View(UΔR-skC, UΔC-skR, skC, skR)
	SyyT := UΔ.View(0, 0, pMeas, pMeas)
	Wkp1PlusT := UΔ.View(0, pMeas, UΔR-skC, UΔC-pMeas)

	var Skp1Plus, Syy, Wkp1Plus mat64.Dense
	Skp1Plus.Clone(Skp1PlusT.T())
	Syy.Clone(SyyT.T())
	Wkp1Plus.Clone(Wkp1PlusT.T())

	// Compute Kalman gain.
	var SyyInv mat64.Dense
	if invErr := SyyInv.Inverse(&Syy); err != nil {
		return nil, fmt.Errorf("Syy is not invertible: %s\nSyy=%v", invErr, mat64.Formatted(&SyyInv, mat64.Prefix("    ")))
	}
	var Kkp1 mat64.Dense
	if pMeas == 1 {
		// Then SyyInv is just a scalar.
		Kkp1.Scale(SyyInv.At(0, 0), &Wkp1Plus)
	} else {
		Kkp1.Mul(&Wkp1Plus, &SyyInv)
	}

	// Measurement update
	var xkp1Plus, xkp1Plus1, xkp1Plus2 mat64.Vector
	xkp1Plus1.MulVec(kf.H, &xKp1Minus)
	xkp1Plus1.SubVec(measurement, &xkp1Plus1)
	if rX, _ := xkp1Plus1.Dims(); rX == 1 {
		// xkp1Plus1 is a scalar and mat64 won't be happy.
		var sKkp1 mat64.Dense
		sKkp1.Scale(xkp1Plus1.At(0, 0), &Kkp1)
		rGain, _ := sKkp1.Dims()
		xkp1Plus2.AddVec(sKkp1.ColView(0), mat64.NewVector(rGain, nil))
	} else {
		fmt.Printf("K=%v\nx=%+v\n", mat64.Formatted(&Kkp1, mat64.Prefix("  ")), mat64.Formatted(&xkp1Plus1, mat64.Prefix("  ")))
		xkp1Plus2.MulVec(&Kkp1, &xkp1Plus1)
	}
	xkp1Plus.AddVec(&xKp1Minus, &xkp1Plus2)
	xkp1Plus.AddVec(&xkp1Plus, kf.Noise.Process(kf.step))

	est = NewSqrtEstimate(&xkp1Plus, measurement, &Skp1Plus, &Kkp1)
	kf.prevEst = est.(SquareRootEstimate)
	kf.step++
	return
}

// SquareRootEstimate is the output of each update state of the SquareRoot KF.
// It implements the Estimate interface.
type SquareRootEstimate struct {
	state, meas *mat64.Vector
	stddev      *mat64.Dense
	gain        mat64.Matrix
	cachedCovar mat64.Symmetric
}

// IsWithin2σ returns whether the estimation is within the 2σ bounds.
func (e SquareRootEstimate) IsWithin2σ() bool {
	for i := 0; i < e.state.Len(); i++ {
		twoσ := 2 * math.Sqrt(e.Covariance().At(i, i))
		if e.state.At(i, 0) > twoσ || e.state.At(i, 0) < -twoσ {
			return false
		}
	}
	return true
}

// State implements the Estimate interface.
func (e SquareRootEstimate) State() *mat64.Vector {
	return e.state
}

// Measurement implements the Estimate interface.
func (e SquareRootEstimate) Measurement() *mat64.Vector {
	return e.meas
}

// Covariance implements the Estimate interface.
func (e SquareRootEstimate) Covariance() mat64.Symmetric {
	if e.cachedCovar == nil {
		var covar mat64.Dense
		covar.Mul(e.stddev, e.stddev.T())
		// We don't check whether AsSymDense fails because it Skp1Plus comes from QR,
		// it's the bottom triangle of the upper triangular R. Hence, s*s^T will be symmetric.
		cachedCovar, _ := AsSymDense(&covar)
		e.cachedCovar = cachedCovar
	}
	return e.cachedCovar
}

// Gain the Estimate interface.
func (e SquareRootEstimate) Gain() mat64.Matrix {
	return e.gain
}

func (e SquareRootEstimate) String() string {
	state := mat64.Formatted(e.State(), mat64.Prefix("  "))
	meas := mat64.Formatted(e.Measurement(), mat64.Prefix("  "))
	covar := mat64.Formatted(e.Covariance(), mat64.Prefix("  "))
	gain := mat64.Formatted(e.Gain(), mat64.Prefix("  "))
	return fmt.Sprintf("{\ns=%v\ny=%v\nP=%v\nK=%v\n}", state, meas, covar, gain)
}

// NewSqrtEstimate initializes a new InformationEstimate.
func NewSqrtEstimate(state, meas *mat64.Vector, stddev, gain *mat64.Dense) SquareRootEstimate {
	return SquareRootEstimate{state, meas, stddev, gain, nil}
}
