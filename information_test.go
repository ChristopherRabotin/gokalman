package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestNewInformationEstimateErrors(t *testing.T) {
	est := NewInformationEstimate(mat64.NewVector(1, nil), mat64.NewVector(1, nil), mat64.NewSymDense(1, nil), mat64.NewSymDense(1, nil))
	cov := est.Covariance()
	if !IsNil(cov) {
		t.Fatal("singular information matrix should return an empty covariance matrix")
	}
}

func TestNewInformationErrors(t *testing.T) {
	F, G, _ := Robot1DMatrices()
	H := mat64.NewDense(2, 2, nil)
	x0 := mat64.NewVector(2, nil)
	Covar0 := mat64.NewSymDense(2, []float64{1, 0, 0, 0})
	R := mat64.NewSymDense(1, []float64{0.05})
	Q := mat64.NewSymDense(2, []float64{3e-4, 5e-3, 5e-3, 0.1}) // Q true
	noise := NewNoiseless(Q, R)
	_, _, err := NewInformationFromState(x0, Covar0, F, G, H, noise)
	if err != nil {
		t.Fatal("singular Covar0 failed (should have only displayed a warning)")
	}
	Covar0 = mat64.NewSymDense(3, nil)
	if _, _, err := NewInformation(x0, Covar0, F, G, H, noise); err == nil {
		t.Fatal("x0 and Covar0 of incompatible sizes does not fail")
	}
	x0 = mat64.NewVector(3, nil)
	if _, _, err := NewInformation(x0, Covar0, F, G, H, noise); err == nil {
		t.Fatal("F and Covar0 of incompatible sizes does not fail")
	}
	x0 = mat64.NewVector(2, nil)
	Covar0 = mat64.NewSymDense(2, nil)
	H = mat64.NewDense(3, 3, nil)
	if _, _, err := NewInformation(x0, Covar0, F, G, H, noise); err == nil {
		t.Fatal("H and x0 of incompatible sizes does not fail")
	}
}

func TestInformation(t *testing.T) {
	F, G, Δt := Midterm2Matrices()
	Q := mat64.NewSymDense(3, []float64{2.5e-15, 6.25e-13, (25e-11) / 3, 6.25e-13, (5e-7) / 3, 2.5e-8, (25e-11) / 3, 2.5e-8, 5e-6})
	R := mat64.NewSymDense(1, []float64{0.005 / Δt})
	H := mat64.NewDense(1, 3, []float64{1, 0, 0})
	noise := NewAWGN(Q, R)
	x0 := mat64.NewVector(3, []float64{0, 0.35, 0})
	P0 := ScaledIdentity(3, 10)
	kfS, _, err := NewInformationFromState(x0, P0, F, G, H, noise)
	if err != nil {
		t.Fatal(err)
	}
	i0 := mat64.NewVector(3, nil)
	I0 := mat64.NewSymDense(3, nil)
	kfZ, _, err := NewInformation(i0, I0, F, G, H, noise)

	kfZ.SetStateTransition(F)
	kfZ.SetInputControl(G)
	kfZ.SetMeasurementMatrix(H)
	kfZ.SetNoise(noise)

	for _, kf := range []Information{*kfS, *kfZ} {
		var est Estimate
		yacc := []float64{0.12758, 0.11748, 0.20925, 0.0984, 0.12824, -0.069948, -0.11166, 0.25519, 0.12713, -0.011207, 0.50973, 0.12334, -0.028878, 0.19208, 0.17605, -0.10383, 0.19707, -0.40455, 0.27355, 0.060617, 0.10369, 0.22131, -0.0038337, -0.60504, -0.10213, -0.021907, 0.030875, 0.17578, -0.45262, -0.086119, -0.12265, -0.056002, -0.11744, 0.01039, 0.028251, 0.053642, 0.17204, -0.052963, -0.16611, 0.078431, -0.20175, -0.23044, 0.38302, -0.33455, -0.35916, 0.28959, 0.097137, -0.29778, -0.23343, 0.21113, -0.22098, -0.057898, 0.17649, 0.058624, 0.045438, 0.11104, 0.37742, 0.0013074, 0.34331, 0.37244, 0.01434, -0.35709, 0.14435, -0.20445, -0.031335, -0.35165, -0.091494, -0.34382, 0.36144, -0.3835, 0.10339, -0.055055, -0.17677, -0.12108, -0.094458, -0.38408, 0.03215, 0.5759, 0.3297, -0.63341, 0.11228, 0.32364, -0.36897, 0.050504, 0.25338, -0.040326, 0.37904, 0.083807, -0.1023, 0.19609, 0.43701, -0.067234, 0.11835, 0.10064, 0.1024, 0.19084, 0.22646, -0.17419, 0.27345, 36.295}
		for k := 1; k < 100; k++ {
			yVec := mat64.NewVector(1, []float64{yacc[k]})
			est, err = kf.Update(yVec, mat64.NewVector(1, nil))
			if err != nil {
				t.Fatal(err)
			}

			// At k=99, there is an especially high yacc in order to test another line in the code.
			if !est.IsWithin2σ() && k != 99 {
				t.Logf("[WARN] 2σ bound breached: k=%d -> %s ", k, est)
			}
			if k == 99 {
				t.Logf("k=%d\n%s", k, est)
			}
		}

		if _, err = kf.Update(mat64.NewVector(1, nil), mat64.NewVector(2, nil)); err == nil {
			t.Fatal("using an invalid control vector does not fail")
		}

		if _, err = kf.Update(mat64.NewVector(2, nil), mat64.NewVector(1, nil)); err == nil {
			t.Fatal("using an invalid measurement vector does not fail")
		}
	}
}

func TestInformationMultiD(t *testing.T) {
	// DT system
	Δt := 0.01
	F := mat64.NewDense(4, 4, []float64{1, 0.01, 5e-5, 0, 0, 1, 0.01, 0, 0, 0, 1, 0, 0, 0, 0, 1.0005})
	G := mat64.NewDense(4, 1, []float64{(5e-7) / 3, 5e-5, 0.01, 0})
	H := mat64.NewDense(2, 4, []float64{1, 0, 0, 0, 0, 0, 1, 1})
	// Noise
	Q := mat64.NewSymDense(4, []float64{2.5e-15, 6.25e-13, (25e-11) / 3, 0, 6.25e-13, (5e-7) / 3, 2.5e-8, 0, (25e-11) / 3, 2.5e-8, 5e-6, 0, 0, 0, 0, 5.302e-4})
	R := mat64.NewSymDense(2, []float64{0.005 / Δt, 0, 0, 0.0005 / Δt})

	// Vanilla KF
	noise := NewAWGN(Q, R)
	x0 := mat64.NewVector(4, []float64{0, 0.35, 0, 0})
	P0 := ScaledIdentity(4, 10)
	kf, _, err := NewInformationFromState(x0, P0, F, G, H, noise)
	t.Logf("%s", kf)
	if err != nil {
		panic(err)
	}

	measurements := []*mat64.Vector{mat64.NewVector(2, []float64{-0.80832, -0.011207}), mat64.NewVector(2, []float64{0.39265, 0.060617})}

	for _, measurement := range measurements {
		_, err := kf.Update(measurement, mat64.NewVector(1, nil))
		if err != nil {
			t.Fatal(err)
		}
	}

}
