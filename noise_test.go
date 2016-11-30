package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestImplementsNoise(t *testing.T) {
	implements := func(Noise) {}
	implements(new(Noiseless))
	implements(new(BatchNoise))
	implements(new(AWGN))
}

func TestBlankNoise(t *testing.T) {
	nl := Noiseless{2, 3}
	pR, _ := nl.Process(1).Dims()
	mR, _ := nl.Measurement(1).Dims()
	if pR != 2 {
		t.Fatal("expected only 2 rows of process noise")
	}
	if mR != 3 {
		t.Fatal("expected only 3 rows of measurement noise")
	}

	Q := nl.ProcessMatrix()
	qR, qC := Q.Dims()
	if qR != pR {
		t.Fatal("Q is of wrong size")
	}
	for i := 0; i < qR; i++ {
		for j := 0; j < qC; j++ {
			if Q.At(i, j) != 0 {
				t.Fatalf("Q(%d, %d) != 0", i, j)
			}
		}
	}

	R := nl.MeasurementMatrix()
	rR, rC := R.Dims()
	if rR != mR {
		t.Fatalf("R is of wrong size: %d != %d", rR, mR)
	}
	for i := 0; i < rR; i++ {
		for j := 0; j < rC; j++ {
			if R.At(i, j) != 0 {
				t.Fatalf("R(%d, %d) != 0", i, j)
			}
		}
	}
}

func TestBatchNoise(t *testing.T) {
	process := make([]*mat64.Vector, 4)
	measurements := make([]*mat64.Vector, 4)
	pR := 3
	mR := 2
	for i := 0; i < 4; i++ {
		process[i] = mat64.NewVector(pR, []float64{float64(i) + 1.0, float64(i) + 2.0, float64(i) + 3.0})
		measurements[i] = mat64.NewVector(mR, []float64{float64(i)*2.0 + 1.0, float64(i)*2.0 + 2.0})
	}
	batch := BatchNoise{process, measurements}
	for k := 0; k < 4; k++ {
		batch.Process(k)
		batch.Measurement(k)
	}

	Q := batch.ProcessMatrix()
	qR, qC := Q.Dims()
	if qR != pR {
		t.Fatal("Q is of wrong size")
	}
	for i := 0; i < qR; i++ {
		for j := 0; j < qC; j++ {
			if Q.At(i, j) != 0 {
				t.Fatalf("Q(%d, %d) != 0", i, j)
			}
		}
	}

	R := batch.MeasurementMatrix()
	rR, rC := R.Dims()
	if rR != mR {
		t.Fatalf("R is of wrong size: %d != %d", rR, mR)
	}
	for i := 0; i < rR; i++ {
		for j := 0; j < rC; j++ {
			if R.At(i, j) != 0 {
				t.Fatalf("R(%d, %d) != 0", i, j)
			}
		}
	}

	assertPanic(t, func() {
		batch.Process(4)
	})
	assertPanic(t, func() {
		batch.Measurement(4)
	})
}

func TestAWGN(t *testing.T) {
	assertPanic(t, func() {
		badQ := mat64.NewSymDense(2, []float64{1, 1, 1, 1})
		badR := mat64.NewSymDense(3, []float64{2, 3, 1, 3, 4, 6, 1, 6, 7})
		NewAWGN(badQ, badR)
	})

	assertPanic(t, func() {
		Q := mat64.NewSymDense(2, []float64{1, 0, 0, 1})
		badR := mat64.NewSymDense(3, []float64{2, 3, 1, 3, 4, 6, 1, 6, 7})
		NewAWGN(Q, badR)
	})

	Q := mat64.NewSymDense(2, []float64{1, 0, 0, 1})
	R := mat64.NewSymDense(2, []float64{20, 0.05, 0.05, 20})
	n := NewAWGN(Q, R)

	if !mat64.Equal(Q, n.ProcessMatrix()) {
		t.Fatal("Q and n.ProcessMatrix are not equal.")
	}

	if !mat64.Equal(R, n.MeasurementMatrix()) {
		t.Fatal("T and n.MeasurementMatrix are not equal.")
	}

	pk0 := n.Process(0)
	pk1 := n.Process(1)
	mk0 := n.Measurement(0)
	mk1 := n.Measurement(1)
	if pR, _ := pk0.Dims(); pR != 2 {
		t.Fatalf("process noise is a vector with %d rows (instead of 2)", pR)
	} else {
		vecEqual := true
		for i := 0; i < pR; i++ {
			if pk0.At(i, 0) != pk1.At(i, 0) {
				vecEqual = false
				break
			}
		}
		if vecEqual {
			t.Fatal("process noise at two different time steps is identical")
		}
	}
	if mR, _ := mk0.Dims(); mR != 2 {
		t.Fatalf("measurement noise is a vector with %d rows (instead of 3)", mR)
	} else {
		vecEqual := true
		for i := 0; i < mR; i++ {
			if mk0.At(i, 0) != mk1.At(i, 0) {
				vecEqual = false
				break
			}
		}
		if vecEqual {
			t.Fatal("measurement noise at two different time steps is identical")
		}
	}
}
