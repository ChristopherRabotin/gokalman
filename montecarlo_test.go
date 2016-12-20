package gokalman

import (
	"strings"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestMCRuns(t *testing.T) {
	// Create a Vanilla KF.
	F, _, Δt := Midterm2Matrices()
	Q := mat64.NewSymDense(3, []float64{2.5e-15, 6.25e-13, (25e-11) / 3, 6.25e-13, (5e-7) / 3, 2.5e-8, (25e-11) / 3, 2.5e-8, 5e-6})
	R := mat64.NewSymDense(1, []float64{0.005 / Δt})
	H := mat64.NewDense(1, 3, []float64{1, 0, 0})
	G := mat64.NewDense(1, 1, nil)
	noise := NewAWGN(Q, R)
	x0 := mat64.NewVector(3, []float64{0, 0.35, 0})
	P0 := ScaledIdentity(3, 10)
	kf, _, _ := NewPurePredictorVanilla(x0, P0, F, G, H, noise)
	steps := 10
	runs := NewMonteCarloRuns(5, steps, 1, 1, kf)
	if len(runs.Runs) != 5 {
		t.Fatal("requesting 5 runs did not generate five")
	}
	for r, run := range runs.Runs {
		if len(run.Estimates) != 10 {
			t.Fatalf("sample #%d does not have 10 steps", r)
		}
	}
	files := runs.AsCSV([]string{"x", "y", "z"})
	if len(files) != 3 {
		t.Fatal("less than 3 files returned from a three component state")
	}

	if len(strings.Split(files[0], "\n")) != 11 {
		t.Fatalf("unexpected number of lines in the file: %d", len(files[0]))
	}

	// Test chisquare:
	NISmeans, NEESmeans, err := NewChiSquare(kf, runs, 1, true, true)
	if err != nil {
		panic(err)
	}
	if len(NISmeans) != len(NEESmeans) || len(NISmeans) != steps {
		t.Fatal("invalid number of steps returned from ChiSquare tests")
	}

	if _, _, err := NewChiSquare(kf, runs, 1, false, false); err == nil {
		t.Fatal("attempting to run Chisquare with neither NIS nor NEES fails")
	}
}
