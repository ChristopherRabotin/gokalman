package main

import (
	"fmt"
	"os"

	"github.com/ChristopherRabotin/gokalman"
	"github.com/gonum/matrix/mat64"
)

func main() {
	Δt := 0.1
	F := mat64.NewDense(2, 2, []float64{1, Δt, 0, 1})
	G := mat64.NewDense(2, 1, []float64{0.5 * Δt * Δt, Δt})
	H := mat64.NewDense(1, 2, []float64{1, 0})
	R := mat64.NewSymDense(1, []float64{0.005 / Δt})
	//Q := mat64.NewSymDense(2, []float64{3e-4, 5e-3, 0, 0.1})
	Q := mat64.NewSymDense(2, []float64{5e-3, 0, 0, 1e-3})
	noise := gokalman.NewAWGN(Q, R)
	x0 := mat64.NewVector(2, []float64{0, 0})
	P0 := gokalman.Identity(2)
	kf, _, _ := gokalman.NewPurePredictorVanilla(x0, P0, F, G, H, noise)
	runs := gokalman.NewMonteCarloRuns(50, 120, 1, 1, kf)
	fmt.Printf("runs: %d\testimates: %d\n", len(runs.Runs), len(runs.Runs[0].Estimates))
	// Run the Chi square tests.
	NISmeans, NEESmeans, err := gokalman.NewChiSquare(kf, runs, 1, true, true)
	if err != nil {
		panic(err)
	}
	// Output the NIS and NEES to a CSV file.
	f, _ := os.Create("./chisquare.csv")
	f.WriteString("NIS,NEES\n")
	for k := 0; k < len(NISmeans); k++ {
		f.WriteString(fmt.Sprintf("%f,%f\n", NISmeans[k], NEESmeans[k]))
	}
	f.Close()
}
