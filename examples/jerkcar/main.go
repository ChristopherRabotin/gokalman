package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"sync"

	"github.com/ChristopherRabotin/gokalman"
	"github.com/gonum/matrix/mat64"
)

func main() {
	// Load CSV measurement file.
	var control []*mat64.Vector
	if f, err := os.Open("uvec.csv"); err != nil {
		panic(err)
	} else {
		// Create a new reader.
		r := csv.NewReader(bufio.NewReader(f))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			value, err := strconv.ParseFloat(record[0], 64)
			if err != nil {
				fmt.Printf("could not parse float: %s", err)
				continue
			}
			control = append(control, mat64.NewVector(1, []float64{value}))
		}
	}

	// Both yacchist and yposthist are single line CSV files.
	singleRecord := func(name string) []float64 {
		var dest []float64
		if f, err := os.Open(name); err != nil {
			panic(err)
		} else {
			// Create a new reader.
			r := csv.NewReader(bufio.NewReader(f))
			record, err := r.Read()
			if err == io.EOF {
				return dest
			}

			for _, svalue := range record {
				value, err := strconv.ParseFloat(svalue, 64)
				if err != nil {
					fmt.Printf("could not parse float: %s", err)
					continue
				}
				if math.IsNaN(value) {
					value = 0
				}
				dest = append(dest, value)
			}
		}
		return dest
	}

	yacc := singleRecord("yacchist.csv")
	ypos := singleRecord("yposhist.csv")

	// Prepare the estimate channel.
	var wg sync.WaitGroup
	vanillaEstChan := make(chan (gokalman.Estimate), 1)
	informationEstChan := make(chan (gokalman.Estimate), 1)
	processEst := func(fn string, estChan chan (gokalman.Estimate)) {
		wg.Add(1)
		ce, _ := gokalman.NewCSVExporter([]string{"position", "velocity", "acceleration"}, ".", fn+".csv")
		for {
			est, more := <-estChan
			if !more {
				ce.Close()
				wg.Done()
				break
			}
			ce.Write(est)
		}
	}
	go processEst("vanilla", vanillaEstChan)
	go processEst("information", informationEstChan)

	// DT system
	Δt := 0.01
	F := mat64.NewDense(4, 4, []float64{1, 0.01, 5e-5, 0, 0, 1, 0.01, 0, 0, 0, 1, 0, 0, 0, 0, 1.0005})
	G := mat64.NewDense(4, 1, []float64{(5e-7) / 3, 5e-5, 0.01, 0})
	// Note that we will be using two difference H matrices, which we'll swap on the fly.
	H1 := mat64.NewDense(2, 4, []float64{1, 0, 0, 0, 0, 0, 1, 1})
	H2 := mat64.NewDense(2, 4, []float64{0, 0, 0, 0, 0, 0, 1, 1})
	// Noise
	Q := mat64.NewSymDense(4, []float64{2.5e-15, 6.25e-13, (25e-11) / 3, 0, 6.25e-13, (5e-7) / 3, 2.5e-8, 0, (25e-11) / 3, 2.5e-8, 5e-6, 0, 0, 0, 0, 5.302e-4})
	R := mat64.NewSymDense(2, []float64{0.005 / Δt, 0, 0, 0.0005 / Δt})

	// Vanilla KF
	noise := gokalman.NewAWGN(Q, R)
	x0 := mat64.NewVector(4, []float64{0, 0.35, 0, 0})
	Covar0 := gokalman.ScaledIdentity(4, 10)
	vanillaKF, err := gokalman.NewVanilla(x0, Covar0, F, G, H2, noise)
	if err != nil {
		panic(err)
	}

	// Information KF
	i0 := mat64.NewVector(4, nil)
	I0 := mat64.NewSymDense(4, nil)
	infoKF, err := gokalman.NewInformation(i0, I0, F, G, H2, noise)
	if err != nil {
		panic(err)
	}

	for k, yaccK := range yacc {
		measurement := mat64.NewVector(2, []float64{ypos[k], yaccK})
		if k%10 == 0 {
			// Switch to using H1
			vanillaKF.H = H1
		}
		vanillaEst, err := vanillaKF.Update(measurement, control[k])
		if err != nil {
			panic(err)
		}
		vanillaEstChan <- vanillaEst
		infoEst, err := infoKF.Update(measurement, control[k])
		if err != nil {
			panic(err)
		}
		informationEstChan <- infoEst
		if k%10 == 0 {
			// Switch back to using H2
			vanillaKF.H = H2
		}
	}
	close(vanillaEstChan)
	close(informationEstChan)

	wg.Wait()
}
