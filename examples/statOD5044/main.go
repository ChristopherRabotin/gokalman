package main

import (
	"fmt"
	"sync"

	"github.com/ChristopherRabotin/gokalman"
	"github.com/gonum/matrix/mat64"
)

func main() {
	// Prepare the estimate channels.
	var wg sync.WaitGroup
	vanillaEstChan := make(chan (gokalman.Estimate), 1)
	informationEstChan := make(chan (gokalman.Estimate), 1)
	sqrtEstChan := make(chan (gokalman.Estimate), 1)

	processEst := func(fn string, estChan chan (gokalman.Estimate)) {
		wg.Add(1)
		ce, _ := gokalman.NewCSVExporter([]string{"velocity", "acceleration", "angular rate", "angular acceleration"}, ".", fn+".csv")
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
	go processEst("sqrt", sqrtEstChan)

	// DT system
	Δt := 0.1
	F := mat64.NewDense(4, 4, []float64{1, 0.1, 0, 7.726e-2, 4.015e-7, 1, 0, 1.545, -2.319e-16, -1.732e-9, 1, 0.1, -6.956e-15, -3.465e-8, 0, 1})
	G := mat64.NewDense(4, 2, []float64{5e-3, 3.85e-7, 0.1, 1.157e-5, -5.775e-11, 7.487e-7, 1.732e-9, 1.498e-5})
	H := mat64.NewDense(2, 4, []float64{1, 0, 0, 0, 0, 0, 1, 0})
	// Noise
	Q := mat64.NewSymDense(4, []float64{6.669e-16, 1.001e-14, 3.823e-19, 5.150e-18, 1.001e-14, 2.002e-13, 1.030e-17, 1.545e-16, 3.862e-19, 1.030e-17, 6.667e-19, 1.000e-17, 5.150e-18, 1.545e-16, 1.000e-17, 2.000e-16})
	R := mat64.NewSymDense(2, []float64{2e-6, 0, 0, 2e9})
	R.ScaleSym(1/Δt, R)

	// Control matrix
	T := mat64.NewDense(2, 4, []float64{0.930124736616832, 1.395260337125255, -0.000008568056356, 15.440297905873823, 0.000001749639349, 0.000000859493456, 0.001999922457941, 5.177881640687808})
	// As per problem statement, we'll be running the KF with no control vector per say, but using an updated F matrix.
	var Fcl mat64.Dense
	Fcl.Mul(G, T)
	Fcl.Sub(F, &Fcl)

	Gcl := mat64.NewDense(4, 2, nil)

	// Initial conditions
	x0 := mat64.NewVector(4, []float64{2, 0.5, 0, 0})
	P0 := mat64.NewSymDense(4, []float64{5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.00001})

	// Truth generation, via a vanilla KF with AWGN.
	truthNoise := gokalman.NewAWGN(Q, R)
	truthKF, err := gokalman.NewVanilla(x0, P0, &Fcl, Gcl, H, truthNoise)
	if err != nil {
		panic(err)
	}

	scPeriod := 5.431e3                 // Spacecraft period.
	samples := int((scPeriod / 4) * Δt) // Propagation time in samples.
	stateTruth := make([]*mat64.Vector, samples)
	measurements := make([]*mat64.Vector, samples)
	prevState := mat64.NewVector(4, nil)
	prevState.CopyVec(x0)

	for k := 0; k < samples; k++ {
		est, kferr := truthKF.Update(mat64.NewVector(2, nil), mat64.NewVector(4, nil))
		if kferr != nil {
			panic(fmt.Errorf("k=%d %s", k, kferr))
		}
		prevState = est.State()
		stateTruth[k] = prevState
		measurements[k] = est.Measurement()
	}

	// Vanilla KF
	noiseKF := gokalman.NewNoiseless(Q, R)
	vanillaKF, err := gokalman.NewVanilla(x0, P0, &Fcl, Gcl, H, noiseKF)
	if err != nil {
		panic(err)
	}

	// Information KF.
	i0 := mat64.NewVector(4, nil)
	I0 := mat64.NewSymDense(4, nil)
	infoKF, err := gokalman.NewInformation(i0, I0, &Fcl, Gcl, H, noiseKF)
	if err != nil {
		panic(err)
	}

	// SquareRoot KF
	sqrtKF, err := gokalman.NewSquareRoot(x0, P0, &Fcl, Gcl, H, noiseKF)
	if err != nil {
		panic(err)
	}

	filters := []gokalman.KalmanFilter{vanillaKF, infoKF, sqrtKF}
	chans := [](chan gokalman.Estimate){vanillaEstChan, informationEstChan, sqrtEstChan}

	// Generate for a quarter of orbit.
	for k := 0; k < samples; k++ {
		for i, kf := range filters {
			kfChan := chans[i]
			est, err := kf.Update(measurements[k], mat64.NewVector(4, nil))
			kfChan <- est
			if err != nil {
				panic(fmt.Errorf("k=%d %s", k, err))
			}

		}
	}

	// Close all channels
	for _, kfChan := range chans {
		close(kfChan)
	}

	wg.Wait()
}
