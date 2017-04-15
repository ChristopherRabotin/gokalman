package gokalman

import (
	"fmt"
	"math"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/ChristopherRabotin/smd"
	"github.com/gonum/matrix/mat64"
)

func TestSRIFR0(t *testing.T) {
	x0 := mat64.NewVector(3, []float64{0, 0.35, 0})
	P0 := ScaledIdentity(3, 10)
	Q := mat64.NewSymDense(6, nil)
	R := mat64.NewSymDense(2, []float64{math.Pow(5e-3, 2), 0, 0, math.Pow(5e-6, 2)})
	noise := NewNoiseless(Q, R)
	_, est0, err := NewSRIF(x0, P0, 3, true, noise)
	if err != nil {
		t.Fatal(err)
	}

	if !mat64.EqualApprox(est0.Covariance(), P0, 1e-12) {
		t.Fatalf("difference in P0 and computed covariance:\n%+v\n%+v", mat64.Formatted(P0), mat64.Formatted(est0.Covariance()))
	}
}

func TestSRIFUpdate(t *testing.T) {
	R := mat64.NewDense(2, 2, []float64{0.1, 0, 0, 0.1})
	H := mat64.NewDense(3, 2, []float64{1, -2, 2, -1, 1, 1})
	b := mat64.NewVector(2, []float64{0.2, 0.2})
	y := mat64.NewVector(3, []float64{-1.1, 1.2, 1.8})
	Rk, bk, ek, err := measurementSRIFUpdate(R, H, b, y)
	if err != nil {
		t.Fatalf("%s", err)
	}
	expEk := mat64.NewVector(3, []float64{-0.1319, 0.0871, -0.2810})
	if !mat64.EqualApprox(ek, expEk, 1e-4) {
		fmt.Printf("%+v", mat64.Formatted(ek))
		expEk.SubVec(ek, expEk)
		t.Fatalf("ek wrong by:\n%+v", mat64.Formatted(expEk))
	}
	expBk := mat64.NewVector(2, []float64{-1.2727, -2.0607})
	if !mat64.EqualApprox(bk, expBk, 1e-4) {
		expBk.SubVec(bk, expBk)
		t.Fatalf("bk wrong by:\n%+v", mat64.Formatted(expBk))
	}
	expRk := mat64.NewDense(2, 2, []float64{-2.4515, 1.2237, 0, -2.1243})
	if !mat64.EqualApprox(Rk, expRk, 1e-4) {
		expRk.Sub(expRk, Rk)
		t.Fatalf("Rk wrong by:\n%+v", mat64.Formatted(expRk))
	}
}

// The following is an example of StatOD using smd and gokalman
var wg sync.WaitGroup

func TestSRIFFullODExample(t *testing.T) {
	//_SRIFFullODExample(false, t)
	_SRIFFullODExample(true, t)
}

func _SRIFFullODExample(smoothing bool, t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	startDT := time.Date(2017, 1, 1, 0, 0, 0, 0, time.UTC)
	endDT := startDT.Add(time.Duration(24) * time.Hour)
	// Define the orbits
	leo := smd.NewOrbitFromOE(7000, 0.001, 30, 80, 40, 0, smd.Earth)

	// Define the stations
	σρ := math.Pow(1e-3, 2)    // m , but all measurements in km.
	σρDot := math.Pow(1e-3, 2) // m/s , but all measurements in km/s.
	st1 := smd.NewStation("st1", 0, 10, -35.398333, 148.981944, σρ, σρDot)
	st2 := smd.NewStation("st2", 0, 10, 40.427222, 355.749444, σρ, σρDot)
	st3 := smd.NewStation("st3", 0, 10, 35.247164, 243.205, σρ, σρDot)
	stations := []smd.Station{st1, st2, st3}

	measurements := make(map[time.Time]smd.Measurement)
	measurementTimes := []time.Time{}
	numMeasurements := 0 // Easier to count them here than to iterate the map to count.

	// Define the special export functions
	export := smd.ExportConfig{Filename: "SRIFFullOD", Cosmo: false, AsCSV: true, Timestamp: false}
	export.CSVAppendHdr = func() string {
		hdr := "secondsSinceEpoch,"
		for _, st := range stations {
			hdr += fmt.Sprintf("%sRange,%sRangeRate,%sNoisyRange,%sNoisyRangeRate,", st.Name, st.Name, st.Name, st.Name)
		}
		return hdr[:len(hdr)-1] // Remove trailing comma
	}
	export.CSVAppend = func(state smd.State) string {
		Δt := state.DT.Sub(startDT).Seconds()
		str := fmt.Sprintf("%f,", Δt)
		θgst := Δt * smd.EarthRotationRate
		roundedDT := state.DT.Truncate(time.Second)
		// Compute visibility for each station.
		for _, st := range stations {
			measurement := st.PerformMeasurement(θgst, state)
			if measurement.Visible {
				// Sanity check
				if _, exists := measurements[roundedDT]; exists {
					t.Fatalf("already have a measurement for %s", state.DT)
				}
				measurements[roundedDT] = measurement
				measurementTimes = append(measurementTimes, roundedDT)
				numMeasurements++
				str += measurement.CSV()
			} else {
				str += ",,,,"
			}
		}
		return str[:len(str)-1] // Remove trailing comma
	}

	// Generate the true orbit -- Mtrue
	timeStep := 10 * time.Second
	scName := "LEO"
	smd.NewPreciseMission(smd.NewEmptySC(scName, 0), leo, startDT, endDT, smd.Perturbations{Jn: 2}, timeStep, false, export).Propagate()

	// Let's mark those as the truth so we can plot that.
	stateTruth := make([]*mat64.Vector, len(measurements))
	truthMeas := make([]*mat64.Vector, len(measurements))
	for measNo, measTime := range measurementTimes {
		measurement := measurements[measTime]
		stateTruth[measNo] = measurement.State.Vector()
		truthMeas[measNo] = measurement.StateVector()
	}
	truth := NewBatchGroundTruth(stateTruth, truthMeas)

	// Compute number of states which will be generated.
	numStates := int((measurementTimes[len(measurementTimes)-1].Sub(measurementTimes[0])).Seconds()/timeStep.Seconds()) + 1
	residuals := make([]*mat64.Vector, numStates)
	estHistory := make([]*SRIFEstimate, numStates)
	stateHistory := make([]*mat64.Vector, numStates) // Stores the histories of the orbit estimate (to post compute the truth)

	// Get the first measurement as an initial orbit estimation.
	firstDT := measurementTimes[0]
	estOrbit := measurements[firstDT].State.Orbit
	startDT = firstDT //.Add(-timeStep)
	// TODO: Add noise to initial orbit estimate.

	// Perturbations in the estimate
	estPerts := smd.Perturbations{Jn: 2}

	stateEstChan := make(chan (smd.State), 1)
	mEst := smd.NewPreciseMission(smd.NewEmptySC(scName+"Est", 0), &estOrbit, startDT, startDT.Add(-1), estPerts, timeStep, true, smd.ExportConfig{})
	mEst.RegisterStateChan(stateEstChan)

	// Go-routine to advance propagation.
	go mEst.PropagateUntil(measurementTimes[len(measurementTimes)-1], true)

	// KF filter initialization stuff.

	// Initialize the KF noise
	σQExponent := 6.0
	σQx := math.Pow(10, -2*σQExponent)
	var σQy, σQz float64
	noiseQ := mat64.NewSymDense(3, []float64{σQx, 0, 0, 0, σQy, 0, 0, 0, σQz})
	noiseR := mat64.NewSymDense(2, []float64{σρ, 0, 0, σρDot})
	noiseKF := NewNoiseless(noiseQ, noiseR)

	// Take care of measurements.
	estChan := make(chan (Estimate), 1)
	go processEst("hybridkf", estChan, 1e-3, 1e-6, t)

	prevP := mat64.NewSymDense(6, nil)
	var covarDistance float64 = 50
	var covarVelocity float64 = 1
	for i := 0; i < 3; i++ {
		prevP.SetSym(i, i, covarDistance)
		prevP.SetSym(i+3, i+3, covarVelocity)
	}

	visibilityErrors := 0
	var prevStationName = ""
	measNo := 1
	stateNo := 0
	kf, _, err := NewSRIF(mat64.NewVector(6, nil), prevP, 2, false, noiseKF)
	if err != nil {
		panic(fmt.Errorf("%s", err))
	}
	// Now let's do the filtering.
	for {
		state, more := <-stateEstChan
		if !more {
			break
		}
		stateNo++
		// Just to test with a non triangular R and b vector, let's switch about half way.
		if stateNo == 200 {
			kf.nonTriR = true
		}
		roundedDT := state.DT.Truncate(time.Second)
		measurement, exists := measurements[roundedDT]
		if !exists {
			if measNo == 0 {
				time.Sleep(time.Second)
				t.Fatalf("should start KF at first measurement: \n%s (got)\n%s (exp)", roundedDT, measurementTimes[0])
			}
			// There is no truth measurement here, let's only predict the KF covariance.
			kf.Prepare(state.Φ, nil)
			est, perr := kf.Predict()
			if perr != nil {
				t.Fatalf("[ERR!] (#%04d)\n%s", measNo, perr)
			}
			if smoothing {
				// Save to history in order to perform smoothing.
				estHistory[stateNo-1] = est
				stateHistory[stateNo-1] = nil
			} else {
				// Stream to CSV file
				estChan <- truth.ErrorWithOffset(-1, est, nil)
			}
			continue
		}
		if roundedDT != measurementTimes[measNo] {
			t.Fatalf("[ERR!] %04d delta = %s\tstate=%s\tmeas=%s", measNo, state.DT.Sub(measurementTimes[measNo]), state.DT, measurementTimes[measNo])
		}

		// Let's perform a full update since there is a measurement.
		if measurement.Station.Name != prevStationName {
			t.Logf("[info] #%04d %s in visibility of %s (T+%s)\n", measNo, scName, measurement.Station.Name, measurement.State.DT.Sub(startDT))
			prevStationName = measurement.Station.Name
		}

		// Compute "real" measurement
		computedObservation := measurement.Station.PerformMeasurement(measurement.Timeθgst, state)
		if !computedObservation.Visible {
			t.Logf("[WARN] station %s should see the SC but does not\n", measurement.Station.Name)
			visibilityErrors++
		}

		Htilde := computedObservation.HTilde()
		kf.Prepare(state.Φ, Htilde)
		est, err := kf.Update(measurement.StateVector(), computedObservation.StateVector())
		if err != nil {
			t.Fatalf("[ERR!] %s", err)
		}

		if !est.IsWithin2σ() {
			t.Fatalf("%s", est)
		}
		if stateNo == 1 {
			t.Logf("\n%s", est)
		}

		prevP = est.Covariance().(*mat64.SymDense)
		// Compute residual
		residual := mat64.NewVector(2, nil)
		residual.MulVec(Htilde, est.State())
		residual.AddScaledVec(residual, -1, est.ObservationDev())
		residual.ScaleVec(-1, residual)
		residuals[stateNo-1] = residual

		if smoothing {
			// Save to history in order to perform smoothing.
			estHistory[stateNo-1] = est
			stateHistory[stateNo-1] = state.Vector()
		} else {
			// Stream to CSV file
			estChan <- truth.ErrorWithOffset(measNo, est, state.Vector())
		}
		measNo++
	} // end while true

	if smoothing {
		fmt.Println("[INFO] Smoothing started")
		// Perform the smoothing. First, play back all the estimates backward, and then replay the smoothed estimates forward to compute the difference.
		if err := kf.SmoothAll(estHistory); err != nil {
			panic(err)
		}
		// Replay forward
		for _, est := range estHistory {
			estChan <- est
		}
		fmt.Println("[INFO] Smoothing completed")
	}

	close(estChan)
	wg.Wait()

	severity := "INFO"
	if visibilityErrors > 0 {
		severity = "WARNING"
	}
	t.Logf("[%s] %d visibility errors\n", severity, visibilityErrors)
	// Write the residuals to a CSV file
	f, ferr := os.Create("./hkf-residuals.csv")
	if ferr != nil {
		panic(ferr)
	}
	defer f.Close()
	f.WriteString("rho,rhoDot\n")
	for _, residual := range residuals {
		csv := "0,0\n"
		if residual != nil {
			csv = fmt.Sprintf("%f,%f\n", residual.At(0, 0), residual.At(1, 0))
		}
		if _, err := f.WriteString(csv); err != nil {
			panic(err)
		}
	}
}

func processEst(fn string, estChan chan (Estimate), rmsPos, rmsVel float64, t *testing.T) {
	wg.Add(1)
	// We also compute the RMS here.
	numMeasurements := 0
	rmsPosition := 0.0
	rmsVelocity := 0.0
	ce, _ := NewCustomCSVExporter([]string{"x", "y", "z", "xDot", "yDot", "zDot"}, ".", fn+".csv", 3)
	for {
		est, more := <-estChan
		if !more {
			ce.Close()
			wg.Done()
			break
		}
		numMeasurements++
		for i := 0; i < 3; i++ {
			rmsPosition += math.Pow(est.State().At(i, 0), 2)
			rmsVelocity += math.Pow(est.State().At(i+3, 0), 2)
		}
		ce.Write(est)
	}
	// Compute RMS.
	rmsPosition /= float64(numMeasurements)
	rmsVelocity /= float64(numMeasurements)
	rmsPosition = math.Sqrt(rmsPosition)
	rmsVelocity = math.Sqrt(rmsVelocity)
	t.Logf("RMS: Position = %f\tVelocity = %f\n", rmsPosition, rmsVelocity)
	// We don't have any unmodeled dynamics, so the RMS should  be tiny.
	if rmsPosition > rmsPos || rmsVelocity > rmsVel {
		t.Fatal("RMS values too big")
	}
}
