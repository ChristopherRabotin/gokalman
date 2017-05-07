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

func TestHybridBasic(t *testing.T) {
	prevXHat := mat64.NewVector(6, nil)
	prevP := mat64.NewSymDense(6, nil)
	var covarDistance float64 = 50
	var covarVelocity float64 = 1
	for i := 0; i < 3; i++ {
		prevP.SetSym(i, i, covarDistance)
		prevP.SetSym(i+3, i+3, covarVelocity)
	}

	Q := mat64.NewSymDense(6, nil)
	R := mat64.NewSymDense(2, []float64{1e-3, 0, 0, 1e-6})
	noiseKF := NewNoiseless(Q, R)

	hkf, _, err := NewHybridKF(prevXHat, prevP, noiseKF, 2)
	if err != nil {
		t.Fatalf("%s", err)
	}
	// Check that calling Update before "Prepare" returns an error
	_, err = hkf.Update(mat64.NewVector(2, nil), mat64.NewVector(2, nil))
	if err == nil {
		t.Fatal("error should not have been nil when calling Update before Prepare")
	}

	// Check that calling Predict before "Prepare" returns an error
	_, err = hkf.Predict()
	if err == nil {
		t.Fatal("error should not have been nil when calling Predict before Prepare")
	}

	hkf.EnableEKF()
	if hkf.ekfMode == false || !hkf.EKFEnabled() {
		t.Fatal("the KF is still in CKF mode after EKF switch")
	}

	hkf.DisableEKF()
	if hkf.ekfMode == true || hkf.EKFEnabled() {
		t.Fatal("the KF is still in EKF mode after CKF switch")
	}
}

func TestCKFFull(t *testing.T) {
	hybridFullODExample(-15, 0, -15, false, false, false, t)
	hybridFullODExample(-15, 0, -15, true, false, false, t) // Smoothing
	t.Skip("Skipping broken SNC test (not high priority right now)")
	hybridFullODExample(-15, 0, -15, false, true, false, t) // SNC
	hybridFullODExample(-15, 0, -15, false, true, true, t)  // SNC RIC
}

func TestEKFFull(t *testing.T) {
	hybridFullODExample(15, 0, -15, false, false, false, t)
}
func hybridFullODExample(ekfTrigger int, ekfDisableTime, sncDisableTime float64, smoothing, sncEnabled, sncRIC bool, t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	startDT := time.Date(2017, 1, 1, 0, 0, 0, 0, time.UTC)
	endDT := startDT.Add(time.Duration(12) * time.Hour)
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
	export := smd.ExportConfig{Filename: "CKFFullOD", Cosmo: false, AsCSV: true, Timestamp: false}
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
	var ekfWG sync.WaitGroup
	timeStep := 1 * time.Second
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
	t.Logf("Generated %d measurements", len(measurements))
	truth := NewBatchGroundTruth(stateTruth, truthMeas)

	// Compute number of states which will be generated.
	numStates := int((measurementTimes[len(measurementTimes)-1].Sub(measurementTimes[0])).Seconds()/timeStep.Seconds()) + 2
	residuals := make([]*mat64.Vector, numStates)
	estHistory := make([]*HybridKFEstimate, numStates)
	stateHistory := make([]*mat64.Vector, numStates) // Stores the histories of the orbit estimate (to post compute the truth)

	// Get the first measurement as an initial orbit estimation.
	firstDT := measurementTimes[0]
	estOrbit := measurements[firstDT].State.Orbit
	// TODO: Add noise to initial orbit estimate.

	// Perturbations in the estimate
	estPerts := smd.Perturbations{Jn: 2}

	stateEstChan := make(chan (smd.State), 1)
	mEst := smd.NewPreciseMission(smd.NewEmptySC(scName+"Est", 0), &estOrbit, firstDT, firstDT.Add(-1), estPerts, timeStep, true, smd.ExportConfig{})
	mEst.RegisterStateChan(stateEstChan)

	// KF filter initialization stuff.

	// Initialize the KF noise
	σQExponent := 6.0
	σQx := math.Pow(10, -2*σQExponent)
	var σQy, σQz float64
	if !sncRIC {
		σQy = σQx
		σQz = σQx
	}
	noiseQ := mat64.NewSymDense(3, []float64{σQx, 0, 0, 0, σQy, 0, 0, 0, σQz})
	noiseR := mat64.NewSymDense(2, []float64{σρ, 0, 0, σρDot})
	noiseKF := NewNoiseless(noiseQ, noiseR)

	// Take care of measurements.
	estChan := make(chan (Estimate), 1)
	go processEst("hybridkf", estChan, 1e0, 1e-1, t)

	prevP := mat64.NewSymDense(6, nil)
	var covarDistance float64 = 50
	var covarVelocity float64 = 1
	for i := 0; i < 3; i++ {
		prevP.SetSym(i, i, covarDistance)
		prevP.SetSym(i+3, i+3, covarVelocity)
	}

	visibilityErrors := 0

	if smoothing {
		t.Logf("[INFO] Smoothing enabled")
	}

	if ekfTrigger < 0 {
		t.Logf("[WARNING] EKF disabled")
	} else {
		if smoothing {
			t.Logf("[ERROR] Enabling smooth has NO effect because EKF is enabled")
		}
		if ekfTrigger < 10 {
			t.Logf("[WARNING] EKF may be turned on too early")
		} else {
			t.Logf("[INFO] EKF will turn on after %d measurements\n", ekfTrigger)
		}
	}

	var prevStationName = ""
	var prevDT time.Time
	var ckfMeasNo = 0
	measNo := 0
	stateNo := 0
	kf, _, err := NewHybridKF(mat64.NewVector(6, nil), prevP, noiseKF, 2)
	kf.sncEnabled = sncEnabled

	if err != nil {
		t.Fatalf("%s", err)
	}

	// Go-routine to advance propagation.
	if ekfTrigger <= 0 {
		go mEst.PropagateUntil(measurementTimes[len(measurementTimes)-1].Add(timeStep), true)
	} else {
		// Go step by step because the orbit pointer needs to be updated.
		go func() {
			for i, measurementTime := range measurementTimes {
				ekfWG.Wait()
				ekfWG.Add(1)
				mEst.PropagateUntil(measurementTime, i == len(measurementTimes)-1)
			}
		}()
	}

	// Now let's do the filtering.
	for {
		state, more := <-stateEstChan
		if !more {
			break
		}
		stateNo++
		roundedDT := state.DT.Truncate(time.Second)
		measurement, exists := measurements[roundedDT]
		if !exists {
			if measNo == 0 {
				time.Sleep(time.Second)
				t.Fatalf("should start KF at first measurement: \n%s (got)\n%s (exp)", roundedDT, measurementTimes[0])
			}
			// There is no truth measurement here, let's only predict the KF covariance.
			kf.Prepare(state.Φ, nil)
			estI, perr := kf.Predict()
			if perr != nil {
				t.Fatalf("[ERR!] (#%04d)\n%s", measNo, perr)
			}
			est := estI.(*HybridKFEstimate)
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
		if measNo == 0 {
			prevDT = measurement.State.DT
		}

		// Let's perform a full update since there is a measurement.
		ΔtDuration := measurement.State.DT.Sub(prevDT)
		Δt := ΔtDuration.Seconds() // Everything is in seconds.
		// Informational messages.
		if !kf.EKFEnabled() && ckfMeasNo == ekfTrigger {
			// Switch KF to EKF mode
			kf.EnableEKF()
			t.Logf("[info] #%04d EKF now enabled\n", measNo)
		} else if kf.EKFEnabled() && ekfDisableTime > 0 && Δt > ekfDisableTime {
			// Switch KF back to CKF mode
			kf.DisableEKF()
			ckfMeasNo = 0
			t.Logf("[info] #%04d EKF now disabled (Δt=%s)\n", measNo, ΔtDuration)
		}

		if measurement.Station.Name != prevStationName {
			t.Logf("[info] #%04d %s in visibility of %s (T+%s)\n", measNo, scName, measurement.Station.Name, measurement.State.DT.Sub(firstDT))
			prevStationName = measurement.Station.Name
		}

		// Compute "real" measurement
		computedObservation := measurement.Station.PerformMeasurement(measurement.Timeθgst, state)
		if !computedObservation.Visible {
			t.Logf("[WARN] #%04d %s station %s should see the SC but does not\n", measNo, state.DT, measurement.Station.Name)
			visibilityErrors++
		}

		Htilde := computedObservation.HTilde()
		kf.Prepare(state.Φ, Htilde)
		if sncEnabled {
			if Δt < sncDisableTime {
				if sncRIC {
					// Build the RIC DCM
					rUnit := smd.Unit(state.Orbit.R())
					cUnit := smd.Unit(state.Orbit.H())
					iUnit := smd.Unit(smd.Cross(rUnit, cUnit))
					dcmVals := make([]float64, 9)
					for i := 0; i < 3; i++ {
						dcmVals[i] = rUnit[i]
						dcmVals[i+3] = cUnit[i]
						dcmVals[i+6] = iUnit[i]
					}
					// Update the Q matrix in the PQW
					dcm := mat64.NewDense(3, 3, dcmVals)
					var QECI, QECI0 mat64.Dense
					QECI0.Mul(noiseQ, dcm.T())
					QECI.Mul(dcm, &QECI0)
					QECISym, err := AsSymDense(&QECI)
					if err != nil {
						t.Logf("[ERR!] QECI is not symmertric!")
						panic(err)
					}
					kf.SetNoise(NewNoiseless(QECISym, noiseR))
				}
				// Only enable SNC for small time differences between measurements.
				Γtop := ScaledDenseIdentity(3, math.Pow(Δt, 2)/2)
				Γbot := ScaledDenseIdentity(3, Δt)
				Γ := mat64.NewDense(6, 3, nil)
				Γ.Stack(Γtop, Γbot)
				kf.PreparePNT(Γ)
			}
		}
		estI, err := kf.Update(measurement.StateVector(), computedObservation.StateVector())
		if err != nil {
			t.Fatalf("[ERR!] %s", err)
		}
		est := estI.(*HybridKFEstimate)
		if !est.IsWithin2σ() {
			t.Logf("[WARN] #%04d @ %s: not within 2-sigma", measNo, state.DT)
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
		residuals[stateNo] = residual

		if smoothing {
			// Save to history in order to perform smoothing.
			estHistory[stateNo-1] = est
			stateHistory[stateNo-1] = state.Vector()
		} else {
			// Stream to CSV file
			estChan <- truth.ErrorWithOffset(measNo, est, state.Vector())
		}
		prevDT = measurement.State.DT

		// If in EKF, update the reference trajectory.
		if kf.EKFEnabled() {
			// Update the state from the error.
			R, V := state.Orbit.RV()
			for i := 0; i < 3; i++ {
				R[i] += est.State().At(i, 0)
				V[i] += est.State().At(i+3, 0)
			}
			mEst.Orbit = smd.NewOrbitFromRV(R, V, smd.Earth)
		}
		ckfMeasNo++
		measNo++
		if ekfTrigger > 0 {
			ekfWG.Done()
		}
	} // end while true

	if smoothing {
		fmt.Println("[INFO] Smoothing started")
		// Perform the smoothing. First, play back all the estimates backward, and then replay the smoothed estimates forward to compute the difference.
		if err := kf.SmoothAll(estHistory); err != nil {
			t.Fatalf("smoothing failed: %s", err)
		}
		// Replay forward
		replayMeasNo := 0
		for estNo, est := range estHistory {
			thisNo := replayMeasNo
			if stateHistory[estNo] == nil {
				thisNo = -1
			}
			estChan <- truth.ErrorWithOffset(thisNo, est, stateHistory[estNo])
			if stateHistory[estNo] != nil {
				replayMeasNo++
			}
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
