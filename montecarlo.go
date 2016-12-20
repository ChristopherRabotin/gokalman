package gokalman

import (
	"fmt"
	"strings"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// MonteCarloRuns stores MC runs.
type MonteCarloRuns struct {
	runs, steps int
	Runs        []MonteCarloRun
}

// Mean returns the mean of all the samples for the given time step.
func (mc MonteCarloRuns) Mean(step int) (mean []float64) {
	// Take the first run in order to know the size.
	states := make(map[int][]float64)
	rows, _ := mc.Runs[0].Estimates[0].State().Dims()
	for i := 0; i < rows; i++ {
		states[i] = make([]float64, len(mc.Runs))
	}
	// Gather information
	for r, run := range mc.Runs {
		state := run.Estimates[step].State()
		for i := 0; i < rows; i++ {
			states[i][r] = state.At(i, 0)
		}
	}
	means := make([]float64, rows)
	for i := 0; i < rows; i++ {
		means[i] = stat.Mean(states[i], nil)
	}
	return means
}

// StdDev returns the standard deviation of all the samples for the given time step.
func (mc MonteCarloRuns) StdDev(step int) (mean []float64) {
	// Take the first run in order to know the size.
	states := make(map[int][]float64)
	rows, _ := mc.Runs[0].Estimates[0].State().Dims()
	for i := 0; i < rows; i++ {
		states[i] = make([]float64, len(mc.Runs))
	}
	// Gather information
	for r, run := range mc.Runs {
		state := run.Estimates[step].State()
		for i := 0; i < rows; i++ {
			states[i][r] = state.At(i, 0)
		}
	}
	devs := make([]float64, rows)
	for i := 0; i < rows; i++ {
		devs[i] = stat.StdDev(states[i], nil)
	}
	return devs
}

// AsCSV is used as a CSV serializer. Does not include the header.
func (mc MonteCarloRuns) AsCSV(headers []string) []string {
	rows, _ := mc.Runs[0].Estimates[0].State().Dims()
	rtn := make([]string, rows)

	for i := 0; i < rows; i++ {
		header := headers[i]
		lines := make([]string, mc.steps+1) // One line per step, plus header.
		for rNo := 0; rNo < mc.runs; rNo++ {
			lines[0] += fmt.Sprintf("%s-%d,", header, rNo)
		}
		lines[0] += header + "-mean," + header + "-stddev"

		for k := 0; k < mc.steps; k++ {
			for rNo, run := range mc.Runs {
				lines[k+1] += fmt.Sprintf("%f,", run.Estimates[k].State().At(i, 0))

				if rNo == mc.runs-1 {
					// Last run reached, let's add the mean and stddev for this step.
					mean := mc.Mean(k)
					stddev := mc.StdDev(k)
					lines[k+1] += fmt.Sprintf("%f,%f", mean[i], stddev[i])
				}
			}
		}
		rtn[i] = strings.Join(lines, "\n")
	}
	return rtn
}

// NewMonteCarloRuns run monte carlos on the provided filter.
func NewMonteCarloRuns(samples, steps, rowsH int, controls []*mat64.Vector, kf KalmanFilter) MonteCarloRuns {
	runs := make([]MonteCarloRun, samples)
	if len(controls) < 1 {
		panic("must provide at least one control vector for size")
	} else if len(controls) == 1 {
		ctrlSize, _ := controls[0].Dims()
		controls = make([]*mat64.Vector, steps)
		// Populate with zero controls
		for k := 0; k < steps; k++ {
			controls[k] = mat64.NewVector(ctrlSize, nil)
		}
	}
	for sample := 0; sample < samples; sample++ {
		MCRun := MonteCarloRun{Estimates: make([]Estimate, steps)}
		for k := 0; k < steps; k++ {
			est, _ := kf.Update(mat64.NewVector(rowsH, nil), controls[k])
			MCRun.Estimates[k] = est
		}
		runs[sample] = MCRun
	}
	return MonteCarloRuns{samples, steps, runs}
}

// MonteCarloRun stores the results of an MC run.
type MonteCarloRun struct {
	Estimates []Estimate
}
