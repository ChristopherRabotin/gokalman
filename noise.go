package gokalman

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/distmv"
)

// Noise allows to handle the noise for a KF.
type Noise interface {
	Process(k int) *mat64.Vector        // Returns the process noise w at step k
	Measurement(k int) *mat64.Vector    // Returns the measurement noise w at step k
	ProcessMatrix() mat64.Symmetric     // Returns the process noise matrix Q
	MeasurementMatrix() mat64.Symmetric // Returns the measurement noise matrix R
	String() string                     // Stringer interface implementation
}

// Noiseless is noiseless and implements the Noise interface.
type Noiseless struct {
	Q, R                         mat64.Symmetric
	processSize, measurementSize int
}

// NewNoiseless creates new AWGN noise from the provided Q and R.
func NewNoiseless(Q, R mat64.Symmetric) *Noiseless {
	if Q == nil || R == nil {
		panic("Q and R must be specified")
	}
	rQ, _ := Q.Dims()
	rR, _ := R.Dims()
	return &Noiseless{Q, R, rQ, rR}
}

// Process returns a vector of the correct size.
func (n Noiseless) Process(k int) *mat64.Vector {
	return mat64.NewVector(n.processSize, nil)
}

// Measurement returns a vector of the correct size.
func (n Noiseless) Measurement(k int) *mat64.Vector {
	return mat64.NewVector(n.measurementSize, nil)
}

// ProcessMatrix implements the Noise interface.
func (n Noiseless) ProcessMatrix() mat64.Symmetric {
	return n.Q
}

// MeasurementMatrix implements the Noise interface.
func (n Noiseless) MeasurementMatrix() mat64.Symmetric {
	return n.R
}

// String implements the Stringer interface.
func (n Noiseless) String() string {
	return fmt.Sprintf("Noiseless{\nQ=%v\nR=%v}\n", mat64.Formatted(n.Q, mat64.Prefix("  ")), mat64.Formatted(n.R, mat64.Prefix("  ")))
}

// BatchNoise implements the Noise interface.
type BatchNoise struct {
	process     []*mat64.Vector // Array of process noise
	measurement []*mat64.Vector // Array of process noise
}

// Process implements the Noise interface.
func (n BatchNoise) Process(k int) *mat64.Vector {
	if k >= len(n.process) {
		panic(fmt.Errorf("no process noise defined at step k=%d", k))
	}
	return n.process[k]
}

// Measurement implements the Noise interface.
func (n BatchNoise) Measurement(k int) *mat64.Vector {
	if k >= len(n.measurement) {
		panic(fmt.Errorf("no measurement noise defined at step k=%d", k))
	}
	return n.measurement[k]
}

// ProcessMatrix implements the Noise interface.
func (n BatchNoise) ProcessMatrix() mat64.Symmetric {
	rows, _ := n.process[0].Dims()
	return mat64.NewSymDense(rows, nil)
}

// MeasurementMatrix implements the Noise interface.
func (n BatchNoise) MeasurementMatrix() mat64.Symmetric {
	rows, _ := n.measurement[0].Dims()
	return mat64.NewSymDense(rows, nil)
}

// String implements the Stringer interface.
func (n BatchNoise) String() string {
	return "BatchNoise"
}

// AWGN implements the Noise interface and generates an Additive white Gaussian noise.
type AWGN struct {
	Q, R        mat64.Symmetric
	process     *distmv.Normal
	measurement *distmv.Normal
}

// NewAWGN creates new AWGN noise from the provided Q and R.
func NewAWGN(Q, R mat64.Symmetric) *AWGN {
	//randomSeed := rand.New(rand.NewSource(time.Now().UnixNano()))
	seed := rand.New(rand.NewSource(1)) // TODO: Restore the random seed after debugging.
	sizeQ, _ := Q.Dims()
	process, ok := distmv.NewNormal(make([]float64, sizeQ), Q, seed)
	if !ok {
		panic("process noise invalid")
	}
	sizeR, _ := R.Dims()
	meas, ok := distmv.NewNormal(make([]float64, sizeR), R, seed)
	if !ok {
		panic("measurement noise invalid")
	}
	return &AWGN{Q, R, process, meas}
}

// ProcessMatrix implements the Noise interface.
func (n AWGN) ProcessMatrix() mat64.Symmetric {
	return n.Q
}

// MeasurementMatrix implements the Noise interface.
func (n AWGN) MeasurementMatrix() mat64.Symmetric {
	return n.R
}

// Process implements the Noise interface.
func (n AWGN) Process(k int) *mat64.Vector {
	r := n.process.Rand(nil)
	return mat64.NewVector(len(r), r)
}

// Measurement implements the Noise interface.
func (n AWGN) Measurement(k int) *mat64.Vector {
	r := n.measurement.Rand(nil)
	return mat64.NewVector(len(r), r)
}

// String implements the Stringer interface.
func (n AWGN) String() string {
	return fmt.Sprintf("AWGN{\nQ=%v\nR=%v}\n", mat64.Formatted(n.Q, mat64.Prefix("  ")), mat64.Formatted(n.R, mat64.Prefix("  ")))
}
