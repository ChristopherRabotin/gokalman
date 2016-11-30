package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/distmv"
)

// Noise allows to handle the noise for a KF.
type Noise interface {
	Process(k int) *mat64.Vector     // Returns the process noise w at step k
	Measurement(k int) *mat64.Vector // Returns the measurement noise w at step k
}

// Noiseless is noiseless and implements the Noise interface.
type Noiseless struct {
	processSize     int
	measurementSize int
}

// Process returns a vector of the correct size.
func (n Noiseless) Process(k int) *mat64.Vector {
	return mat64.NewVector(n.processSize, nil)
}

// Measurement returns a vector of the correct size.
func (n Noiseless) Measurement(k int) *mat64.Vector {
	return mat64.NewVector(n.measurementSize, nil)
}

// BatchNoise implements the Noise interface.
type BatchNoise struct {
	process     []*mat64.Vector // Array of process noise
	measurement []*mat64.Vector // Array of process noise
}

// Process implements the Noise interface.
func (n BatchNoise) Process(k int) *mat64.Vector {
	if k > len(n.process) {
		panic(fmt.Errorf("no process noise defined at step k=%d", k))
	}
	return n.process[k]
}

// Measurement implements the Noise interface.
func (n BatchNoise) Measurement(k int) *mat64.Vector {
	if k > len(n.measurement) {
		panic(fmt.Errorf("no measurement noise defined at step k=%d", k))
	}
	return n.measurement[k]
}

// AWGN implements the Noise interface and generates an Additive white Gaussian noise.
type AWGN struct {
	process     *distmv.Normal
	measurement *distmv.Normal
}

// NewAWGN creates new AWGN noise from the provided Q and R.
func NewAWGN(Q, R mat64.Symmetric) *AWGN {
	sizeQ, _ := Q.Dims()
	process, _ := distmv.NewNormal(make([]float64, sizeQ), Q, nil)
	sizeR, _ := Q.Dims()
	meas, _ := distmv.NewNormal(make([]float64, sizeR), R, nil)
	return &AWGN{process, meas}
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
