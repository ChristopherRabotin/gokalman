package gokalman

import "testing"

func TestImplementsKF(t *testing.T) {
	implements := func(KalmanFilter) {}
	implements(new(Vanilla))
}

func TestImplementsEst(t *testing.T) {
	implements := func(Estimate) {}
	implements(VanillaEstimate{})
}
