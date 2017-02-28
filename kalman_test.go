package gokalman

import "testing"

func TestImplementsKF(t *testing.T) {
	implements := func(KalmanFilter) {}
	implements(new(Vanilla))
	implements(new(Information))
	implements(new(SquareRoot))
}

func TestImplementsHybridKF(t *testing.T) {
	implements := func(HybridKalmanFilter) {}
	implements(new(HybridCKF))
}

func TestImplementsEst(t *testing.T) {
	implements := func(Estimate) {}
	implements(VanillaEstimate{})
	implements(InformationEstimate{})
	implements(SquareRootEstimate{})
}
