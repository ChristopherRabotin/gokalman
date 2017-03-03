package gokalman

import "testing"

func TestImplementsKF(t *testing.T) {
	implements := func(KalmanFilter) {}
	implements(new(Vanilla))
	implements(new(Information))
	implements(new(SquareRoot))
}

func TestImplementsEst(t *testing.T) {
	implements := func(Estimate) {}
	implements(VanillaEstimate{})
	implements(InformationEstimate{})
	implements(SquareRootEstimate{})
	implements(HybridKFEstimate{})
}
