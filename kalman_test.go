package gokalman

import "testing"

func TestImplementsLDKF(t *testing.T) {
	implements := func(LDKF) {}
	implements(new(Vanilla))
	implements(new(Information))
	implements(new(SquareRoot))
}

func TestImplementsNLDKF(t *testing.T) {
	implements := func(NLDKF) {}
	implements(new(SRIF))
	implements(new(HybridKF))
}

func TestImplementsEst(t *testing.T) {
	implements := func(Estimate) {}
	implements(VanillaEstimate{})
	implements(InformationEstimate{})
	implements(SquareRootEstimate{})
	implements(HybridKFEstimate{})
	implements(SRIFEstimate{})
}
