package gokalman

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestSRIFR0(t *testing.T) {
	x0 := mat64.NewVector(3, []float64{0, 0.35, 0})
	P0 := ScaledIdentity(3, 10)
	_, est0, err := NewSquareRootInformation(x0, P0, 3)
	if err != nil {
		t.Fatal(err)
	}

	if !mat64.Equal(est0.Covariance(), P0) {
		t.Fatalf("difference in P0 and computed covariance:\n%+v\n%+v", mat64.Formatted(P0), mat64.Formatted(est0.Covariance()))
	}

}
