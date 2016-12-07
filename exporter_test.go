package gokalman

import (
	"os"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestImplementsExporter(t *testing.T) {
	implements := func(Exporter) {}
	implements(new(CSVExporter))
}

func TestCSVExportFail(t *testing.T) {
	_, err := NewCSVExporter([]string{"position", "velocity", "acceleration"}, "/noNoNoNo/", "temp.csv")
	if err == nil {
		t.Fatal("no issue when trying to create a file on root")
	}
}

func TestCSVExport(t *testing.T) {
	ce, err := NewCSVExporter([]string{"position", "velocity", "acceleration"}, ".", "temp.csv")
	if err != nil {
		t.Fatalf("could not create file %s", err)
	}
	initEst := VanillaEstimate{state: mat64.NewVector(3, []float64{0, 0.35, 0}), covar: ScaledIdentity(3, 10)}
	err = ce.Write(initEst)
	if err != nil {
		t.Fatalf("could not write estimate to file %s", err)
	}
	err = ce.Close()
	if err != nil {
		t.Fatalf("could not close file %s", err)
	}
	os.Remove(ce.hdlr.Name())
}
