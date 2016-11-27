package gokalman

import "testing"

func assertPanic(t *testing.T, f func()) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("code did not panic")
		}
	}()
	f()
}

func TestIdentity(t *testing.T) {
	n := 3
	i33 := Identity(n)
	if r, c := i33.Dims(); r != n || r != c {
		t.Fatalf("i11 has dimensions (%dx%d)", r, c)
	}
	for i := 0; i < n; i++ {
		if i33.At(i, i) != 1 {
			t.Fatalf("i33(%d,%d) != 1", i, i)
		}
		for j := 0; j < n; j++ {
			if i != j && i33.At(i, j) != 0 {
				t.Fatalf("i33(%d,%d) != 0", i, j)
			}
		}
	}
}

func TestCheckDims(t *testing.T) {
	i22 := Identity(2)
	i33 := Identity(3)
	methods := []DimensionAgreement{rows2cols, cols2rows, cols2cols, rows2rows, rowsAndcols}
	for _, meth := range methods {
		if err := checkMatDims(i22, i22, "i22", "i22", meth); err != nil {
			t.Fatalf("method %+v fails: %s", meth, err)
		}
		if err := checkMatDims(i22, i33, "i22", "i33", meth); err == nil {
			t.Fatalf("method %+v does not error when using i22 and i33 ", meth)
		}
	}
}
