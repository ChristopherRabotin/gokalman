package gokalman

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// DimensionAgreement defines how two matrices' dimensions should agree.
type DimensionAgreement uint8

const (
	dimErrMsg                    = "dimensions must agree: "
	rows2cols DimensionAgreement = iota + 1
	cols2rows
	cols2cols
	rows2rows
	rowsAndcols
)

// checkMatDims checks the matrix dimensions match provided a DimensionAgreement. Returns an error if not.
func checkMatDims(m1, m2 mat64.Matrix, name1, name2 string, method DimensionAgreement) error {
	r1, c1 := m1.Dims()
	r2, c2 := m2.Dims()
	switch method {
	case rows2cols:
		if r1 != c2 {
			return fmt.Errorf("%s%s(%dx...) %s(...x%d)", dimErrMsg, name1, r1, name2, c2)
		}
		break
	case cols2rows:
		if c1 != r2 {
			return fmt.Errorf("%s%s(...x%d) %s(%dx...)", dimErrMsg, name1, c1, name2, r2)
		}
		break
	case cols2cols:
		if c1 != c2 {
			return fmt.Errorf("%s%s(...x%d) %s(...x%d)", dimErrMsg, name1, c1, name2, c2)
		}
		break
	case rows2rows:
		if r1 != r2 {
			return fmt.Errorf("%s%s(%dx...) %s(%dx...)", dimErrMsg, name1, r1, name2, r2)
		}
		break
	case rowsAndcols:
		if c1 != c2 || r1 != r2 {
			return fmt.Errorf("%s%s(%dx%d) %s(%dx%d)", dimErrMsg, name1, r1, c1, name2, r2, c2)
		}
		break
	}
	return nil
}
