package gokalman

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// Exporter defines an export interface.
type Exporter interface {
	Write(Estimate) error
	Close() error
}

// CSVExporter returns a new CSV exporter.
type CSVExporter struct {
	covarBound float64
	delimiter  string
	hdlr       *os.File
}

// Close closes the file.
func (e CSVExporter) Close() (err error) {
	err = e.WriteRawLn(fmt.Sprintf("# Closing date (UTC): %s\n", time.Now().UTC()))
	if err != nil {
		return
	}
	return e.hdlr.Close()
}

// Write writes the estimate to the CSV file.
func (e CSVExporter) Write(est Estimate) error {
	r, _ := est.State().Dims()
	vals := make([]string, r*3)
	for i := 0; i < r*3; i += 3 {
		vals[i] = fmt.Sprintf("%f", est.State().At(i/3, 0))
		covar := e.covarBound * math.Sqrt(est.Covariance().At(i/3, i/3))
		vals[i+1] = fmt.Sprintf("%f", covar)
		vals[i+2] = fmt.Sprintf("%f", -1*covar)
	}
	_, err := e.hdlr.WriteString(strings.Join(vals, e.delimiter) + "\n")
	return err
}

// WriteRaw writes a raw line to the CSV file.
func (e CSVExporter) WriteRaw(s string) error {
	_, err := e.hdlr.WriteString(s)
	return err
}

// WriteRawLn writes a raw line to the CSV file.
func (e CSVExporter) WriteRawLn(s string) error {
	_, err := e.hdlr.WriteString(s + "\n")
	return err
}

// NewCustomCSVExporter initializes a new CSV export.
// NOTE: Prefix header with `_` to state that it does not relate to the estimation (and therefore won't have the added covariance bounds).
func NewCustomCSVExporter(headers []string, filepath, filename string, covarBound float64) (e *CSVExporter, err error) {
	f, err := os.Create(fmt.Sprintf("%s/%s", filepath, filename))
	if err != nil {
		return
	}
	delimiter := ","
	// Header
	hdr := make([]string, len(headers)*3)
	bhdr := fmt.Sprintf("%.0fs", covarBound)
	for i := 0; i < len(headers)*3; i += 3 {
		header := headers[i/3]
		if header[0:1] == "_" {
			// Non covar header
			hdr[i] = header[1:len(header)]
			continue
		}
		hdr[i] = headers[i/3]
		hdr[i+1] = hdr[i] + "+" + bhdr
		hdr[i+2] = hdr[i] + "-" + bhdr
	}
	// Remove extra headers
	trimmedHdrs := make([]string, 0)
	for _, thdr := range hdr {
		if len(thdr) > 0 {
			trimmedHdrs = append(trimmedHdrs, thdr)
		}
	}
	f.WriteString(fmt.Sprintf("# Creation date (UTC): %s\n%s\n", time.Now(), strings.Join(trimmedHdrs, delimiter)))
	e = &CSVExporter{covarBound, delimiter, f}
	return
}

// NewCSVExporter initializes a new CSV export.
func NewCSVExporter(headers []string, filepath, filename string) (e *CSVExporter, err error) {
	return NewCustomCSVExporter(headers, filepath, filename, 2)
}
