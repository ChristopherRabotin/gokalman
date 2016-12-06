
[![Build Status](https://travis-ci.org/ChristopherRabotin/gokalman.svg?branch=master)](https://travis-ci.org/ChristopherRabotin/gokalman) [![Coverage Status](https://coveralls.io/repos/ChristopherRabotin/gokalman/badge.svg?branch=master&service=github)](https://coveralls.io/github/ChristopherRabotin/gokalman?branch=master)
[![goreport](https://goreportcard.com/badge/github.com/ChristopherRabotin/gokalman)](https://goreportcard.com/report/github.com/ChristopherRabotin/gokalman)


# gokalman
Go lang implementations of the Kalman Filter and its variantes, along with examples in statistical orbit determination.

# Usage
```go
estimateChan := make(chan(Estimate), 1)
go processEstimates(estimateChan)
kf := New[KalmanFilter](...) // e.g. NewVanilla(...)
for k, measurement := range measurements {
	newEstimate, err := kf.Update(measurement, controlVectors[k])
	if err != nil {
		processError(err)
		continue
	}
	estimateChan <- newEstimate
}
close(estimateChan)
// Should add a usage of sync.
```
