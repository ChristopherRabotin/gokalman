package gokalman

// KalmanFilter defines a general KF.
type KalmanFilter interface {
	Estimate()
	Stop()
}
