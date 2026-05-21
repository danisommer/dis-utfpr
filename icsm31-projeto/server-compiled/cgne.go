// Package main — CGNE (Conjugate Gradient Normal Error) em Go.
//
// Algoritmo:
//
//	f0 = 0
//	r0 = g - H * f0
//	p0 = H^T * r0
//	loop:
//	    alpha = (r^T r) / (p^T p)
//	    f     = f + alpha * p
//	    r     = r - alpha * H * p
//	    beta  = (r_new^T r_new) / (r^T r)
//	    p     = H^T * r_new + beta * p
package main

import (
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
)

// CGNE resolve H * f = g por gradiente conjugado no erro normal.
func CGNE(H *mat.Dense, g *mat.VecDense, maxIter int, tol float64) (*mat.VecDense, int, time.Duration) {
	t0 := time.Now()

	s, m := H.Dims()

	f := mat.NewVecDense(m, nil)

	r := mat.VecDenseCopyOf(g)
	var Hf mat.VecDense
	Hf.MulVec(H, f)
	r.SubVec(r, &Hf)

	p := mat.NewVecDense(m, nil)
	p.MulVec(H.T(), r)

	rNormSq := mat.Dot(r, r)
	prevRNorm := math.Sqrt(rNormSq)

	Hp := mat.NewVecDense(s, nil)
	HtR := mat.NewVecDense(m, nil)

	nIter := 0
	for i := 0; i < maxIter; i++ {
		nIter = i + 1

		pNormSq := mat.Dot(p, p)
		if pNormSq == 0 {
			break
		}
		alpha := rNormSq / pNormSq

		f.AddScaledVec(f, alpha, p)

		Hp.MulVec(H, p)
		r.AddScaledVec(r, -alpha, Hp)

		newRNormSq := mat.Dot(r, r)
		newRNorm := math.Sqrt(newRNormSq)

		epsilon := newRNorm - prevRNorm
		if math.Abs(epsilon) < tol {
			break
		}
		prevRNorm = newRNorm

		if rNormSq == 0 {
			break
		}
		beta := newRNormSq / rNormSq

		HtR.MulVec(H.T(), r)
		// p = H^T r + beta * p
		p.AddScaledVec(HtR, beta, p)

		rNormSq = newRNormSq
	}

	return f, nIter, time.Since(t0)
}
