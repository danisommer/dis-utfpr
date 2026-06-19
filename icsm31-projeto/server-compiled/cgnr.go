// Package main — CGNR (Conjugate Gradient Normal Residual) em Go.
//
// Algoritmo:
//
//	f0 = 0
//	r0 = g - H * f0
//	z0 = H^T * r0
//	p0 = z0
//	loop:
//	    w_i   = H * p_i
//	    alpha = ||z_i||^2 / ||w_i||^2
//	    f     = f + alpha * p_i
//	    r     = r - alpha * w_i
//	    z_i+1 = H^T * r
//	    beta  = ||z_i+1||^2 / ||z_i||^2
//	    p_i+1 = z_i+1 + beta * p_i
package main

import (
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
)

// CGNR resolve H * f = g por gradiente conjugado no residual normal.
//
// Parametros:
//   - H: matriz de modelo (S x M)
//   - g: vetor de sinal (tamanho S)
//   - maxIter: numero maximo de iteracoes
//   - tol: tolerancia para |epsilon|
//
// Retorna:
//   - f: vetor reconstruido (tamanho M)
//   - nIter: numero de iteracoes executadas
//   - tempo: duracao da reconstrucao
func CGNR(H *mat.Dense, g *mat.VecDense, maxIter int, tol float64) (*mat.VecDense, int, time.Duration) {
	t0 := time.Now()

	_, m := H.Dims()

	f := mat.NewVecDense(m, nil)

	r := mat.VecDenseCopyOf(g)
	var Hf mat.VecDense
	Hf.MulVec(H, f)
	r.SubVec(r, &Hf)

	z := mat.NewVecDense(m, nil)
	z.MulVec(H.T(), r)

	p := mat.VecDenseCopyOf(z)

	zNormSq := mat.Dot(z, z)
	// epsilon = ||r_i+1||_2 - ||r_i||_2 (diferenca de normas, conforme enunciado)
	prevRNorm := math.Sqrt(mat.Dot(r, r))

	nIter := 0
	w := mat.NewVecDense(H.RawMatrix().Rows, nil)
	zNext := mat.NewVecDense(m, nil)

	for i := 0; i < maxIter; i++ {
		nIter = i + 1

		w.MulVec(H, p)
		wNormSq := mat.Dot(w, w)
		if wNormSq == 0 {
			break
		}

		alpha := zNormSq / wNormSq

		f.AddScaledVec(f, alpha, p)
		r.AddScaledVec(r, -alpha, w)

		newRNorm := math.Sqrt(mat.Dot(r, r))
		epsilon := newRNorm - prevRNorm
		if math.Abs(epsilon) < tol {
			break
		}
		prevRNorm = newRNorm

		zNext.MulVec(H.T(), r)
		zNextNormSq := mat.Dot(zNext, zNext)

		if zNormSq == 0 {
			break
		}
		beta := zNextNormSq / zNormSq

		// p = zNext + beta * p
		p.AddScaledVec(zNext, beta, p)

		z.CopyVec(zNext)
		zNormSq = zNextNormSq
	}

	return f, nIter, time.Since(t0)
}
