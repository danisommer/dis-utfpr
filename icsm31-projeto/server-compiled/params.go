// Package main — Parametros do enunciado (Algoritmos e definicoes).
//
//	c      = ||H^T H||_2              // fator de reducao
//	lambda = max(abs(H^T g)) * 0.10  // coeficiente de regularizacao
//
// Como ||H^T H||_2 = sigma_max(H)^2 (maior autovalor de H^T H), c e obtido
// por iteracao de potencia sobre H^T H, sem montar H^T H nem rodar SVD na
// matriz cheia. O resultado e cacheado por chave (caminho de H).
package main

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

var (
	cCache   = make(map[string]float64)
	cCacheMu sync.RWMutex
)

// ReductionFactor calcula c = ||H^T H||_2 (maior autovalor de H^T H) por
// iteracao de potencia. cacheKey vazio desativa o cache (ex.: H inline).
func ReductionFactor(H *mat.Dense, cacheKey string) float64 {
	if cacheKey != "" {
		cCacheMu.RLock()
		if v, ok := cCache[cacheKey]; ok {
			cCacheMu.RUnlock()
			return v
		}
		cCacheMu.RUnlock()
	}

	const maxIter = 200
	const tol = 1e-9

	_, m := H.Dims()
	v := mat.NewVecDense(m, nil)
	for i := 0; i < m; i++ {
		// vetor inicial deterministico e nao-nulo
		v.SetVec(i, 1.0)
	}
	nv := mat.Norm(v, 2)
	if nv == 0 {
		return 0
	}
	v.ScaleVec(1.0/nv, v)

	s, _ := H.Dims()
	hv := mat.NewVecDense(s, nil)
	w := mat.NewVecDense(m, nil)

	eigval := 0.0
	for iter := 0; iter < maxIter; iter++ {
		// w = (H^T H) v
		hv.MulVec(H, v)
		w.MulVec(H.T(), hv)
		nw := mat.Norm(w, 2)
		if nw == 0 {
			eigval = 0
			break
		}
		v.ScaleVec(1.0/nw, w)
		if math.Abs(nw-eigval) <= tol*nw {
			eigval = nw
			break
		}
		eigval = nw
	}

	if cacheKey != "" {
		cCacheMu.Lock()
		cCache[cacheKey] = eigval
		cCacheMu.Unlock()
	}
	return eigval
}

// RegularizationLambda calcula lambda = max(abs(H^T g)) * 0.10.
func RegularizationLambda(H *mat.Dense, g *mat.VecDense) float64 {
	_, m := H.Dims()
	htg := mat.NewVecDense(m, nil)
	htg.MulVec(H.T(), g)

	maxAbs := 0.0
	for i := 0; i < m; i++ {
		if a := math.Abs(htg.AtVec(i)); a > maxAbs {
			maxAbs = a
		}
	}
	return maxAbs * 0.10
}
