// Package main — Carregamento da matriz H a partir de arquivo CSV.
//
// O servidor Go le a matriz H em formato CSV (linhas = sensores,
// colunas = pixels). Em producao, este loader pode ser estendido para
// suportar formatos binarios (.bin) para acelerar a inicializacao.
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

var (
	hCache   = make(map[string]*mat.Dense)
	hCacheMu sync.RWMutex
)

// LoadH carrega a matriz H de um arquivo CSV (com cache em memoria).
func LoadH(path string) (*mat.Dense, error) {
	hCacheMu.RLock()
	if H, ok := hCache[path]; ok {
		hCacheMu.RUnlock()
		return H, nil
	}
	hCacheMu.RUnlock()

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("abrir %s: %w", path, err)
	}
	defer f.Close()

	var rows [][]float64
	scanner := bufio.NewScanner(f)
	// matrizes grandes — buffer generoso
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 64*1024*1024)

	cols := -1
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.FieldsFunc(line, func(r rune) bool {
			return r == ',' || r == ' ' || r == '\t' || r == ';'
		})
		row := make([]float64, len(fields))
		for i, s := range fields {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, fmt.Errorf("parse '%s' linha %d: %w", s, len(rows)+1, err)
			}
			row[i] = v
		}
		if cols == -1 {
			cols = len(row)
		} else if len(row) != cols {
			return nil, fmt.Errorf("colunas inconsistentes na linha %d: %d != %d",
				len(rows)+1, len(row), cols)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("leitura: %w", err)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("arquivo vazio: %s", path)
	}

	flat := make([]float64, 0, len(rows)*cols)
	for _, r := range rows {
		flat = append(flat, r...)
	}
	H := mat.NewDense(len(rows), cols, flat)

	hCacheMu.Lock()
	hCache[path] = H
	hCacheMu.Unlock()

	r, c := H.Dims()
	log.Printf("Matriz H carregada de %s, shape=(%d, %d)", path, r, c)
	return H, nil
}
