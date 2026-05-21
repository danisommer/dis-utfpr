// Package main — Servidor HTTP compilado (Go) para reconstrucao de imagens
// por CGNR/CGNE. Mesmo contrato do servidor Python (porta 5002 por padrao).
//
// Endpoint:
//
//	POST /reconstruct
//	    Body JSON:
//	      {
//	        "g": [...],              // vetor de sinal
//	        "H": [[...], ...],       // opcional — matriz de modelo
//	        "H_path": "...",         // opcional — caminho local p/ matriz
//	        "algorithm": "cgnr"|"cgne",
//	        "model": 1 | 2,
//	        "apply_gain": true|false
//	      }
//	    Resposta JSON:
//	      {
//	        "algorithm": "...",
//	        "image_base64": "<PNG em base64>",
//	        "width": int, "height": int,
//	        "n_iter": int,
//	        "tempo_reconstrucao_s": float,
//	        "started_at": "ISO8601",
//	        "finished_at": "ISO8601",
//	        "server": "go"
//	      }
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

type modelConfig struct {
	S, N, W, H int
}

var modelConfigs = map[int]modelConfig{
	1: {S: 794, N: 64, W: 60, H: 60},
	2: {S: 436, N: 64, W: 30, H: 30},
}

type reconstructRequest struct {
	G         []float64   `json:"g"`
	H         [][]float64 `json:"H,omitempty"`
	HPath     string      `json:"H_path,omitempty"`
	Algorithm string      `json:"algorithm"`
	Model     int         `json:"model"`
	ApplyGain *bool       `json:"apply_gain,omitempty"`
}

type reconstructResponse struct {
	Algorithm           string  `json:"algorithm"`
	ImageBase64         string  `json:"image_base64"`
	Width               int     `json:"width"`
	Height              int     `json:"height"`
	NIter               int     `json:"n_iter"`
	TempoReconstrucaoS  float64 `json:"tempo_reconstrucao_s"`
	StartedAt           string  `json:"started_at"`
	FinishedAt          string  `json:"finished_at"`
	Server              string  `json:"server"`
}

func vectorToPNG(f *mat.VecDense, width, height int) ([]byte, error) {
	n := f.Len()
	if n != width*height {
		return nil, fmt.Errorf("dimensao incorreta: %d != %d*%d", n, width, height)
	}

	// normaliza para [0, 255]
	minV, maxV := math.Inf(1), math.Inf(-1)
	for i := 0; i < n; i++ {
		v := f.AtVec(i)
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	span := maxV - minV
	if span == 0 {
		span = 1
	}

	img := image.NewGray(image.Rect(0, 0, width, height))
	// reshape column-major (compatibilidade com NumPy 'F')
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := x*height + y
			v := (f.AtVec(idx) - minV) / span * 255.0
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			img.SetGray(x, y, color.Gray{Y: uint8(v)})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func denseFromRows(rows [][]float64) (*mat.Dense, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("matriz vazia")
	}
	cols := len(rows[0])
	flat := make([]float64, 0, len(rows)*cols)
	for i, r := range rows {
		if len(r) != cols {
			return nil, fmt.Errorf("linha %d tem %d colunas, esperado %d", i, len(r), cols)
		}
		flat = append(flat, r...)
	}
	return mat.NewDense(len(rows), cols, flat), nil
}

func defaultHPath(model int) string {
	if env := os.Getenv(fmt.Sprintf("H_MODEL_%d_PATH", model)); env != "" {
		return env
	}
	for _, cand := range []string{
		fmt.Sprintf("data/H-%d.csv", model),
		fmt.Sprintf("data/H_modelo_%d.csv", model),
	} {
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	return fmt.Sprintf("data/H-%d.csv", model)
}

func reconstructHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "metodo nao permitido", http.StatusMethodNotAllowed)
		return
	}

	var req reconstructRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("json invalido: %v", err), http.StatusBadRequest)
		return
	}

	cfg, ok := modelConfigs[req.Model]
	if !ok {
		http.Error(w, fmt.Sprintf("modelo invalido: %d", req.Model), http.StatusBadRequest)
		return
	}

	if len(req.G) == 0 {
		http.Error(w, "campo 'g' ausente", http.StatusBadRequest)
		return
	}
	g := mat.NewVecDense(len(req.G), append([]float64(nil), req.G...))

	var H *mat.Dense
	switch {
	case len(req.H) > 0:
		d, err := denseFromRows(req.H)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		H = d
	case req.HPath != "":
		d, err := LoadH(req.HPath)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		H = d
	default:
		d, err := LoadH(defaultHPath(req.Model))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		H = d
	}

	applyGain := true
	if req.ApplyGain != nil {
		applyGain = *req.ApplyGain
	}
	if applyGain {
		ApplySignalGain(g, cfg.S, cfg.N)
	}

	algo := strings.ToLower(strings.TrimSpace(req.Algorithm))
	started := time.Now().UTC()

	var (
		f     *mat.VecDense
		nIter int
		dur   time.Duration
	)
	switch algo {
	case "cgnr":
		f, nIter, dur = CGNR(H, g, 10, 1e-4)
	case "cgne":
		f, nIter, dur = CGNE(H, g, 10, 1e-4)
	default:
		http.Error(w, fmt.Sprintf("algoritmo invalido: %s", algo), http.StatusBadRequest)
		return
	}

	finished := time.Now().UTC()

	pngBytes, err := vectorToPNG(f, cfg.W, cfg.H)
	if err != nil {
		http.Error(w, fmt.Sprintf("falha ao gerar PNG: %v", err), http.StatusInternalServerError)
		return
	}

	resp := reconstructResponse{
		Algorithm:          strings.ToUpper(algo),
		ImageBase64:        base64.StdEncoding.EncodeToString(pngBytes),
		Width:              cfg.W,
		Height:             cfg.H,
		NIter:              nIter,
		TempoReconstrucaoS: dur.Seconds(),
		StartedAt:          started.Format(time.RFC3339Nano),
		FinishedAt:         finished.Format(time.RFC3339Nano),
		Server:             "go",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("falha encode: %v", err)
	}

	log.Printf("reconstruct ok algo=%s model=%d iter=%d tempo=%.4fs",
		algo, req.Model, nIter, dur.Seconds())
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"server": "go",
	})
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "5002"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/reconstruct", reconstructHandler)
	mux.HandleFunc("/health", healthHandler)

	addr := ":" + port
	log.Printf("Servidor compilado iniciando na porta %s", port)

	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 15 * time.Second,
	}
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("servidor falhou: %v", err)
	}
}
