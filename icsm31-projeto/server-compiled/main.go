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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
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
	Algorithm          string  `json:"algorithm"`
	ImageBase64        string  `json:"image_base64"`
	Width              int     `json:"width"`
	Height             int     `json:"height"`
	NIter              int     `json:"n_iter"`
	TempoReconstrucaoS float64 `json:"tempo_reconstrucao_s"`
	ReductionFactor    float64 `json:"reduction_factor"`
	LambdaReg          float64 `json:"lambda_reg"`
	StartedAt          string  `json:"started_at"`
	FinishedAt         string  `json:"finished_at"`
	Server             string  `json:"server"`
}

// textPair representa um chunk tEXt (keyword -> texto) a ser embutido no PNG.
type textPair struct {
	key string
	val string
}

// addPNGText insere chunks tEXt antes do primeiro chunk IDAT de um PNG ja
// codificado. O encoder padrao do Go (image/png) nao escreve tEXt, entao
// fazemos a insercao manual respeitando o formato: len(4) tipo(4) dados crc(4).
// A insercao antes do IDAT (e nao do IEND) garante que leitores que so
// processam chunks ate a imagem (ex.: PIL em modo lazy) tambem enxerguem os
// metadados — mesmo comportamento do servidor Python.
func addPNGText(pngBytes []byte, texts []textPair) []byte {
	const sigLen = 8
	if len(pngBytes) < sigLen+12 || len(texts) == 0 {
		return pngBytes
	}

	// localiza o inicio do primeiro chunk IDAT.
	idatPos := bytes.Index(pngBytes, []byte("IDAT"))
	if idatPos < 4 {
		return pngBytes
	}
	insertAt := idatPos - 4 // inicio do campo length do chunk IDAT

	var chunks bytes.Buffer
	for _, t := range texts {
		var data bytes.Buffer
		data.WriteString(t.key)
		data.WriteByte(0) // separador nulo entre keyword e texto
		data.WriteString(t.val)
		raw := data.Bytes()

		var lenBuf [4]byte
		binary.BigEndian.PutUint32(lenBuf[:], uint32(len(raw)))
		chunks.Write(lenBuf[:])

		typeAndData := append([]byte("tEXt"), raw...)
		chunks.Write(typeAndData)

		var crcBuf [4]byte
		binary.BigEndian.PutUint32(crcBuf[:], crc32.ChecksumIEEE(typeAndData))
		chunks.Write(crcBuf[:])
	}

	out := make([]byte, 0, len(pngBytes)+chunks.Len())
	out = append(out, pngBytes[:insertAt]...)
	out = append(out, chunks.Bytes()...)
	out = append(out, pngBytes[insertAt:]...)
	return out
}

func vectorToPNG(f *mat.VecDense, width, height int, texts []textPair) ([]byte, error) {
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
	return addPNGText(buf.Bytes(), texts), nil
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
	var hKey string
	switch {
	case len(req.H) > 0:
		d, err := denseFromRows(req.H)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		H = d
	case req.HPath != "":
		hKey = req.HPath
		d, err := LoadH(hKey)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		H = d
	default:
		hKey = defaultHPath(req.Model)
		d, err := LoadH(hKey)
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

	// Parametros do enunciado: c = ||H^T H||_2 (cacheado por H) e
	// lambda = max(abs(H^T g)) * 0.10 (com o sinal ja com ganho).
	cReduction := ReductionFactor(H, hKey)
	lambdaReg := RegularizationLambda(H, g)

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

	metadata := []textPair{
		{"algorithm", strings.ToUpper(algo)},
		{"started_at", started.Format(time.RFC3339Nano)},
		{"finished_at", finished.Format(time.RFC3339Nano)},
		{"size", fmt.Sprintf("%dx%d", cfg.W, cfg.H)},
		{"iterations", fmt.Sprintf("%d", nIter)},
		{"reduction_factor", fmt.Sprintf("%.6g", cReduction)},
		{"lambda_reg", fmt.Sprintf("%.6g", lambdaReg)},
		{"server", "go"},
	}

	pngBytes, err := vectorToPNG(f, cfg.W, cfg.H, metadata)
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
		ReductionFactor:    cReduction,
		LambdaReg:          lambdaReg,
		StartedAt:          started.Format(time.RFC3339Nano),
		FinishedAt:         finished.Format(time.RFC3339Nano),
		Server:             "go",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("falha encode: %v", err)
	}

	log.Printf("reconstruct ok algo=%s model=%d iter=%d tempo=%.4fs c=%.4g lambda=%.4g",
		algo, req.Model, nIter, dur.Seconds(), cReduction, lambdaReg)
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
