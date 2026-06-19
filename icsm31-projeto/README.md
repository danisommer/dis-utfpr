# ICSM31 — Reconstrução de Imagens por Gradiente Conjugado

Projeto da disciplina de **Desenvolvimento Integrado de Sistemas (ICSM31)** — UTFPR.

---

## 1. Visão Geral do Projeto

Este projeto implementa um sistema cliente-servidor para **reconstrução de imagens** a partir de sinais adquiridos por arranjos de sensores, utilizando dois algoritmos iterativos da família do **Gradiente Conjugado**:

- **CGNR** — *Conjugate Gradient Normal Residual*
- **CGNE** — *Conjugate Gradient Normal Error*

### Contexto da Disciplina

A disciplina de Desenvolvimento Integrado de Sistemas tem por objetivo aplicar conceitos de engenharia de software, computação paralela e análise de desempenho na construção de um sistema realista. O problema escolhido — reconstrução de imagens por métodos iterativos — combina álgebra linear de grande porte, comunicação em rede e comparação empírica entre linguagens de programação.

### Arquitetura Geral

```
                          ┌──────────────────────────────┐
                          │   Cliente (Python)           │
                          │   - Carrega H e g            │
                          │   - Aplica ganho de sinal    │
                          │   - Dispara envios paralelos │
                          └────────────┬─────────────────┘
                                       │ HTTP (JSON)
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
        ┌─────────────────────────┐      ┌─────────────────────────┐
        │ Servidor Interpretado   │      │ Servidor Compilado      │
        │ Python + Flask + NumPy  │      │ Go + net/http + gonum   │
        │ Porta 5001              │      │ Porta 5002              │
        └─────────────────────────┘      └─────────────────────────┘
                       │                                │
                       └──────────────┬─────────────────┘
                                      ▼
                          ┌──────────────────────────┐
                          │  Relatório PDF + Imagens │
                          │  (gerado pelo cliente)   │
                          └──────────────────────────┘
```

O cliente envia o **mesmo sinal `g`** para ambos os servidores, garantindo comparação justa entre as duas implementações. Cada servidor executa o algoritmo de reconstrução solicitado e devolve a imagem com metadados.

---

## 2. Algoritmos Implementados

### 2.1 CGNR — Conjugate Gradient Normal Residual

```
f0 = 0
r0 = g - H * f0
z0 = H^T * r0
p0 = z0

for i = 0, 1, ... until convergence:
    w_i    = H * p_i
    α_i    = ||z_i||²₂ / ||w_i||²₂
    f_i+1  = f_i + α_i * p_i
    r_i+1  = r_i − α_i * w_i
    z_i+1  = H^T * r_i+1
    β_i    = ||z_i+1||²₂ / ||z_i||²₂
    p_i+1  = z_i+1 + β_i * p_i
```

### 2.2 CGNE — Conjugate Gradient Normal Error

```
f0 = 0
r0 = g - H * f0
p0 = H^T * r0

for i = 0, 1, ... until convergence:
    α_i    = (r_i^T * r_i) / (p_i^T * p_i)
    f_i+1  = f_i + α_i * p_i
    r_i+1  = r_i − α_i * H * p_i
    β_i    = (r_i+1^T * r_i+1) / (r_i^T * r_i)
    p_i+1  = H^T * r_i+1 + β_i * p_i
```

### 2.3 Definições e Siglas

| Símbolo | Descrição |
| --- | --- |
| g | Vetor de sinal (entrada) |
| H | Matriz de modelo |
| f | Imagem reconstruída (saída) |
| S | Número de amostras do sinal |
| N | Número de elementos sensores |
| ε | Erro de convergência |
| α, β | Coeficientes do gradiente conjugado |

### 2.4 Parâmetros Calculados

```
c = ||H^T * H||₂              # Fator de redução
λ = max(abs(H^T * g)) * 0.10  # Coeficiente de regularização
ε = ||r_i+1||² - ||r_i||²     # Critério de parada (erro)
```

### 2.5 Ganho de Sinal

Antes da reconstrução, aplica-se um ganho ao sinal `g` para compensar atenuações:

```
for c = 1 .. N:
    for l = 1 .. S:
        γ_l = 100 + (1/20) * sqrt(l * l)
        g[l, c] = g[l, c] * γ_l
```

---

## 3. Critério de Parada

O algoritmo para quando uma das condições for satisfeita:

- `|ε| < 1e-4` (erro menor que 0.0001), **OU**
- O número de iterações atingir **10** (máximo permitido).

---

## 4. Requisitos das Imagens Geradas

Cada imagem reconstruída deve conter, obrigatoriamente, como metadados (PNG `tEXt` chunk) **e** como anotação no relatório:

- Identificação do algoritmo utilizado (`CGNR` ou `CGNE`)
- Data e hora de **início** da reconstrução
- Data e hora do **término** da reconstrução
- Tamanho em pixels (ex: `60x60`)
- Número de **iterações** executadas

---

## 5. Especificação do Cliente

O cliente é responsável por:

1. **Carregar** a matriz de modelo `H` e os vetores de sinal `g` a partir do diretório `data/`.
2. **Aplicar o ganho de sinal** descrito em [2.5](#25-ganho-de-sinal) com fator de modelo escolhido **aleatoriamente** a cada envio.
3. **Enviar sequências** de sinais a ambos os servidores em **intervalos de tempo aleatórios** (0.5 s a 3 s).
4. Garantir que o mesmo `g` é enviado para **ambos** os servidores (comparação justa).
5. **Coletar** as imagens retornadas e gerar um **relatório PDF** contendo:
   - Imagem reconstruída
   - Algoritmo utilizado
   - Número de iterações
   - Tempo de reconstrução
   - Servidor de origem (Python ou Go)

---

## 6. Especificação dos Servidores

Dois servidores independentes que respondem à **mesma API HTTP**:

| Característica | Servidor Interpretado | Servidor Compilado |
| --- | --- | --- |
| Linguagem | Python 3.10+ | Go 1.21+ |
| Tipagem | Dinâmica, fraca | Estática, forte |
| Stack | Flask + NumPy + SciPy | net/http + gonum/mat |
| Porta | 5001 | 5002 |

Ambos devem:

- Receber o sinal `g` (e opcionalmente a matriz `H`) via HTTP `POST /reconstruct`
- Executar o algoritmo solicitado (`cgnr` ou `cgne`)
- Retornar a imagem reconstruída (PNG em base64) acompanhada de metadados
- Maximizar throughput (reconstruções por segundo)

A especificação completa do protocolo está em [`shared/protocol.md`](shared/protocol.md).

---

## 7. Modelos de Dados Disponíveis

### Modelo 1 — Imagens 60×60 pixels

- **Matriz H**: dimensão `50816 × 3600`
- **S** = 794 amostras
- **N** = 64 sensores
- Imagens de teste: `img1`, `img2`, `img3` (60×60 px)

### Modelo 2 — Imagens 30×30 pixels

- **Matriz H**: dimensão `27904 × 900`
- **S** = 436 amostras
- **N** = 64 sensores
- Imagens de teste: `img1`, `img2`, `img3` (30×30 px)

> Os arquivos de dados (`H`, `g`, `A`, `F`) são fornecidos pelo professor via Moodle.
> Veja [`data/README_DADOS.md`](data/README_DADOS.md) para a convenção de nomes e instruções de instalação.
>
> **Resumo da convenção:**
> - `H-1.csv` / `H-2.csv` → matrizes de modelo
> - `G-*.csv` / `g-30x30-*.csv` → sinais brutos (cliente envia `apply_gain=true`)
> - `A-*.csv` → sinais com ganho já aplicado (cliente envia `apply_gain=false`)
> - `F-*.png` / `f-30x30-*.png` → imagens ground-truth para comparação

---

## 8. Instalação e Execução

### Pré-requisitos

- Python **3.10+** com `pip`
- Go **1.21+**
- Bibliotecas Python: `numpy`, `scipy`, `Pillow`, `flask`, `requests`, `reportlab`

### Instalação

```bash
# Entrar no projeto
cd icsm31-projeto

# Instalar dependências Python (cliente + servidor interpretado)
pip install -r client/requirements.txt
pip install -r server-interpreted/requirements.txt

# Compilar servidor Go
cd server-compiled
go mod tidy
go build -o server_compiled .
cd ..
```

### Executando

```bash
# Terminal 1 — Servidor interpretado (porta 5001)
python server-interpreted/server.py

# Terminal 2 — Servidor compilado (porta 5002)
./server-compiled/server_compiled

# Terminal 3 — Cliente
python client/client.py
```

O cliente irá:

1. Carregar `H` e `g` do diretório `data/`
2. Disparar várias requisições para ambos os servidores
3. Coletar os resultados
4. Gerar `reports/relatorio_<timestamp>.pdf`

---

## 9. Estrutura do Relatório Comparativo

O relatório final compara os dois servidores e inclui:

- **Tempo médio de reconstrução** por imagem (ms)
- **Número médio de iterações** até convergência
- **Uso de memória** (MB)
- **Ocupação de CPU** (%)
- **Throughput**: imagens reconstruídas por segundo
- **Análise qualitativa** das imagens geradas
- **Conclusões** sobre trade-offs entre linguagem interpretada vs compilada

O cliente gera **automaticamente** um relatório comparativo preenchido
(`reports/relatorio_comparativo_<timestamp>.md`) com tempos médios, desvio
padrão, iterações médias e throughput reais de cada execução — produzido por
[`client/comparative_report.py`](client/comparative_report.py). O template em
branco [`reports/relatorio_comparativo_template.md`](reports/relatorio_comparativo_template.md)
permanece como guia para a análise qualitativa (imagens lado a lado, gráficos).

---

## 10. Estrutura de Diretórios

```
icsm31-projeto/
├── README.md                              ← este arquivo
├── client/                                ← cliente Python
│   ├── client.py
│   ├── report_generator.py
│   └── requirements.txt
├── server-interpreted/                    ← servidor Python (Flask)
│   ├── server.py
│   ├── cgnr.py
│   ├── cgne.py
│   ├── signal_gain.py
│   └── requirements.txt
├── server-compiled/                       ← servidor Go
│   ├── main.go
│   ├── cgnr.go
│   ├── cgne.go
│   ├── signal_gain.go
│   ├── loader.go
│   └── go.mod
├── shared/
│   └── protocol.md                        ← documentação da API
├── data/
│   └── README_DADOS.md                    ← onde colocar H e g
├── reports/
│   └── relatorio_comparativo_template.md
└── docs/
    └── algoritmos.md                      ← teoria dos algoritmos
```

---

## Licença e Créditos

Projeto acadêmico desenvolvido para a disciplina **ICSM31 — Desenvolvimento Integrado de Sistemas**, **UTFPR**.
