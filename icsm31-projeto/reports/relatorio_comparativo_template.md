# Relatório Comparativo — CGNR/CGNE em Python vs Go

**Disciplina:** ICSM31 — Desenvolvimento Integrado de Sistemas
**Universidade:** UTFPR
**Autores:** _<preencher>_
**Data:** _<preencher>_

---

## 1. Introdução e Objetivos

Este relatório compara duas implementações de servidores que executam algoritmos de reconstrução de imagens (CGNR e CGNE):

- **Servidor Interpretado:** Python + Flask + NumPy
- **Servidor Compilado:** Go + net/http + gonum/mat

O objetivo é avaliar **trade-offs** entre linguagens interpretadas (dinamicamente tipadas) e compiladas (estaticamente tipadas) no contexto de computação numérica intensiva.

### Hipóteses iniciais

- _<H1>_ Servidor Go terá menor tempo médio de reconstrução.
- _<H2>_ Servidor Python terá maior throughput em rajadas (pelo paralelismo do NumPy/BLAS).
- _<H3>_ Ambos convergirão em número de iterações similar (são o mesmo algoritmo).

---

## 2. Metodologia de Teste

### 2.1 Ambiente

| Item            | Valor              |
| --------------- | ------------------ |
| Sistema operacional | _<ex: macOS 15.x>_ |
| Hardware (CPU)  | _<ex: Apple M2>_   |
| Memória RAM     | _<ex: 16 GB>_      |
| Python          | _<ex: 3.12>_       |
| Go              | _<ex: 1.22>_       |
| NumPy           | _<versão>_         |
| Gonum           | _<versão>_         |

### 2.2 Carga de teste

- **N reconstruções** por servidor, distribuídas entre CGNR e CGNE.
- **M sinais** distintos por modelo (1 e 2).
- Intervalos aleatórios (0.5 s a 3 s) entre rodadas.
- Medição de tempo via cliente e via servidor (campo `tempo_reconstrucao_s`).

### 2.3 Métricas coletadas

| Métrica                       | Como medir                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------- |
| Tempo de reconstrução (ms)    | Campo `tempo_reconstrucao_s` da resposta (não inclui rede)                      |
| RTT cliente-servidor (ms)     | `time.perf_counter()` no cliente, em volta da chamada `requests.post`           |
| Número de iterações           | Campo `n_iter` da resposta                                                      |
| Uso de memória (MB)           | `psutil`, `Activity Monitor`, ou `ps -o rss` durante a carga                    |
| Ocupação de CPU (%)           | `top -pid <PID>` ou `psutil.cpu_percent` no processo do servidor                |
| Throughput (req/s)            | `total_reqs / tempo_total` em rajadas paralelas (sem o `sleep` entre rodadas)   |

---

## 3. Resultados

### 3.1 Tabela de tempos médios

| Algoritmo | Modelo | Servidor | Tempo médio (ms) | Desvio padrão (ms) | Iterações médias |
| --------- | ------ | -------- | ---------------- | ------------------ | ----------------- |
| CGNR      | 1      | Python   |                  |                    |                   |
| CGNR      | 1      | Go       |                  |                    |                   |
| CGNR      | 2      | Python   |                  |                    |                   |
| CGNR      | 2      | Go       |                  |                    |                   |
| CGNE      | 1      | Python   |                  |                    |                   |
| CGNE      | 1      | Go       |                  |                    |                   |
| CGNE      | 2      | Python   |                  |                    |                   |
| CGNE      | 2      | Go       |                  |                    |                   |

### 3.2 Tabela de recursos

| Servidor | Memória pico (MB) | CPU média (%) | Throughput (req/s) |
| -------- | ----------------- | ------------- | ------------------ |
| Python   |                   |               |                    |
| Go       |                   |               |                    |

### 3.3 Gráficos sugeridos

1. **Boxplot** de tempo de reconstrução por (algoritmo × servidor).
2. **Barplot** de throughput (req/s) por servidor para cada modelo.
3. **Lineplot** de tempo médio em função do número da rodada (para detectar warm-up / aquecimento de cache).
4. **Heatmap** de número de iterações até convergência por par (algoritmo × imagem).

---

## 4. Análise das Imagens Reconstruídas

Inserir aqui imagens lado a lado (Python vs Go) para cada combinação (algoritmo × modelo × imagem teste). Comentar:

- As imagens são **visualmente idênticas**? Diferenças numéricas (norma da diferença em pixels)?
- CGNR vs CGNE: qual produz imagem mais nítida? Mais ruído?
- Modelo 1 (60×60) vs Modelo 2 (30×30): impacto da resolução.

> Norma de diferença esperada entre Python e Go: ≤ 1e-6 (apenas erro de arredondamento de ponto flutuante).

---

## 5. Discussão — Linguagem Interpretada vs Compilada

### 5.1 Performance

- Comparar tempo médio Python (com NumPy/BLAS) vs Go (com gonum).
- Discutir: o "gargalo" do NumPy é o overhead do interpretador, ou o BLAS dominante anula a vantagem do Go puro?
- Avaliar comportamento sob carga concorrente (várias requisições simultâneas).

### 5.2 Desenvolvimento

- Tempo para escrever e depurar cada versão.
- Facilidade de leitura/manutenção.
- Tooling de tipos (mypy vs compilador Go).

### 5.3 Operação

- Tamanho do binário/imagem Docker.
- Tempo de cold start (carregar `H`).
- Footprint de memória.

---

## 6. Conclusão

- Resumo das hipóteses validadas e refutadas.
- Recomendação sobre qual stack adotar para um sistema em produção que precise atender a esse problema.
- Trabalhos futuros: GPU (CUDA / Metal), versão em C++ com Eigen, paralelismo MPI etc.

---

## Apêndice A — Hardware e logs

_<colar `uname -a`, `sysctl -n hw.ncpu`, etc.>_

## Apêndice B — Como reproduzir

```bash
# 1. Iniciar servidores
python server-interpreted/server.py &
./server-compiled/server_compiled &

# 2. Executar cliente
python client/client.py --rounds 50

# 3. O PDF é gerado em reports/relatorio_<timestamp>.pdf
```
