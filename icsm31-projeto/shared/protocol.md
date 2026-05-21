# Protocolo de Comunicação Cliente ↔ Servidor

Ambos os servidores (Python e Go) expõem a mesma API HTTP. Isso permite que o cliente use exatamente o mesmo payload contra os dois servidores e compare resultados.

---

## 1. Endpoints

| Método | Rota             | Descrição                                  |
| ------ | ---------------- | ------------------------------------------ |
| `GET`  | `/health`        | Verifica se o servidor está respondendo    |
| `POST` | `/reconstruct`   | Executa uma reconstrução CGNR ou CGNE      |

### Portas padrão

| Servidor    | Porta |
| ----------- | ----- |
| Python      | 5001  |
| Go          | 5002  |

---

## 2. `POST /reconstruct`

### Request — `application/json`

```json
{
  "g": [0.123, 0.456, ...],
  "H": [[...], [...], ...],
  "H_path": "/caminho/absoluto/H_modelo_1.npy",
  "algorithm": "cgnr",
  "model": 1,
  "apply_gain": true
}
```

| Campo         | Tipo              | Obrigatório | Descrição                                                                        |
| ------------- | ----------------- | ----------- | -------------------------------------------------------------------------------- |
| `g`           | `array[float]`    | sim         | Vetor de sinal de comprimento `S * N`                                            |
| `H`           | `array[array[float]]` | não     | Matriz de modelo (S x M). Use apenas para testes; é pesada de serializar         |
| `H_path`      | `string`          | não         | Caminho absoluto local para um arquivo `.npy` (Python) ou `.csv` (Go)            |
| `algorithm`   | `string`          | sim         | `"cgnr"` ou `"cgne"`                                                             |
| `model`       | `int`             | sim         | `1` (60×60) ou `2` (30×30)                                                       |
| `apply_gain`  | `bool`            | não         | Se `true` (padrão), aplica o ganho `gamma_l = 100 + (1/20)*sqrt(l*l)`            |

> Se `H` e `H_path` forem ambos omitidos, o servidor usa o caminho padrão definido pela variável de ambiente `H_MODEL_<N>_PATH` ou, na falta dela, `data/H_modelo_<N>.npy` (Python) / `data/H_modelo_<N>.csv` (Go).

### Response — `application/json`

```json
{
  "algorithm": "CGNR",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "width": 60,
  "height": 60,
  "n_iter": 7,
  "tempo_reconstrucao_s": 1.2345,
  "started_at": "2025-05-20T22:11:00.123Z",
  "finished_at": "2025-05-20T22:11:01.357Z",
  "server": "python"
}
```

| Campo                    | Tipo     | Descrição                                                  |
| ------------------------ | -------- | ---------------------------------------------------------- |
| `algorithm`              | `string` | `"CGNR"` ou `"CGNE"`                                       |
| `image_base64`           | `string` | PNG da imagem reconstruída, codificado em base64           |
| `width`, `height`        | `int`    | Dimensões da imagem em pixels                              |
| `n_iter`                 | `int`    | Número de iterações executadas                             |
| `tempo_reconstrucao_s`   | `float`  | Tempo gasto na chamada do algoritmo (sem rede)             |
| `started_at`             | `string` | Timestamp ISO 8601 (UTC) do início da reconstrução         |
| `finished_at`            | `string` | Timestamp ISO 8601 (UTC) do fim da reconstrução            |
| `server`                 | `string` | `"python"` ou `"go"`                                       |

### Códigos de erro

| Status | Significado                                       |
| ------ | ------------------------------------------------- |
| 200    | Sucesso                                           |
| 400    | Campos obrigatórios faltando ou inválidos         |
| 405    | Método HTTP incorreto (apenas POST permitido)     |
| 500    | Erro interno (falha ao carregar H, etc.)          |

---

## 3. `GET /health`

### Response — `application/json`

```json
{
  "status": "ok",
  "server": "python"
}
```

---

## 4. Convenções de dados

- `g` deve ser **column-major** (`order='F'`): para cada sensor `c` em `[0, N)`, as `S` amostras consecutivas correspondem a `l = 1..S`.
- `H` tem dimensões `(S * N, M)` onde `M = W * H` (pixels).
- A imagem retornada também segue convenção column-major (compatível com `numpy.reshape(order='F')`).

---

## 5. Exemplo de uso (curl)

```bash
curl -X POST http://localhost:5001/reconstruct \
     -H 'Content-Type: application/json' \
     -d '{
           "g": [0.1, 0.2, ...],
           "H_path": "/abs/path/to/H_modelo_1.npy",
           "algorithm": "cgnr",
           "model": 1
         }'
```
