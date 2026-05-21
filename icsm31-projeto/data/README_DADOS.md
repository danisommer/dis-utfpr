# Diretório `data/`

Este diretório armazena as **matrizes de modelo `H`** e os **vetores de sinal `g`** fornecidos pelo professor via Moodle.

> **Os arquivos não estão versionados no repositório** — você precisa baixá-los do Moodle e colocá-los aqui.

---

## 1. Estrutura esperada

```
data/
├── H_modelo_1.npy             ← matriz H do modelo 1 (50816 x 3600)
├── H_modelo_1.csv             ← (opcional) versão CSV para o servidor Go
├── H_modelo_2.npy             ← matriz H do modelo 2 (27904 x 900)
├── H_modelo_2.csv             ← (opcional) versão CSV para o servidor Go
├── sinais_modelo_1/
│   ├── g_img1.npy
│   ├── g_img2.npy
│   └── g_img3.npy
└── sinais_modelo_2/
    ├── g_img1.npy
    ├── g_img2.npy
    └── g_img3.npy
```

Tanto o servidor **Python** quanto o **cliente** preferem `.npy`. O servidor **Go** lê CSV — gere as versões CSV uma única vez, conforme [seção 3](#3-convertendo-de-npy-para-csv).

---

## 2. Modelos disponíveis

| Modelo | Tamanho da imagem | Dimensão de H | S (amostras) | N (sensores) |
| ------ | ----------------- | ------------- | ------------ | ------------ |
| 1      | 60 × 60 px        | 50816 × 3600  | 794          | 64           |
| 2      | 30 × 30 px        | 27904 × 900   | 436          | 64           |

---

## 3. Convertendo de `.npy` para `.csv`

Se o professor fornecer `H_modelo_*.npy`, gere a versão CSV (para o servidor Go) com um snippet simples:

```python
import numpy as np

for m in (1, 2):
    H = np.load(f"H_modelo_{m}.npy")
    np.savetxt(f"H_modelo_{m}.csv", H, delimiter=",", fmt="%.6e")
```

> Atenção: o arquivo CSV é **muito maior** que o `.npy`. Para o modelo 1 (50816×3600), o CSV pode ocupar **~3 GB**. Considere usar o servidor Go com um leitor binário em produção.

---

## 4. Sobrescrevendo os caminhos via variáveis de ambiente

Os servidores procuram a matriz nas seguintes variáveis (na ordem):

1. Campo `H_path` do request JSON.
2. `$H_MODEL_1_PATH` ou `$H_MODEL_2_PATH`.
3. `data/H_modelo_<N>.npy` (Python) ou `data/H_modelo_<N>.csv` (Go).

Exemplo:

```bash
export H_MODEL_1_PATH=/dados/professor/H_60x60.npy
export H_MODEL_2_PATH=/dados/professor/H_30x30.npy
python server-interpreted/server.py
```

---

## 5. Verificação rápida

Antes de rodar o cliente, verifique se tudo está no lugar:

```bash
python - <<'PY'
import numpy as np, os, glob
for m in (1, 2):
    p = f"data/H_modelo_{m}.npy"
    if os.path.exists(p):
        H = np.load(p, mmap_mode="r")
        print(f"OK  {p}: shape={H.shape}")
    else:
        print(f"FALTA {p}")
    for g in glob.glob(f"data/sinais_modelo_{m}/*.npy"):
        v = np.load(g, mmap_mode="r")
        print(f"  signal {g}: len={v.size}")
PY
```
