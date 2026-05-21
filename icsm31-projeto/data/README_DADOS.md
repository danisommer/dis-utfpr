# Diretório `data/`

Este diretório armazena as **matrizes de modelo `H`**, os **vetores de sinal `g`** e as **imagens de referência** fornecidos pelo professor via Moodle.

> Os arquivos não são versionados no repositório — você baixa do Moodle e coloca aqui.

---

## 1. Convenção de nomes (como o professor distribui)

| Arquivo                                  | Conteúdo                                                   | Dimensão        | Modelo |
| ---------------------------------------- | ---------------------------------------------------------- | --------------- | ------ |
| `H-1.csv`                                | Matriz de modelo para imagens **60×60**                    | 50816 × 3600    | 1      |
| `H-2.csv`                                | Matriz de modelo para imagens **30×30**                    | 27904 × 900     | 2      |
| `G-1.csv`, `G-2.csv`                     | Sinais **brutos** (sem ganho aplicado) — modelo 1          | 50816 × 1       | 1      |
| `g-30x30-1.csv`, `g-30x30-2.csv`         | Sinais **brutos** (sem ganho aplicado) — modelo 2          | 27904 × 1       | 2      |
| `A-60x60-1.csv`                          | Sinal **com ganho já aplicado** — modelo 1                 | 50816 × 1       | 1      |
| `A-30x30-1.csv`                          | Sinal **com ganho já aplicado** — modelo 2                 | 27904 × 1       | 2      |
| `F-1.png`, `F-2.png`                     | Imagens **ground-truth** 60×60 (modelo 1)                  | 60 × 60         | 1      |
| `f-30x30-1.png`, `f-30x30-2.png`         | Imagens **ground-truth** 30×30 (modelo 2)                  | 30 × 30         | 2      |

### Diferença entre `G-*`/`g-30x30-*` e `A-*`

- **`G-*`** e **`g-30x30-*`** são sinais "brutos" — valores muito pequenos (próximos de 0). Precisam do **ganho de sinal** ser aplicado antes da reconstrução:

  ```
  γ_l = 100 + (1/20) * sqrt(l * l)
  ```

- **`A-*`** já vem com o ganho aplicado — valores em ordens de magnitude maiores (centenas a milhares).

O cliente (`client/client.py`) detecta automaticamente qual é o caso pelo prefixo do arquivo e envia o flag `apply_gain` correto para o servidor:

| Prefixo do arquivo  | `apply_gain` enviado |
| ------------------- | -------------------- |
| `G-`, `g-30x30-`    | `true`               |
| `A-`                | `false`              |

---

## 2. Estrutura esperada

```
data/
├── H-1.csv                ← matriz H modelo 1 (~679 MB)
├── H-2.csv                ← matriz H modelo 2 (~110 MB)
├── G-1.csv                ← sinal bruto 1 (modelo 1)
├── G-2.csv                ← sinal bruto 2 (modelo 1)
├── g-30x30-1.csv          ← sinal bruto 1 (modelo 2)
├── g-30x30-2.csv          ← sinal bruto 2 (modelo 2)
├── A-60x60-1.csv          ← sinal com ganho (modelo 1)
├── A-30x30-1.csv          ← sinal com ganho (modelo 2)
├── F-1.png, F-2.png       ← ground-truth 60×60
└── f-30x30-1.png          ← ground-truth 30×30
    f-30x30-2.png
```

---

## 3. (Opcional) Acelerar a leitura: converter `H-*.csv` para `.npy`

O arquivo `H-1.csv` tem **~679 MB**, o que torna a primeira leitura lenta. O servidor mantém a matriz em cache na memória, então só o primeiro request é lento — mas você pode pré-converter para `.npy` (NumPy binário) para reduzir o cold start de ~30 s para alguns segundos:

```bash
python - <<'PY'
import numpy as np
for m in (1, 2):
    src = f"data/H-{m}.csv"
    dst = f"data/H-{m}.npy"
    print(f"convertendo {src} -> {dst} ...")
    H = np.loadtxt(src, delimiter=",")
    np.save(dst, H)
    print(f"  shape={H.shape}, dtype={H.dtype}")
PY
```

O servidor Python prefere `.npy` quando ambos existirem. O servidor Go continua usando o `.csv` (não tem leitor `.npy` nativo); se quiser acelerá-lo também, escreva um pequeno conversor para um formato binário Go-friendly.

---

## 4. Sobrescrevendo os caminhos via variáveis de ambiente

Os servidores procuram a matriz na seguinte ordem:

1. Campo `H_path` do request JSON (mais alta prioridade — é o cliente que decide).
2. `$H_MODEL_1_PATH` ou `$H_MODEL_2_PATH`.
3. Padrões: `data/H-<N>.npy`, `data/H-<N>.csv`, `data/H_modelo_<N>.*`.

Exemplo:

```bash
export H_MODEL_1_PATH=/dados/professor/H-1.csv
export H_MODEL_2_PATH=/dados/professor/H-2.csv
python server-interpreted/server.py
```

---

## 5. Verificação rápida

Antes de rodar o cliente, valide o que está no diretório:

```bash
python - <<'PY'
import numpy as np, os, glob
for m in (1, 2):
    h_csv = f"data/H-{m}.csv"
    h_npy = f"data/H-{m}.npy"
    if os.path.exists(h_npy):
        H = np.load(h_npy, mmap_mode="r"); print(f"OK  {h_npy}: shape={H.shape}")
    elif os.path.exists(h_csv):
        with open(h_csv) as f: ncols = len(f.readline().strip().split(","))
        nlines = sum(1 for _ in open(h_csv))
        print(f"OK  {h_csv}: shape=({nlines}, {ncols})  [CSV — converter p/ .npy para acelerar]")
    else:
        print(f"FALTA H modelo {m}")

print()
patterns = {
    1: ["G-*.csv", "A-60x60-*.csv"],
    2: ["g-30x30-*.csv", "A-30x30-*.csv"],
}
for m, pats in patterns.items():
    print(f"-- sinais modelo {m} --")
    for pat in pats:
        for p in sorted(glob.glob(os.path.join("data", pat))):
            v = np.loadtxt(p); kind = "BRUTO  " if "G" in os.path.basename(p)[:2] or "g-" in os.path.basename(p)[:2] else "C/GANHO"
            print(f"  {kind} {p}  len={v.size}  range=[{v.min():.2f}, {v.max():.2f}]")
PY
```

Saída esperada (exemplo):

```
OK  data/H-1.csv: shape=(50816, 3600)  [CSV — converter p/ .npy para acelerar]
OK  data/H-2.csv: shape=(27904, 900)   [CSV — converter p/ .npy para acelerar]

-- sinais modelo 1 --
  BRUTO   data/G-1.csv         len=50816  range=[-0.00, 0.00]
  BRUTO   data/G-2.csv         len=50816  range=[-0.00, 0.00]
  C/GANHO data/A-60x60-1.csv   len=50816  range=[-1506.70, 1545.50]
-- sinais modelo 2 --
  BRUTO   data/g-30x30-1.csv   len=27904  range=[-0.00, 0.00]
  BRUTO   data/g-30x30-2.csv   len=27904  range=[-0.00, 0.00]
  C/GANHO data/A-30x30-1.csv   len=27904  range=[-310340.00, 321500.00]
```
