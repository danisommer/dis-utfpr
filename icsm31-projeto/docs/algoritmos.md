# Algoritmos de Reconstrução por Gradiente Conjugado

Este documento aprofunda a teoria por trás dos dois algoritmos implementados no projeto: **CGNR** e **CGNE**. Para o pseudo-código direto, veja o [`README.md`](../README.md). Aqui o foco é **por que** esses algoritmos funcionam e **quando** preferir um sobre o outro.

---

## 1. Problema

Queremos resolver um sistema linear:

$$
H \cdot f = g
$$

- $H \in \mathbb{R}^{S \times M}$ é a **matriz de modelo** (operador físico do problema).
- $g \in \mathbb{R}^{S}$ é o **vetor de sinal** medido pelos sensores.
- $f \in \mathbb{R}^{M}$ é a **imagem desconhecida** (que queremos reconstruir).

Em reconstrução de imagens por ultrassom (e contextos similares), o sistema é tipicamente:

- **Sobredeterminado** ($S > M$, mais equações que incógnitas).
- **Mal condicionado** (pequenas perturbações em $g$ causam grandes perturbações em $f$).
- **Grande demais** para resolver com inversão direta — $H$ pode ter dezenas de milhões de entradas.

Por isso usamos **métodos iterativos** que trabalham com produtos matriz-vetor e convergem em poucas iterações para uma solução aceitável.

---

## 2. As "Equações Normais"

Multiplicando ambos os lados por $H^\top$:

$$
H^\top H \cdot f = H^\top g
$$

A matriz $H^\top H$ é **quadrada**, **simétrica** e **positiva semi-definida**. Esse é exatamente o tipo de matriz para o qual o **método do Gradiente Conjugado (CG)** original (Hestenes & Stiefel, 1952) foi projetado.

CG aplicado às equações normais leva a duas variantes:

- **CGNR** — minimiza o resíduo na "saída" ($g - Hf$).
- **CGNE** — minimiza o erro na "entrada" ($f^* - f$, onde $f^*$ é a solução de norma mínima).

---

## 3. CGNR — Conjugate Gradient Normal Residual

Resolve, implicitamente:

$$
\min_f \tfrac{1}{2}\,\|Hf - g\|_2^2
$$

ou seja, **mínimos quadrados**.

### Pseudocódigo

```
f0 = 0
r0 = g - H f0           # residuo no espaço dos sinais
z0 = H^T r0             # gradiente das equações normais
p0 = z0                 # direção de busca

para i = 0, 1, ... :
    w_i    = H p_i
    α_i    = ||z_i||²/||w_i||²
    f_i+1  = f_i + α_i p_i
    r_i+1  = r_i − α_i w_i
    z_i+1  = H^T r_i+1
    β_i    = ||z_i+1||²/||z_i||²
    p_i+1  = z_i+1 + β_i p_i
```

### Características

- 1 produto $H \cdot p$ e 1 produto $H^\top \cdot r$ por iteração.
- Converge em no máximo $\text{rank}(H)$ iterações em aritmética exata.
- Bom quando o **ruído está em $g$** (o que tipicamente é o caso de sensores físicos).

---

## 4. CGNE — Conjugate Gradient Normal Error

Resolve:

$$
\min_f \|f\|_2^2 \quad \text{sujeito a} \quad Hf = g
$$

— a solução de **norma mínima**. Equivalente a aplicar CG ao sistema $H H^\top y = g$, com $f = H^\top y$.

### Pseudocódigo

```
f0 = 0
r0 = g - H f0
p0 = H^T r0

para i = 0, 1, ... :
    α_i    = (r_i^T r_i)/(p_i^T p_i)
    f_i+1  = f_i + α_i p_i
    r_i+1  = r_i − α_i H p_i
    β_i    = (r_i+1^T r_i+1)/(r_i^T r_i)
    p_i+1  = H^T r_i+1 + β_i p_i
```

### Características

- Mesmo custo por iteração que CGNR (1 $H p$ + 1 $H^\top r$).
- Tende a produzir **soluções mais "suaves"**.
- Recomendado para problemas **subdeterminados** ou onde se quer regularização implícita.

---

## 5. Critério de Parada

Implementamos um critério dual:

1. **Convergência:** $|\,\epsilon\,| < 10^{-4}$, onde
   - **CGNR**: $\epsilon = \|r_{i+1}\|^2 - \|r_i\|^2$
   - **CGNE**: $\epsilon = \|r_{i+1}\| - \|r_i\|$

2. **Limite duro:** no máximo **10 iterações**, para limitar o tempo de resposta do servidor.

A escolha de poucas iterações se justifica porque, em problemas mal condicionados, iterar demais **amplifica o ruído** (fenômeno conhecido como *semiconvergência*). É preferível parar cedo.

---

## 6. Ganho de Sinal

Antes da reconstrução, aplicamos um ganho dependente do tempo de chegada da amostra:

$$
\gamma_l = 100 + \frac{1}{20}\sqrt{l \cdot l} = 100 + \frac{l}{20}
$$

para $l = 1, \dots, S$. Isso compensa a atenuação progressiva do sinal ao longo do tempo de aquisição (modelo simples do tipo TGC — *Time Gain Compensation*, comum em ultrassom).

A aplicação é em **column-major**: para cada sensor $c \in [0, N)$, multiplicamos as $S$ amostras consecutivas pelo respectivo $\gamma_l$.

---

## 7. Custo Computacional por Iteração

| Operação           | Custo (FLOPs)  | Observação                                |
| ------------------ | -------------- | ----------------------------------------- |
| $H \cdot p$        | $O(S \cdot M)$ | Produto matriz-vetor                      |
| $H^\top \cdot r$   | $O(S \cdot M)$ | Produto matriz-vetor                      |
| Produtos internos  | $O(S + M)$     | Desprezíveis face aos `MV`                |
| Atualizações vetoriais | $O(S + M)$ | Desprezíveis face aos `MV`                |

**Total:** $\approx 2 \cdot S \cdot M$ FLOPs por iteração. Para o Modelo 1 ($S = 50816$, $M = 3600$), isso dá $\approx 3.7 \times 10^8$ FLOPs por iteração.

---

## 8. Implementações neste projeto

| Linguagem | Lib numérica | Arquivo               |
| --------- | ------------ | --------------------- |
| Python    | NumPy        | [`server-interpreted/cgnr.py`](../server-interpreted/cgnr.py) e [`cgne.py`](../server-interpreted/cgne.py) |
| Go        | gonum/mat    | [`server-compiled/cgnr.go`](../server-compiled/cgnr.go) e [`cgne.go`](../server-compiled/cgne.go) |

Ambas as implementações:

- Usam `float64` para máxima precisão.
- Pré-alocam vetores de trabalho para reduzir alocações em laço.
- Cacheiam $H$ em memória para amortizar o custo de leitura entre requisições.

---

## 9. Referências

1. Hestenes, M. R., & Stiefel, E. (1952). *Methods of Conjugate Gradients for Solving Linear Systems*. J. Research Nat. Bur. Standards.
2. Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.
3. Hansen, P. C. (1998). *Rank-Deficient and Discrete Ill-Posed Problems*. SIAM.
4. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
