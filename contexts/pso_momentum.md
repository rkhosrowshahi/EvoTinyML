# Soft-Replacement PSO with Learning Rate and Momentum

A variant of PSO for noisy, data-driven fitness (mini-batch loss). It replaces PSO's **hard, greedy replacement** of the personal and global bests with a **soft, learning-rate-driven move** of those bests, optionally smoothed by first- and second-moment accumulators (Adam-style). Because there is no strict "keep the better one" comparison, the cross-batch incomparability and the optimizer's-curse bias vanish by construction. The cost is that a bad batch nudges the attractors in a wrong direction, so selection bias is traded for update noise (the ordinary SGD regime).

This is deliberately close to the ES philosophy: move a solution with a learning rate and momentum instead of forcing a hard overwrite.

---

## 1. Intuition first

Standard PSO stores a *frozen* best. On noisy fitness that frozen value gets stuck at a lucky low batch and rejects honest challengers, so the swarm stagnates.

Soft replacement removes the freeze entirely. Instead of asking *"is the challenger strictly better? if so, overwrite,"* it asks *"in which direction is the good region, and take a small, momentum-smoothed step toward it."* The best is no longer a stored champion; it is a **slowly moving anchor** that tracks where good solutions have recently been.

Two attractors are treated this way:

- **Personal anchor** $p_i$: drifts toward each particle's own good positions.
- **Global anchor** $g$: drifts toward the swarm's best recent position.

The particle dynamics (velocity, position) are unchanged; only *how the anchors are updated* changes.

---

## 2. Notation

- Objective: full-dataset loss $F(\theta) = \frac{1}{N}\sum_n \ell(\theta; d_n)$.
- Mini-batch estimate at iteration $t$ on shared batch $B_t$ (size $b$):
$$f_t(\theta) = \frac{1}{b}\sum_{n\in B_t}\ell(\theta;d_n),\qquad \mathbb{E}[f_t(\theta)] = F(\theta).$$
- Swarm of $M$ particles: position $x_i$, velocity $v_i$, personal anchor $p_i$, global anchor $g$.
- New hyperparameters (all small, dimensionless where possible):
  - $\eta_p, \eta_g$: learning rates for the personal and global anchor moves.
  - $\beta_1$: first-moment decay (momentum), e.g. $0.9$.
  - $\beta_2$: second-moment decay (adaptive scaling), e.g. $0.999$.
  - $\varepsilon$: small constant for numerical stability, e.g. $10^{-8}$.
- Standard PSO constants: inertia $w$, cognitive $c_1$, social $c_2$; $r_1, r_2 \sim U(0,1)$.

Each anchor carries its own moment buffers: $m^p_i, u^p_i$ for $p_i$ and $m^g, u^g$ for $g$ (first and second moments, same shape as the parameters).

---

## 3. The core idea in one line

Replace

$$p_i \leftarrow \begin{cases} x_i & f(x_i) < f(p_i)\\ p_i & \text{else}\end{cases}\qquad\text{(hard)}$$

with

$$p_i \leftarrow p_i + \eta_p\,\hat s^{\,p}_i,\qquad \hat s^{\,p}_i = \text{momentum-smoothed, variance-scaled version of } (x_i - p_i)\qquad\text{(soft)}$$

and the analogous rule for $g$. The vector $(x_i - p_i)$ is the **direction from the anchor toward the current, freshly scored position**; we only move along it when the current position looks better on the shared batch, and even then only by a small learning-rate step.

---

## 4. Step-by-step algorithm

### Step 0. Initialize

For every particle $i$:

$$x_i \sim \text{init},\quad v_i = 0,\quad p_i = x_i,\quad m^p_i = 0,\quad u^p_i = 0.$$

Global anchor: $g = \arg\min_i f_0(x_i)$ on an initial batch, and $m^g = 0,\ u^g = 0$.

### Step 1. Sample one shared batch

$$B_t = \text{sample\_batch}(b).$$

All particles and all anchors are scored on this **same** $B_t$ this iteration (common random numbers). This is what makes the direction signals fair.

### Step 2. Score candidates and anchors on $B_t$

$$a_i = f_t(x_i),\qquad q_i = f_t(p_i)\quad\text{for all } i,\qquad q_g = f_t(g).$$

$a_i$ scores the current position; $q_i$ re-scores the personal anchor; $q_g$ re-scores the global anchor. Re-scoring on the *current* batch (not a stored old value) is essential: it is what keeps the direction estimate honest.

### Step 3. Form the improvement direction and its sign gate

For the personal anchor, define the raw pull toward the current position:

$$d^p_i = x_i - p_i.$$

Gate it by whether the current position actually looks better on this batch. A soft gate is smoother than a hard 0/1:

$$\gamma^p_i = \sigma\!\Big(\frac{q_i - a_i}{\tau}\Big)\in(0,1),$$

where $\sigma$ is the logistic function and $\tau > 0$ is a temperature (in loss units; can be set to the swarm's paired-difference spread $\hat s_t$ so it self-scales). When the current position is clearly better ($a_i \ll q_i$), $\gamma^p_i \to 1$ and we move fully; when it is clearly worse, $\gamma^p_i \to 0$ and we barely move; near the noise floor we move partially. The gated step is

$$s^p_i = \gamma^p_i\, d^p_i.$$

> Hard-gate variant (cheaper, noisier): $\gamma^p_i = \mathbb{1}[a_i \le q_i]$. The logistic gate is recommended because it degrades gracefully under noise instead of flipping.

For the global anchor, let $x_\star = \arg\min_i a_i$ be the best current position on $B_t$, and define analogously:

$$d^g = x_\star - g,\qquad \gamma^g = \sigma\!\Big(\frac{q_g - a_\star}{\tau}\Big),\qquad s^g = \gamma^g\, d^g.$$

### Step 4. First-moment (momentum) accumulation

Smooth each gated step with an EMA so single-batch noise does not jerk the anchors:

$$m^p_i \leftarrow \beta_1\, m^p_i + (1-\beta_1)\, s^p_i,\qquad m^g \leftarrow \beta_1\, m^g + (1-\beta_1)\, s^g.$$

This is the "momentum" component: the anchor keeps drifting in a consistent direction across iterations and averages out the per-batch jitter.

### Step 5. Second-moment (adaptive scaling) accumulation

Track the per-coordinate magnitude so noisy coordinates take smaller steps:

$$u^p_i \leftarrow \beta_2\, u^p_i + (1-\beta_2)\, (s^p_i)^2,\qquad u^g \leftarrow \beta_2\, u^g + (1-\beta_2)\, (s^g)^2,$$

where $(\cdot)^2$ is elementwise. Coordinates with large, erratic steps get down-weighted; stable coordinates keep their pace. (Omit this step for a pure-momentum version.)

### Step 6. Bias correction (optional, Adam-style)

Early iterations have moments biased toward zero. Correct with the iteration count $t$:

$$\hat m^p_i = \frac{m^p_i}{1-\beta_1^{\,t}},\qquad \hat u^p_i = \frac{u^p_i}{1-\beta_2^{\,t}},$$

and analogously $\hat m^g, \hat u^g$.

### Step 7. Move the anchors (soft replacement)

This is the heart of the method: anchors move by a learning-rate step, never by a hard overwrite.

$$\boxed{\,p_i \leftarrow p_i + \eta_p\,\frac{\hat m^p_i}{\sqrt{\hat u^p_i}+\varepsilon},\qquad g \leftarrow g + \eta_g\,\frac{\hat m^g}{\sqrt{\hat u^g}+\varepsilon}\,}$$

Variants:

- **Soft PSO with momentum (SoftMPSO / `--algo soft_m_pso`):** heavy-ball on the gated step, no separate $\eta$ or bias correction:
  $$m^p_i \leftarrow \beta\, m^p_i + s^p_i,\qquad p_i \leftarrow p_i + m^p_i$$
  (and likewise for $g$). Default $\beta=0.9$.
- **Pure momentum / EMA first moment (1MPSO):** $p_i \leftarrow p_i + \eta_p\, \hat m^p_i$, and likewise for $g$, with $m \leftarrow \beta_1 m + (1-\beta_1)s$.
- **Plain soft step (no moments):** $p_i \leftarrow p_i + \eta_p\, s^p_i$. This is the minimal form: $p_i \leftarrow p_i + \eta_p\,\gamma^p_i (x_i - p_i)$, i.e. a gated exponential move toward good positions.

Because the update is additive and learning-rate scaled, there is **no comparison to freeze**, so the cross-batch and optimizer's-curse problems cannot occur. The price: if $\gamma$ mis-fires on a bad batch, the anchor drifts slightly wrong, exactly the SGD noise trade-off.

### Step 8. Standard PSO dynamics (unchanged)

$$v_i \leftarrow w\, v_i + c_1 r_1\,(p_i - x_i) + c_2 r_2\,(g - x_i),\qquad x_i \leftarrow x_i + v_i.$$

The particles are still pulled toward the personal and global anchors as in ordinary PSO; only the anchors themselves now move softly.

### Step 9. Loop

Return to Step 1 until the budget is exhausted. Return $g$ (or a Polyak/tail average $\bar g$ of $g$ over the last iterations for extra stability).

---

## 5. Compact pseudocode

```
# init
for i in 1..M:
    x[i] ~ init;  v[i] = 0;  p[i] = x[i]
    mp[i] = 0;  up[i] = 0
g = argmin_i f(x[i], B_0);  mg = 0;  ug = 0

for t in 1..T:
    B_t = sample_batch(b)                       # shared batch

    # score current positions and anchors on the SAME batch
    for i in 1..M:
        a[i] = f(x[i], B_t)
        q[i] = f(p[i], B_t)
    qg = f(g, B_t)
    istar = argmin_i a[i];  a_star = a[istar]

    tau = std(a - q) + eps                       # self-scaling temperature

    # ---- personal anchors: soft move ----
    for i in 1..M:
        gate = sigmoid((q[i] - a[i]) / tau)
        s    = gate * (x[i] - p[i])
        mp[i] = b1*mp[i] + (1-b1)*s
        up[i] = b2*up[i] + (1-b2)*s*s
        mhat  = mp[i] / (1 - b1**t)
        uhat  = up[i] / (1 - b2**t)
        p[i]  = p[i] + eta_p * mhat / (sqrt(uhat) + eps)

    # ---- global anchor: soft move ----
    gate_g = sigmoid((qg - a_star) / tau)
    sg     = gate_g * (x[istar] - g)
    mg = b1*mg + (1-b1)*sg
    ug = b2*ug + (1-b2)*sg*sg
    mhat_g = mg / (1 - b1**t)
    uhat_g = ug / (1 - b2**t)
    g = g + eta_g * mhat_g / (sqrt(uhat_g) + eps)

    # ---- PSO dynamics: unchanged ----
    for i in 1..M:
        v[i] = w*v[i] + c1*r1*(p[i]-x[i]) + c2*r2*(g-x[i])
        x[i] = x[i] + v[i]

return g   # or Polyak average of g over last iterations
```

---

## 6. Why each piece is there

| Component | Layer it acts on | What it fixes / does |
|---|---|---|
| Score anchors on the shared $B_t$ (Step 2) | estimation | Makes the improvement direction a same-batch, honest signal (CRN). |
| Soft gate $\gamma = \sigma((q-a)/\tau)$ (Step 3) | selection | Replaces the hard 0/1 keep-or-reject; degrades gracefully under noise. |
| Learning-rate anchor move (Step 7) | selection | Removes the freeze entirely, so no cross-batch/curse bias. |
| First moment $m$ (Step 4) | dynamics of the anchor | Averages per-batch noise; consistent drift direction. |
| Second moment $u$ (Step 5) | dynamics of the anchor | Per-coordinate step scaling; damps noisy coordinates. |
| $\tau = \text{std}(a-q)$ (Step 3) | estimation | Self-scaling temperature so the gate tracks the noise floor. |
| Unchanged $v,x$ update (Step 8) | dynamics | Keeps the swarm's exploration and social/cognitive structure. |

---

## 7. Hyperparameter starting points

- $\eta_p \in [0.1, 0.5]$, $\eta_g \in [0.05, 0.3]$ (smaller for $g$: it should move slower than the personal anchors).
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$ (Adam defaults transfer well).
- $\tau = \hat s_t$ (swarm paired-difference std) so it self-scales; or a fixed small value in loss units if you prefer.
- Standard PSO: $w \in [0.4, 0.9]$ (optionally decayed), $c_1 = c_2 \approx 1.5$–$2.0$.
- If you want the minimal version, drop the second moment and bias correction and use the plain soft step $p_i \leftarrow p_i + \eta_p\,\gamma^p_i(x_i - p_i)$.

---

## 8. Trade-offs and when to prefer it

**Prefer soft replacement when:**

- Fitness is noisy (mini-batch loss) and hard-replacement PSO stagnates.
- You want ES-like stability (learning rate + momentum) but keep a swarm of attractors for exploration.
- You are comfortable tuning a few learning-rate / momentum knobs.

**Be aware that:**

- With no hard comparison, a mis-fired gate on a bad batch drifts the anchor slightly wrong: you trade **selection bias** for **update noise** (the SGD regime). Momentum and the soft gate mitigate this but do not eliminate it.
- Pushed to the extreme ($\gamma \equiv 1$, no gate), the personal-best concept dissolves and the method becomes an ES with a swarm-shaped update. The gate is what preserves the "best" semantics.
- It introduces more hyperparameters than plain PSO. If you want the *smallest* change to fix stagnation while staying purely PSO, the shared-batch re-scoring with an adaptive acceptance margin is lighter; soft replacement is the choice when you specifically want the learning-rate-plus-momentum behavior.

---

## 9. Relationship to the other approaches

- **vs. hard-replacement PSO:** removes the freeze at the root (no comparison to get stuck on).
- **vs. re-scoring + adaptive margin:** both re-score on the shared batch; that method keeps hard selection but makes it noise-robust, whereas this one abandons hard selection for a soft move. Re-scoring targets the *bias*; this method sidesteps the bias entirely at the cost of update noise.
- **vs. EMA on the best score:** that reduces *variance* of a stored estimate; this reduces reliance on any stored estimate at all by moving the anchor continuously.
- **vs. ES / SGD-Adam:** essentially imports the ES update rule into PSO's anchors while retaining the swarm's velocity dynamics and multi-attractor exploration.