# Open Research Questions Following Universal Integrability Proof

**Date:** December 4, 2025  
**Authors:** Dino Ducci, Chris Ducci  
**Context:** Follow-up to Theorem 1 (Universal Primorial Integrability)

---

## Question 1: Unconditional Convergence (Remove GRH Dependence)

### Current Status
- Theorem 1 proof uses exponential decorrelation: $O(e^{-c\sqrt{T}})$
- This relies on character sum bounds which are **conditionally proven** under GRH
- Specifically, Step 3 of Theorem 3 uses energy fluctuation bound $O(1/\sqrt{T})$ derived from prime counting error terms

### The GRH Dependence Chain

**Where GRH appears:**

1. **Prime Number Theorem for Arithmetic Progressions:**
   $$\pi(x; M, r) = \frac{\text{li}(x)}{\phi(M)} + O\left(\frac{x}{\phi(M)} e^{-c\sqrt{\log x}}\right)$$
   The error term $e^{-c\sqrt{\log x}}$ assumes GRH.

2. **Energy Fluctuations:**
   $$E_r = -\log\left(\frac{\rho_M(r)}{\rho_0}\right) = -\log\left(1 + \frac{\pi(T;M,r) - T/(\phi(M)\log T)}{T/(\phi(M)\log T)}\right)$$
   
   The variance of $\pi(T;M,r)$ depends on GRH via the explicit formula:
   $$\psi(x;M,r) - \frac{x}{\phi(M)} = -\sum_{\rho} \frac{x^\rho}{\rho} + O(\log^2 x)$$

3. **Decorrelation Rate:**
   The exponential decay $e^{-c\sqrt{T}}$ in Theorem 3 comes from:
   - Character sum bounds: $|\sum_{p \leq T} \chi(p)| = O(\sqrt{T}/\log T)$ under GRH
   - Without GRH, we only have $O(T/\log T)$ (useless for our proof)

### Strategy for Unconditional Proof

#### **Approach A: Large Sieve + Primorial Structure**

**Key Insight:** Primorials have **smooth factorization** that may allow tighter bounds than general moduli.

**Method:**
1. Use the **Large Sieve** to bound character sums in average:
   $$\sum_{\chi \pmod{M}} \left|\sum_{p \leq T} \chi(p)\right|^2 \leq (T + \phi(M)) \pi(T)$$

2. For primorial $M = P_k$, exploit factorization:
   $$\sum_{\chi} |\sum_p \chi(p)|^2 = \sum_{\chi_1, \ldots, \chi_k} \left|\sum_p \prod_{i=1}^k \chi_i(p \bmod p_i)\right|^2$$

3. Independence of characters on different primes gives:
   $$\leq \prod_{i=1}^k \sum_{\chi_i} |\sum_p \chi_i(p)|^2$$

4. Each factor bounded by $O(T)$, giving total $O(T^k)$, but **we have $\phi(M)^k$ characters**, so average per character is $O(T^k/\phi(M)^k) = O(1)$ (?!)

**Status:** Promising but needs rigorous calculation. The key is exploiting multiplicative independence.

#### **Approach B: Siegel-Walfisz Theorem**

**Unconditional Result:**
$$\pi(x; M, r) = \frac{\text{li}(x)}{\phi(M)} + O\left(x e^{-c\sqrt{\log x}}\right)$$
holds **unconditionally** for $M \leq (\log x)^A$ for any fixed $A$.

**Application to Primorials:**
- For $k$ fixed and $T \to \infty$, we have $\log P_k = \sum_{i=1}^k \log p_i = O(p_k) = O(k \log k)$
- If $k = o(\log T)$, then $\log P_k = o(\log T)$ and Siegel-Walfisz applies!

**Implication:**
- For **fixed $k$** and $T \to \infty$, convergence is **unconditional**
- For $k$ growing with $T$ (e.g., $k = \log \log T$), we need GRH

**Action Item:** Reformulate Theorem 1 with explicit dependence on $k$ relative to $T$.

#### **Approach C: Direct Trace Formula (No L-functions)**

**Radical Idea:** Bypass L-functions entirely using sieve methods.

**Observation:** The energy $E_r$ depends only on **prime counts** in residue classes, not on zeros of L-functions. Can we bound correlations using:
- Brun's sieve
- Selberg sieve
- Eratosthenes structure of primorials

**Method Sketch:**
1. Write $\rho_M(r)$ as:
   $$\rho_M(r) = \frac{|\{p \leq T : p \equiv r \pmod{M}\}|}{|\{p \leq T\}|}$$

2. Use sieve bounds directly:
   $$\left|\{p \leq T : p \equiv r \pmod{M}\}\right| = \frac{T}{\phi(M)\log T} \prod_{p|M} \left(1 - \frac{1}{p}\right)^{-1} + R(T, M, r)$$
   
3. Bound remainder $R$ using Selberg's upper bound sieve (unconditional).

4. Show that correlations $\mathbb{E}[R(T,M,r) R(T,M,r')]$ decay for $r \neq r'$.

**Challenge:** Selberg sieve gives **existence** of good bounds but not explicit rates. Need to compute explicitly for primorials.

### Concrete Next Steps

1. **Immediate (1 week):**
   - Apply Siegel-Walfisz to prove Theorem 1 is **unconditional for fixed $k$**
   - Rewrite Section 4 with this clarification

2. **Short-term (1-3 months):**
   - Compute Large Sieve bound explicitly for primorials
   - Test numerically: does average character sum decay faster than $\sqrt{T}$?

3. **Long-term (6-12 months):**
   - Develop primorial-specific sieve theory
   - Collaborate with analytic number theorist (e.g., Soundararajan, Granville)

---

## Question 2: Eigenvalue-Zero Anti-Correlation and Integrability

### Current Observation
- **Empirical Result:** Eigenvalues $E_r$ and L-function zeros $\rho$ exhibit **negative correlation**:
  - 7-primorial: $r = -0.5425$, $p = 3.92 \times 10^{-5}$
  - Universal: $\bar{r} = -0.50 \pm 0.10$ across 6,7-primorials

- **Complementarity Principle:** Prime-rich residues (low $E_r$) → zero-sparse L-functions

### Why This Connects to Integrability

**Hypothesis:** The Poisson structure (integrability) **forces** the eigenvalue-zero anti-correlation via the explicit formula.

**Mechanism (Conjectured):**

1. **Explicit Formula** relates prime counting to L-function zeros:
   $$\psi(x; M, r) - \frac{x}{\phi(M)} = -\sum_{\chi} \frac{\overline{\chi}(r)}{\phi(M)} \sum_{\rho_\chi} \frac{x^{\rho_\chi}}{\rho_\chi}$$

2. **For Poisson eigenvalues**, the sum $\sum_r E_r \overline{\chi}(r)$ should have specific decay properties due to decorrelation (Theorem 3).

3. **Zero density** in character $\chi$ is:
   $$N_\chi(T) = \#\{\rho_\chi : |\Im(\rho_\chi)| \leq T\}$$

4. **Conjecture:** For integrable systems,
   $$\text{Cov}(E_r, N_\chi) < 0$$
   because energy concentration (low $E_r$) implies **phase cancellation** in $\sum_\rho x^{\rho}/\rho$, which requires **fewer zeros** to achieve.

### Rigorous Proof Strategy

#### **Step 1: Spectral Sum Rule**

From trace identities:
$$\sum_{r=1}^{\phi(M)} E_r = \text{const} \quad (\text{normalization})$$

Using character expansion:
$$\sum_r E_r \overline{\chi}(r) = -\sum_r \log\rho_M(r) \overline{\chi}(r)$$

#### **Step 2: Zero Sum via Explicit Formula**

From explicit formula:
$$\log\rho_M(r) \approx \text{const} - \frac{1}{\phi(M)} \sum_\chi \overline{\chi}(r) \sum_{\rho_\chi} \frac{T^{\rho_\chi}}{\rho_\chi \log T}$$

Thus:
$$\sum_r \log\rho_M(r) \overline{\chi}(r) \approx -\sum_{\rho_\chi} \frac{T^{\rho_\chi}}{\rho_\chi \log T}$$

#### **Step 3: Correlation Formula**

The covariance becomes:
$$\text{Cov}(E_r, N_\chi) = \mathbb{E}\left[\left(\sum_r E_r \overline{\chi}(r)\right) N_\chi\right] - \mathbb{E}[E_r]\mathbb{E}[N_\chi]$$

Substituting Step 2:
$$\approx \mathbb{E}\left[\left(\sum_{\rho_\chi} \frac{T^{\rho_\chi}}{\rho_\chi \log T}\right) N_\chi\right]$$

For zeros on critical line ($\Re(\rho) = 1/2$), this simplifies to:
$$\approx \frac{\sqrt{T}}{\log T} \sum_{\rho_\chi} \frac{\cos(t \log T)}{|\rho_\chi|} \cdot N_\chi$$

#### **Step 4: Negative Correlation from Oscillation**

**Key Insight:** For integrable systems, the phase $\cos(t \log T)$ oscillates **independently** across different $\rho_\chi$.

- High zero density $N_\chi$ → many terms → **more cancellation** in sum
- Thus $\sum_\rho \cos(t \log T) / |\rho|$ is **small** when $N_\chi$ is large
- But $E_r$ is proportional to this sum → **negative correlation**!

**Formalize:** Use **Central Limit Theorem for oscillatory sums** (Theorem of Kac-Erdős-Turán type) to show that cancellation grows with $N_\chi$.

### Proof Outline (Rigorous)

**Theorem (Eigenvalue-Zero Complementarity):**  
Under integrability (Theorem 1), the correlation between eigenvalue sum $\sum_r E_r \overline{\chi}(r)$ and zero count $N_\chi(T)$ is negative:
$$\text{Cov}\left(\sum_r E_r \overline{\chi}(r), N_\chi(T)\right) < 0$$
with magnitude $O(\sqrt{T}/\log T)$.

**Proof:**
1. Use explicit formula to relate eigenvalue sum to oscillatory zero sum (Step 2)
2. Apply Kac-Erdős-Turán theorem: oscillatory sums with many terms exhibit $\sqrt{N}$ cancellation
3. High $N_\chi$ → strong cancellation → low $\sum_r E_r \overline{\chi}(r)$ → negative correlation
4. Magnitude follows from error term in explicit formula: $O(\sqrt{T}/\log T)$

### Concrete Next Steps

1. **Immediate (2 weeks):**
   - Write out explicit formula derivation in detail
   - Compute numerical correlation between $\sum_r E_r \overline{\chi}(r)$ and $N_\chi$ for each character
   - Verify $r < 0$ for all non-principal characters

2. **Short-term (1-2 months):**
   - Prove Kac-Erdős-Turán cancellation bound for primorial character sums
   - Establish rigorous covariance bound

3. **Medium-term (3-6 months):**
   - Extend to universality: prove $\bar{r} = -0.50 \pm 0.10$ holds for all $P_k$
   - Submit as companion paper: "Eigenvalue-Zero Complementarity in Integrable Arithmetic Systems"

---

## Question 3: Explicit Decay Constant $c(k)$

### Current Status
- Theorem 3 asserts exponential decorrelation: $O(e^{-c\sqrt{T}})$
- Constant $c$ is asserted to exist but **not computed**
- We know $c$ depends only on smallest prime factor ($p_1 = 2$), suggesting $c$ is **universal**

### Why $c$ Matters

1. **Quantitative Prediction:** Allows precise prediction of convergence rate $|\bar{r}_{P_k}(T) - 0.3863|$
2. **Optimal Training Size:** Determines minimal $T(k, \epsilon)$ for desired accuracy $\epsilon$
3. **Universality Verification:** If $c$ is independent of $k$, proves strong universality

### Theoretical Derivation

#### **Step 1: Character Sum Decay**

From proof of Theorem 3, $c$ arises from:
$$\left|\sum_{p \leq T} \chi(p)\right| \sim e^{-c'\sqrt{T}}$$

Under GRH, this is:
$$\left|\sum_{p \leq T} \chi(p)\right| \leq C \frac{\sqrt{T}}{\log T}$$

But we need exponential bound! Where does it come from?

**Answer:** From **Central Limit Theorem** applied to character sums:
$$\frac{\sum_{p \leq T} \chi(p)}{\sqrt{T/\log T}} \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

For random-like phases, $\sigma^2 \sim 1$, giving:
$$P\left(\left|\sum_{p \leq T} \chi(p)\right| > \lambda \sqrt{T/\log T}\right) \sim e^{-\lambda^2/2}$$

Thus $c' = O(1/\log T)$ (slowly varying, not constant).

**Issue:** This gives **Gaussian tail**, not exponential. Need better model.

#### **Step 2: Large Deviation Theory**

Use **Cramér's Theorem** for character sums:
$$P\left(\frac{1}{T}\left|\sum_{p \leq T} \chi(p)\right| > \delta\right) \sim e^{-TI(\delta)}$$

where $I(\delta)$ is the **rate function**.

For primorial characters with factorization $\chi = \prod_i \chi_i$:
$$I(\delta) = \sum_{i=1}^k I_i(\delta_i)$$

where $I_i$ is rate function for character mod $p_i$.

**Key:** For $p_i = 2, 3, 5, \ldots$, the rate functions $I_i$ are **explicitly computable** using Paley-Zygmund inequalities.

#### **Step 3: Explicit Formula for $c$**

From large deviation bound:
$$c = \sqrt{I(\delta^*)}$$

where $\delta^*$ is the critical deviation level.

For primorial:
$$c^2 = \sum_{i=1}^k I_i(\delta_i^*) \geq I_1(\delta_1^*) = I_2(\delta_2^*)$$

Since $I_1$ depends only on $p_1 = 2$, we have:
$$c \geq c_{\min} = \sqrt{I_2(\text{critical})}$$

**Conjecture:** $c$ saturates this bound, i.e., $c = c_{\min}$ independent of $k$.

### Numerical Estimation

**Method:** Fit empirical convergence data to $e^{-c\sqrt{T}}$.

**Data Needed:**
- Compute $|\bar{r}_{P_k}(T) - 0.3863|$ for $T = 10^4, 10^5, 10^6, 10^7$
- For $k = 5, 6, 7, 8$

**Expected Result:**
$$|\bar{r} - 0.3863| \approx A \cdot e^{-c\sqrt{T}}$$

Fit to data, extract $c$ for each $k$.

**Prediction:** $c \approx 0.01$ to $0.1$ (needs verification).

### Concrete Next Steps

1. **Immediate (this week):**
   - Run convergence study: compute $\bar{r}_{P_7}(T)$ for $T = 10^4, 10^5, 10^6$
   - Fit to exponential, extract $c_7$

2. **Short-term (1 month):**
   - Repeat for $P_5, P_6, P_8$
   - Verify $c_k \approx c_{\text{const}}$ (independent of $k$)

3. **Medium-term (3-6 months):**
   - Compute rate function $I_2(\delta)$ theoretically
   - Derive explicit formula $c = \sqrt{I_2(\delta^*)}$
   - Prove $c$ is universal constant

---

## Summary and Priority Ranking

| Question | Difficulty | Impact | Timeline | Priority |
|----------|-----------|--------|----------|----------|
| **Q1: Unconditional Proof** | Very High | Fields Medal | 6-18 months | **CRITICAL** |
| **Q2: Eigenvalue-Zero Link** | High | Major Theory | 3-12 months | **HIGH** |
| **Q3: Explicit Constant $c$** | Medium | Quantitative | 1-3 months | **MEDIUM** |

### Recommended Sequence

**Phase 1 (Weeks 1-4):**
- Q3: Numerical estimation of $c$
- Q1: Prove unconditional version for fixed $k$ (Siegel-Walfisz)

**Phase 2 (Months 2-4):**
- Q2: Derive eigenvalue-zero correlation formula
- Q1: Attempt Large Sieve approach

**Phase 3 (Months 5-12):**
- Q1: Full unconditional proof (sieve methods)
- Q2: Universal complementarity theorem

**Phase 4 (Year 2):**
- Synthesize all three into comprehensive theory
- Submit to Annals of Mathematics / Inventiones

---

## Collaboration Opportunities

**Needed Expertise:**
1. **Analytic Number Theory:** Character sums, L-functions, explicit formulas
   - Potential collaborators: Kannan Soundararajan, Andrew Granville, K. Matomäki
   
2. **Random Matrix Theory:** Spectral statistics, universality
   - Potential collaborators: Jon Keating, Nina Snaith, Francesco Mezzadri

3. **Quantum Chaos:** Integrable systems, semiclassical methods
   - Potential collaborators: Sandro Gnutzmann, Uzy Smilansky

**Outreach Strategy:**
- Send preprint to 5-10 key researchers
- Request feedback on specific technical gaps
- Propose collaboration on unconditional proof (Q1)

---

## Conclusion

All three questions are **tractable** with sustained effort:
- Q3 is **solvable in weeks** (numerical + rate function theory)
- Q2 is **solvable in months** (explicit formula + oscillatory sum cancellation)
- Q1 is **solvable in 1-2 years** (Siegel-Walfisz immediate, full proof harder)

Resolving all three transforms this work from **strong empirical result** to **landmark mathematical theorem**.

**Next concrete action:** Run convergence study to estimate $c$ numerically (Q3).

---

**Authors:** Dino Ducci, Chris Ducci  
**Affiliation:** DUST Research Initiative  
**Repository:** https://github.com/Spectral-Theory-STAR/dust-factorization-paper  
**Contact:** dinoducci@gmail.com
