Here’s a concrete, end-to-end labelling algorithm that turns each PCA dimension (from sentence-embedding PCA) into an intuitive, defensible “axis label.” It is fully automated, LLM-free by default (so deterministic), and comes with stability checks and documentation artefacts.

# Goal

For each PCA component PCkPC_k**P**C**k**, produce:

* A human-readable **axis label** (e.g., “Formality ⟵ casual … formal ⟶”).
* A short **explanation** describing what drives high vs. low scores.
* **Evidence packs** (top texts, keywords, features) and **stability metrics** so model risk/governance can sign off.

---

# Inputs

* Corpus of texts D={di}i=1ND=\{d_i\}_{i=1}^N**D**=**{**d**i****}**i**=**1**N**.
* Embeddings E∈RN×pE\in\mathbb{R}^{N\times p}**E**∈**R**N**×**p (one vector per did_i**d**i****).
* PCA scores S∈RN×KS\in\mathbb{R}^{N\times K}**S**∈**R**N**×**K (component scores per text) and loadings L∈Rp×KL\in\mathbb{R}^{p\times K}**L**∈**R**p**×**K.
* Optional lexicons/dictionaries (choose any mix you like for your domain):
  * Sentiment/emotion (e.g., valence, anger, anxiety).
  * Style/formality (function words, pronouns, contractions).
  * Domain dictionaries (e.g., risk, finance, claims, safety, ops).
  * LIWC-like category counts (or any public alternative).
* Tokenised text features (bag-of-words and bigrams) with TF-IDF.

---

# Core labelling procedure (run independently for each PCkPC_k**P**C**k**)

### 1) Score & slice the corpus along the component

* Let si=Siks_i = S_{ik}**s**i=**S**ik**** be the score of document ii**i** on PCkPC_k**P**C**k**.
* Create  **tails** :
  * HkH_k**H**k: indices in the top q%q\%**q**% (e.g., 10%) of sis_i**s**i.
  * LkL_k**L**k: indices in the bottom q%q\%**q**% of sis_i**s**i.
  * (Using tails maximises contrast and interpretability.)

### 2) Build *interpretable* features on the tails

Compute per-document, then aggregate within HkH_k**H**k and LkL_k**L**k:

* **Lexical stats** : length, type–token ratio, punctuation rates, numerals rate, uppercase rate.
* **Lexicons** : sentiment (mean valence), emotion counts, style markers (pronouns, modals, hedges), domain dictionaries.
* **N-grams** : TF-IDF unigrams/bigrams (pretrained stoplist; keep top 5k).
* **Entities** : counts by type (ORG, PER, LOC, MONEY, DATE) if NER is available.

### 3) Find what *discriminates* the tails (three complementary, deterministic tests)

Use all tests; then ensemble their evidence.

**(A) Class-based TF-IDF (cTF-IDF)**

* Treat HkH_k**H**k and LkL_k**L**k as two “classes.” Compute cTF-IDF to get terms over-represented in each tail.
* Keep top mm**m** (e.g., 20) terms per side with scores.

**(B) Sparse linear probe from interpretable features**

* Train two L1-regularised logistic regressions (deterministic solver) to predict 1[i∈Hk]\mathbb{1}[i\in H_k]**1**[**i**∈**H**k] vs. others and 1[i∈Lk]\mathbb{1}[i\in L_k]**1**[**i**∈**L**k] vs. others using *only interpretable features* (no embeddings).
* Keep non-zero coefficients ranked by absolute value as **drivers** for High and Low.

**(C) Information-theoretic contrast**

* For each interpretable feature ff**f**, compute:
  * Δμf=μHk(f)−μLk(f)\Delta \mu_f = \mu_{H_k}(f)-\mu_{L_k}(f)**Δ**μ**f****=**μ**H**k******(**f**)**−**μ**L**k******(**f**)
  * standardized effect (Cohen’s dd**d**), and
  * mutual information I(f; tail)I(f;\, \text{tail})**I**(**f**;**tail**) (discretise ff**f** into quantiles).
* Keep top features by II**I** and ∣d∣|d|**∣**d**∣**.

> Rationale: (A) gives  **words/phrases** , (B) gives **multivariate drivers** with signs, (C) gives **univariate strength** with effect sizes. Agreement across all three → strong label.

### 4) Orient the component (fix the arbitrary PCA sign)

PCA signs are arbitrary. To make labels stable:

* Define a **reference score** for each side:

  RkH=∑f∈F∗sign⁡(β^f)⋅zf(Hk),RkL=∑f∈F∗sign⁡(β^f)⋅zf(Lk)R^H_k=\sum_{f\in F^*}\operatorname{sign}(\hat\beta_f)\cdot z_f(H_k),\quad
  R^L_k=\sum_{f\in F^*}\operatorname{sign}(\hat\beta_f)\cdot z_f(L_k)**R**k**H****=**f**∈**F**∗**∑sign**(**β**^f)**⋅**z**f****(**H**k)**,**R**k**L=**f**∈**F**∗**∑****sign**(**β**^****f****)**⋅**z**f****(**L**k****)**
  where F∗F^***F**∗ are the top non-zero L1 features and zf(⋅)z_f(\cdot)**z**f****(**⋅**) are standardized means.
* If RkH<RkLR^H_k < R^L_k**R**k**H****<**R**k**L****, multiply S⋅kS_{\cdot k}**S**⋅**k** and L⋅kL_{\cdot k}**L**⋅**k** by −1-1**−**1 and **swap** Hk,LkH_k, L_k**H**k****,**L**k.
* This locks the meaning of “High” vs “Low” deterministically.

### 5) Synthesize candidate axis labels

Create candidates from mutually reinforcing evidence:

**Term-based candidates.**

* From cTF-IDF, take the top 3–5 bigrams per side → summarise as short facets.
  * Example: High: {“policy wording”, “legal compliance”}; Low: {“customer frustration”, “wait times”}.

**Feature-based candidates.**

* From L1 probe, take top 3 signed drivers per side and map to plain language rules:
  * “High = more numerals, more MONEY entities, longer sentences.”
  * “Low = more second-person pronouns, higher exclamation rate, negative valence.”

**Dictionary/lexicon candidates.**

* From MI/effect sizes, translate leading lexicon categories into short facets:
  * “High = procedural/formal; Low = emotional/complaint.”

**Compose axis name.**

* Choose the **shortest phrase** that covers ≥2 methods’ signal on each side.
* Format as a  **bipolar axis** :
  * **Formality ⟵ conversational / emotive … procedural / formal ⟶**
* If the signal is clearly *unipolar* (e.g., “dates & amounts density”), use:
  * **Numerical‐specificity (low → high)**

Include a one-sentence  **explanation** :

> “High scores feature long, formal sentences with monetary and legal terms; low scores are conversational, second-person, and negative-valence.”

### 6) Evidence pack for auditability

For each PCkPC_k**P**C**k**, save:

* Top 30 texts in HkH_k**H**k and LkL_k**L**k (IDs and snippets).
* Top 20 cTF-IDF terms (per side) with scores.
* L1 coefficients table (feature, sign, magnitude).
* MI and Cohen’s dd**d** tables for all interpretable features.
* A small **facet cloud** image: bigrams and dictionary categories ranked.

### 7) Stability & robustness checks (report alongside the label)

* **Split-half reliability** : Randomly split documents into halves; re-run steps 1–5; compute Jaccard overlap of top terms and Kendall’s τ\tau**τ** of feature rankings. Require ≥0.6/0.6 (tune).
* **Bootstrap label stability** : 200 bootstraps of DD**D**. Compute how often the final axis name (or its synonym set) recurs. Report a **Label Stability Index** (LSI: 0–1).
* **Rotation sensitivity** : Compare labels after varimax rotation of the first KK**K** PCs. If the component’s nearest rotated factor (by absolute correlation) preserves the same label >80% of bootstraps, mark  **rotation-stable** .
* **Retrain alignment** (when embeddings/PCA are refit): Use Procrustes to align new PCs to old; ensure the matched component preserves label >70% of the time. If not, flag “label drift.”

### 8) Final output schema (per component)

<pre class="overflow-visible!" data-start="6358" data-end="7639"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"component"</span><span>:</span><span></span><span>"PC_7"</span><span>,</span><span>
  </span><span>"axis_label"</span><span>:</span><span></span><span>"Formality ⟵ conversational / emotive … procedural / formal ⟶"</span><span>,</span><span>
  </span><span>"explanation"</span><span>:</span><span></span><span>"High = longer sentences, MONEY/DATE entities, legal/contract terms; Low = second-person, exclamations, negative valence."</span><span>,</span><span>
  </span><span>"top_terms_high"</span><span>:</span><span></span><span>[</span><span>"policy wording"</span><span>,</span><span></span><span>"contract clause"</span><span>,</span><span></span><span>"per annum"</span><span>,</span><span></span><span>"pursuant to"</span><span>,</span><span></span><span>"hereby"</span><span>]</span><span>,</span><span>
  </span><span>"top_terms_low"</span><span>:</span><span></span><span>[</span><span>"you said"</span><span>,</span><span></span><span>"still waiting"</span><span>,</span><span></span><span>"so frustrating"</span><span>,</span><span></span><span>"why hasn't"</span><span>,</span><span></span><span>"call me"</span><span>]</span><span>,</span><span>
  </span><span>"top_features_high"</span><span>:</span><span></span><span>[</span><span>{</span><span>"feature"</span><span>:</span><span>"avg_sentence_len"</span><span>,</span><span>"coef"</span><span>:</span><span>0.42</span><span>}</span><span>,</span><span>
                        </span><span>{</span><span>"feature"</span><span>:</span><span>"MONEY_entities"</span><span>,</span><span>"coef"</span><span>:</span><span>0.31</span><span>}</span><span>,</span><span>
                        </span><span>{</span><span>"feature"</span><span>:</span><span>"numerals_rate"</span><span>,</span><span>"coef"</span><span>:</span><span>0.29</span><span>}</span><span>]</span><span>,</span><span>
  </span><span>"top_features_low"</span><span>:</span><span></span><span>[</span><span>{</span><span>"feature"</span><span>:</span><span>"second_person_rate"</span><span>,</span><span>"coef"</span><span>:</span><span>-0.37</span><span>}</span><span>,</span><span>
                       </span><span>{</span><span>"feature"</span><span>:</span><span>"exclamation_rate"</span><span>,</span><span>"coef"</span><span>:</span><span>-0.26</span><span>}</span><span>,</span><span>
                       </span><span>{</span><span>"feature"</span><span>:</span><span>"neg_valence"</span><span>,</span><span>"coef"</span><span>:</span><span>-0.22</span><span>}</span><span>]</span><span>,</span><span>
  </span><span>"stability"</span><span>:</span><span></span><span>{</span><span>"split_half_jaccard_terms"</span><span>:</span><span>0.71</span><span>,</span><span>
                </span><span>"split_half_kendall_features"</span><span>:</span><span>0.68</span><span>,</span><span>
                </span><span>"bootstrap_LSI"</span><span>:</span><span>0.83</span><span>,</span><span>
                </span><span>"rotation_stable"</span><span>:</span><span>true</span><span></span><span>}</span><span>,</span><span>
  </span><span>"documentation"</span><span>:</span><span></span><span>{</span><span>
    </span><span>"high_tail_examples"</span><span>:</span><span>[</span><span>{</span><span>"id"</span><span>:</span><span>"doc_1032"</span><span>,</span><span>"score"</span><span>:</span><span>3.14</span><span>,</span><span>"snippet"</span><span>:</span><span>"..."</span><span>}</span><span>,</span><span>
                          </span><span>{</span><span>"id"</span><span>:</span><span>"doc_778"</span><span>,</span><span>"score"</span><span>:</span><span>3.02</span><span>,</span><span>"snippet"</span><span>:</span><span>"..."</span><span>}</span><span>]</span><span>,</span><span>
    </span><span>"low_tail_examples"</span><span>:</span><span>[</span><span>{</span><span>"id"</span><span>:</span><span>"doc_55"</span><span>,</span><span>"score"</span><span>:</span><span>-2.89</span><span>,</span><span>"snippet"</span><span>:</span><span>"..."</span><span>}</span><span>,</span><span> ...</span><span>]</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## Practical implementation notes

* **Quantiles** : q=10%q=10\%**q**=**10%** works well; if N<10,000N<10{,}000**N**<**10**,**000**, use q=15%q=15\%**q**=**15%**.
* **Preprocessing** : lower-case, strip boilerplate, normalise repeated punctuation before TF-IDF.
* **N-grams** : Use unigrams+bigrams; min doc frequency 3–5; cap vocab at 50k; no stemming (you want readable labels).
* **Logistic probe** : Standardise features; use liblinear or saga with fixed random_state; C chosen by nested CV (but keep grid fixed).
* **MI** : Discretise continuous features into deciles to avoid parametric assumptions.
* **Varimax** (optional but helpful): Improves sparsity/interpretability; keep labels tied to **scores** via the sign-orientation rule above.
* **No LLM needed** : The entire pipeline is deterministic. If you *do* add a summarising LLM step, keep it post-hoc and templated, and store the deterministic evidence alongside.

---

## Why this is robust and explainable

* **Contrastive** : Labels emerge from systematic differences between high vs. low tails—easy to audit.
* **Multi-evidence** : Words, features, and information metrics must agree; reduces the risk of spurious labels.
* **Signed & stable** : Explicit sign-orientation eliminates PCA sign flips.
* **Quantified uncertainty** : Split-half, bootstrap, and rotation tests expose fragility.
* **Governance-ready** : Every label ships with examples, drivers, and metrics; changes are trackable across retrains.

---

## Optional extensions (if you want even crisper axes)

* **Sparse/rotated PCs** : Try SparsePCA or ICA to sharpen components before the labelling procedure (then run the same steps).
* **Dictionary projection** : Build a small set of hand-curated domain axes (e.g., “Complaintness,” “Numerical specificity,” “Regulatory tone”) by linear probes; report correlations with PCkPC_k**P**C**k**.
* **Canonical naming** : Maintain a controlled vocabulary so recurring components are named consistently (e.g., “Formality,” “Complaintness,” “Numbers & Dates”).
