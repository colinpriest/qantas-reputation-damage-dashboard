
# PCA Axis Labelling v2: Topic-Aware + Diversity-Controlled

## Objectives (delta vs v1)

1. **Topic grounding.** Labels must reference *what the text is about* (e.g., “Regulatory actions”, “Operational disruptions”), not only style/affect.
2. **Diversity across components.** Prevent repeated labels and near-duplicates by optimising for  *global distinctiveness* .
3. **Governance.** Preserve determinism, sign orientation, and stability metrics; log all artefacts.

---

## Inputs

* Text corpus (D={d_i}), embeddings (E\in\mathbb{R}^{N\times p}).
* PCA scores (S\in\mathbb{R}^{N\times K}), loadings (L\in\mathbb{R}^{p\times K}).
* Token features: TF–IDF (uni/bi-grams; vocab ≤ 50k).
* **Topic model artefacts** (new):
  * One of: **BERTopic** (UMAP off),  **NMF** (TF-IDF), or **class-based SVD (cSVD)** on TF-IDF.
  * Outputs: topic–document matrix (T\in\mathbb{R}^{N\times G}) (row-stochastic), topic descriptors (top n-grams, keyphrases).
* Optional: domain taxonomy (curated list of 20–80 domain topics), lexicons, NER counts.
* Optional LLM: **ChatGPT-4o-mini** (temperature=0) for  *name synthesis only* .

---

## High-level pipeline

For each component (PC_k), derive an **axis label** maximising topic alignment and cross-component distinctiveness.

1. **Tail slicing & sign orientation** (unchanged from v1)

   * Slice (H_k, L_k) by top/bottom quantiles on (S_{\cdot k}).
   * Orient sign with the deterministic probe rule so “High” is meaningful.
2. **Topic extraction (global, not per component)**

   * Fit a *single* topic model on the full corpus with deterministic seeds.
   * Recommendations:
     * **NMF** on TF-IDF (default; reproducible, fast). Rank (G \in [50,150]) tuned by coherence.
     * Turn **off** UMAP in BERTopic if used; set `min_topic_size` and fixed seeds.
   * Save for each topic (g): top 20 n-grams, representative docs, coherence, keyness.
3. **Component–topic association** (new core)

   Compute how strongly each (PC_k) aligns with *what* the docs are about.

   * **Linear association:** (r_{k,g}=\mathrm{corr}(S_{\cdot k}, T_{\cdot g})) (Spearman).
   * **Local enrichment:** Over-representation of topic (g) in (H_k) vs (L_k) by log-odds ratio with informative Dirichlet prior (Monroe et al.).
   * **Predictive probe:** L1-logistic predicting (\mathbb{1}[i\in H_k]) using *only* topic proportions (T_{\cdot g}); take non-zero (\beta_g) as drivers with signs.
   * Aggregate to a **Topic Alignment Score (TAS)** per topic:

     [

     \mathrm{TAS} *{k,g}=\alpha,|r* {k,g}|+\beta,\text{norm}(\text{log-odds}_{k,g})+\gamma,\text{norm}(|\beta_g|)

     ]

     with (\alpha=\beta=\gamma=\tfrac{1}{3}) by default.
4. **Facet synthesis (per side)**

   For High and Low tails separately, build  *topic facets* :

   * Take top (M) topics by (\mathrm{TAS}_{k,g}) on that side (respecting sign from probe).
   * For each topic, derive a 3–5 term descriptor from its top n-grams + cTF-IDF within the tail.
   * Produce **Facet Sets** (F_k^{High}, F_k^{Low}) = lists of ((topic_id, descriptor, TAS)).
5. **Axis candidate generation**

   Create 3–5 candidate **bipolar** axis descriptions that combine *topics* +  *style* :

   * Rule: candidate must include ≥1 topic facet per side; style/affect features may appear only as modifiers.
   * Template (deterministic):

     “ ** /  … ⟵ … ⟶  / ** ”
   * Example:

     “ **Regulatory actions & compliance ⟵ … ⟶ Customer service complaints & delays** ”
   * Score each candidate by  **Interpretability Score (IS)** :

     [

     \mathrm{IS}_k = w_t \cdot \text{mean TAS of included topics}

     + w_e \cdot \text{effect size avg (Cohen’s d) of supporting interpretable features}
     + w_c \cdot \text{phrase coherence (PMI of included n-grams)}

     ]

     Defaults: (w_t=0.6,w_e=0.3,w_c=0.1).
6. **Global diversity-aware label assignment** (new)

   Choose one label per component to  **maximise distinctiveness across components** .

   * Build a set of candidates per component: (C_k={c_{k1},...,c_{km}}) with their IS.
   * Define **pairwise similarity** between candidates (c_{ka},c_{lb}):

     * Semantic: cosine between sentence-embedding of the candidate strings.
     * Topical: Jaccard over topic IDs used in the candidates.
     * Final similarity (sim=0.5,cos + 0.5,Jaccard).
   * **Objective:** pick one (c_k\in C_k) for all (k) to maximise

     [

     \sum_k \mathrm{IS}(c_k) ;-; \lambda\sum_{k<l} sim(c_k,c_l)

     ]

     with (\lambda\in[0.2,0.6]) (default 0.4).
   * Solve by greedy beam search or ILP (binary selection) with a triangle-inequality cut; deterministic seeds ensure repeatability.
   * **Diversity constraints:**

     * No two selected labels may share >50% of topic IDs.
     * At most one label whose head noun is from the same **synset** (WordNet/embedding cluster).
7. **Optional LLM naming pass (guard-railed)**

   Use **ChatGPT-4o-mini** (temperature=0, system prompt fixes vocabulary) to *polish* the best candidate for readability without altering topics.

   * Input: the exact facet sets, top terms, and a “do-not-change” list of required topic words.
   * Output: **Title** (≤7 words) + **Bipolar subtitle** (≤15 words).
   * Post-check: reject if semantic similarity to input facets < 0.85 or if required topic words missing.
8. **Stability & drift tests** (extended)

   * Split-half, bootstrap, rotation (as in v1), **plus** topic-model stability:
     * Topic alignment overlap (Jaccard of top-5 topics per component) across bootstraps ≥0.6.
     * If topic model changes materially (topic mapping via Hungarian <0.7 macro-match), flag label for review.
9. **Outputs per component**

```json
{
  "component": "PC_3",
  "axis_label": "Regulatory actions ⟵ … ⟶ Customer service complaints",
  "subtitle": "Compliance & penalties vs delays, cancellations, and call-centre issues",
  "high_topics": [{"id":12,"name":"Regulation & enforcement","TAS":0.82, "ngrams":["ASIC notice","penalty","licence","pursuant"]}],
  "low_topics": [{"id":44,"name":"Service delays & cancellations","TAS":0.77, "ngrams":["cancelled flight","wait time","refund","on hold"]}],
  "supporting_style": {"high":["formal register","+numerals","+MONEY"], "low":["second-person","negative valence"]},
  "global_scores": {"IS":0.74, "distinctiveness_penalty":0.11, "final_score":0.63},
  "stability": {"topic_overlap_bootstrap":0.69, "rotation_stable":true},
  "evidence": {"top_docs_high":[...], "top_docs_low":[...], "cTFIDF_terms_high":[...], "cTFIDF_terms_low":[...]}
}
```

---

## Implementation details

### Topic modelling choices

* **Default:** NMF(k=100) on TF-IDF (ngram_range=(1,2), min_df=5, max_df=0.9), `init='nndsvd'`, fixed `random_state`.
* **Alt:** BERTopic with `HDBSCAN(min_cluster_size=60)`, `umap.n_neighbors=15`,  **umap.random_state fixed** . Turn off dimensionality reduction to keep reproducibility if needed.
* **Keyphrase extraction:** YAKE or cTF-IDF per topic; keep 10–20 terms.

### Correlation/enrichment

* Compute (r_{k,g}) on ranks (Spearman).
* Log-odds w/ informative prior: (\alpha=0.01) per term/topic; report z-scores.

### Candidate generation heuristics

* Use top 2–3 topics per side (by TAS), ensuring *topic coverage* not just variants (e.g., select at most one of {“delays”, “cancellations”, “call-centre”} if they cluster >0.8).
* Add **single-topic axes** when one topic dominates TAS (>0.6) → “(low→high) Numerical specificity” still allowed, but deprioritised if similar axis already used.

### Global selection solver

* Build a candidate graph; run beam search with beam=8; tie-break by higher stability.
* If diversity constraints infeasible, reduce (\lambda) by 0.05 iteratively until feasible.

### LLM guardrails (if enabled)

* System prompt enforces: “Do not invent topics. Use provided facet words verbatim. Write ≤7 word title.”
* Temperature=0, top_p=1.
* Automatic  *semantic regression test* : embed(input facets) vs embed(output title+subtitle) cosine ≥0.85, else revert to deterministic candidate.

---

## Monitoring & artefacts

* **Per-run report** (HTML/Markdown):
  * Selected labels and near-duplicates rejected (with similarity scores).
  * Component–topic heatmap (|r| or TAS).
  * Evidence packs (docs, terms, probes).
* **Drift dashboard** :
* Topic alignment drift per component across model retrains (Procrustes-aligned PCs).
* Diversity index = average pairwise dissimilarity of labels (target ≥0.6).

---

## Edge cases & fallbacks

* **Diffuse component** (no topic TAS > 0.3): label as *Style/Structure* axis (fallback to v1), but mark as “topic-weak” and exclude from diversity penalty.
* **Overlapping components** : if two PCs demand the same dominant topic, enforce differentiation by pairing with *distinct* secondary topics or by **scope qualifier** (e.g., “Regulatory actions (licensing)” vs “Regulatory actions (penalties)”).
* **Small corpora** : drop to (G\in[20,40]); raise quantile width (q) to 20%.

---

## Why this fixes your issues

* **Topic-aware** : Labels are anchored in high-TAS topics, not just emotion/style.
* **Non-repetitive** : Global optimisation penalises semantic/topic overlap and enforces constraints.
* **Auditable** : Every label is backed by topic IDs, terms, and docs; LLM (if used) is post-hoc, deterministic, and verified.
* **Stable** : Split-half + topic stability + rotation tests expose fragile axes before they reach production.

---
