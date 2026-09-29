# Topology-Aware Federated Aggregation — Research Notes & v0 Design

**Date:** 2026-09-28 · **Team:** Vikram, Saffiya, Sanjana, Arshia · **Guide:** Dr. Swati Jagdale
**Status:** research + v0 design. No code written yet.

> **What this document is:** the background for our Phase-5 novelty. It explains what
> "topology" means in the federated-graph-learning literature, how those papers decide two
> clients are *similar*, **why most of their tricks do not directly transfer to our data**,
> and what a minimal, honest first version (v0) should measure. Written to be readable and
> reusable for slides.

---

## TL;DR (the whole thing in six lines)

1. In the literature, "topology" is measured at **four levels**: raw structure, spectral shape, feature/label distributions, and model dynamics.
2. Most graph-FL papers assume clients are **different graphs from different places**.
3. **Our clients are not that.** They are patient-subsets of *one* dataset — they share the same **organism–antibiotic backbone**.
4. Because of (3), the "obvious" structural statistics (degree distribution, density, motif counts) are **almost identical across our hospitals** — they are structurally blind to what actually makes our hospitals different. **We must not lean on them.**
5. What *does* differ: which **patients** (and their **history features**), which bugs/drugs exist locally, their local degrees, the local **resistance rate**, and the two structural axes we already compute — **homophily** (is resistance clustered or scattered?) and **hubness** (how broad is the drug repertoire?).
6. v0 = a **small fingerprint vector** from those signals, z-scored across hospitals, compared with **cosine similarity**, turned into aggregation weights with a **temperature dial** (temperature → ∞ reproduces plain FedAvg exactly).

---

## 1. What "topology" means in the literature (four levels)

"Client similarity" gets computed at four different levels. They are not competing — they
just look at different things.

| Level | Plain-English meaning | What people actually compute | Example papers |
|---|---|---|---|
| **A. Structural statistics** | The "shape" of the local graph: how big it is, how connected, how clustered | node/edge counts, degree distribution, density, clustering coefficient, motif counts, homophily/assortativity | FedStar, FedSSA (partly), classic graph kernels |
| **B. Spectral / subspace** | The graph's "vibration spectrum" — global shape in a compact form | Laplacian eigenvalues, spectral energy, subspace geometry (Grassmann) | FedSSA |
| **C. Feature / label distributions** | What the nodes/edges "mean": feature summaries and predicted-label mixes | neighbor-feature moments, soft-label matrices, homophily confidence | FedGTA, FedSA-GCL, AdaFGL |
| **D. Model dynamics** | What the *model* does while training: gradients and updates | gradient vectors, weight updates over rounds | GCFL/GCFL+, FedDAG, general personalized FL |

**Two ways similarity is then used:**

- **Consensus weighting** — give more weight to clients that look like the global average (FedGTA).
- **Clustering / personalization** — each client only aggregates with its *nearest peers* (GCFL, FedSA-GCL, FedDAG, FedPG).

---

## 2. How the papers actually measure similarity

| Paper (venue) | What the client sends up (the "fingerprint") | How similarity is measured |
|---|---|---|
| **FedGTA** (VLDB 2024) | *mixed moments* of neighbour features + *local smoothing confidence* (a prediction-confidence score) | group clients whose subgraph **feature-moment distributions** match; weight by smoothing confidence |
| **AdaFGL** (2024) | *Homophily Confidence Score*: run K-step label propagation on masked labelled nodes, take the fraction predicted correctly | splits clients into **homophilous vs heterophilous** regimes and propagates differently — no single global model |
| **FedSSA** | *spectral energy* subspace (from spectral-GNN band responses) | embed each client's subspace on the **Grassmann manifold**; distance = **chordal distance** `d = sqrt((K+1) − ‖QₘᵀQₙ‖²_F)`; then cluster |
| **FedStar** | *degree-based structure embedding* (one-hot degree) + random-walk structure embedding | share **only the structure encoder**; keep feature knowledge local |
| **FedSA-GCL** | *Soft Label Feature Matrix* — neighbours' soft labels aggregated | **cosine similarity** between SFMs → mutually aggregate similar clients |
| **GCFL / GCFL+** | GNN **gradients** | cluster clients by gradient similarity; GCFL+ uses **dynamic time warping (DTW)** over gradient *sequences* to smooth noise |
| **FedDAG** | class-wise **data** similarity + **gradient** similarity | weighted combination → a client adjacency matrix → **hierarchical clustering**; dual-encoder per cluster |
| **FedPG** | local **graph context** vectors | **cosine similarity** → personalized fusion |
| **Classic graph similarity** | graphlet counts, Laplacian spectra, heat/wave/commute-time | **kernels / optimal-transport distances** between structural objects |

**The two workhorses:** cosine similarity on a **compact vector**, or a distance/kernel on a
**structural or spectral object**. Nobody sends raw graphs — everything is distilled.

---

## 3. Our setting — and why global structural statistics are the wrong tool here

This is the part that matters most for us. It is also the part most likely to be asked at
the review.

### 3.1 The library analogy

Imagine two libraries. If you only compare **the catalog of books** — how many titles, which
subjects, how many copies — they look almost identical, because they buy from the same
supplier. If you want to know how they *differ*, you look at **who borrowed what**: the
readers, their histories, which subjects get checked out, and how demand clusters.

Our hospitals are those two libraries. They all draw from the **same supplier** — the
organism–antibiotic "test menu" (the bipartite backbone of bugs × drugs). Every hospital
uses the same bugs and the same antibiotics. What differs is **which patients** show up and
**what their tests say**.

### 3.2 Concretely, on our data

- `degree(E. coli)` = *how many distinct antibiotics E. coli was tested against*. This is
  decided by the **hospital's test menu**, not by the patients. It is almost the same in
  every hospital.
- Same for the density of the bug–drug graph, the number of organism nodes, motif counts,
  etc. — they are properties of the **shared backbone**, not of the client.

So a fingerprint built from those statistics would produce **nearly identical vectors for
every hospital**. Cosine similarity would be ≈ 1 for everyone, weights would be ≈ uniform,
and the "topology-aware" method would silently collapse back to FedAvg — with extra
complexity and no gain. **That is the failure mode to avoid.**

### 3.3 What actually differs between our hospitals

| Signal | Why it differs | Already available? |
|---|---|---|
| **Which patients** are in the hospital | the whole point of the split | ✅ |
| **Patient-history features** (prior resistance results — our strongest signal) | different patients = different histories | ✅ (`triple_feat` / `_patient_history_features`) |
| **Local resistance rate** | different patient mixes | ✅ (`label`) |
| **Which organism/antibiotic nodes appear at all**, and their **local** degrees | some patients never received certain tests, so a hospital may not exercise the whole menu | ✅ (compute from the frame) |
| **Homophily** — is resistance *clustered* by bug, or *scattered*? | structural property of the patients' subgraph | ✅ (`partition._patient_homophily`) |
| **Hubness** — how broad is a patient's drug repertoire? | different patient mixes | ✅ (`partition._patient_hubness`) |
| **Community structure** (Louvain on bug–drug) | which bugs share drugs | ✅ (`partition.louvain_split`) |
| **Specimen mix** (urine/blood/resp) | if specimen features are enabled | ✅ (`culture_description`) |

> **Slide soundbite:** *"Our clients share one graph backbone, so classic degree/motif
> statistics are near-identical across hospitals. We fingerprint the signals that actually
> move: patient history, local resistance rate, and the structural axes homophily and
> hubness."*

---

## 4. v0 — the minimal topology module

A **fingerprint** is a short vector per hospital. Only this vector ever leaves the client.

### 4.1 Fingerprint features (v0)

| # | Feature | Meaning (plain) |
|---|---|---|
| 1 | `log1p(#patients)` | size |
| 2 | `log1p(#organisms)` | local bug variety |
| 3 | `log1p(#antibiotics)` | local drug variety |
| 4 | `log1p(#tested edges)` | local bug–drug pairs actually observed |
| 5 | tested density = `#tested / (#org × #abx)` | how much of the local menu got used |
| 6–7 | mean, std of **organism degree** (distinct antibiotics per bug) | how "broad" local bugs are |
| 8–9 | mean, std of **antibiotic degree** (distinct bugs per drug) | how "broad" local drugs are |
| 10 | **resistance rate** (train) | local label balance |
| 11 | **mean homophily deviation** | resistance clustered vs scattered |
| 12 | **mean hubness** | drug-repertoire breadth |
| 13 | mean **patient-history** resistance rate | the winning signal, summarised |
| 14 | number of Louvain communities (optional) | local community richness |

### 4.2 Similarity and weighting

1. Stack all hospitals into a matrix `F` (k hospitals × 14 features).
2. **Z-score each column across clients** (the features have wildly different scales — log-counts vs rates in [0,1]).
3. Because z-scoring makes the population centroid exactly the origin (where cosine is undefined), centrality is measured as **standardised distance-to-centroid** `dᵢ = ‖zᵢ‖` — small = a typical hospital, large = an odd one out. (Pairwise **cosine** between hospitals is still computed for inspection and future clustering.)
4. Turn centrality into aggregation weights with a **temperature** `T`:
   - **Consensus mode:** `wᵢ ∝ exp(−dᵢ/T)` — hospitals closest to the average contribute most (FedGTA-style).
   - **Distinctiveness mode:** `wᵢ ∝ exp(+dᵢ/T)` — deliberately up-weight the *odd* hospital so averaging cannot erase it. This directly targets our known failure (the worst hospital that FedAvg drags down).
5. **Sanity anchor:** as `T → ∞`, both modes become uniform — exactly FedAvg. If our method cannot beat FedAvg, it must at least reproduce it.

### 4.3 Why both modes matter for us

The literature optimises **mean accuracy**, so "consensus" is the default. Our project's
finding is different: on hard splits **FedAvg already beats pooled**, and the damage is
concentrated in the **worst hospital**. Consensus weighting can make that worse. So v0
deliberately tests consensus *and* distinctiveness, and we judge by **worst-hospital F1**
first, mean second.

---

## 5. Where to grow it (v1+)

| Step | Idea | Lifts from |
|---|---|---|
| v1 | **Pairwise + clustered aggregation**: cluster hospitals by `S_ij`, each client receives its nearest cluster's model | FedSA-GCL, FedDAG, GCFL |
| v1 | **Add spectral features**: 1–2 smallest non-trivial Laplacian eigenvalues of the local bug–drug graph, compared by Euclidean distance (full Grassmann/chordal only if needed) | FedSSA |
| v1 | **Add model dynamics**: cosine similarity of model updates `Δw` across rounds (DTW if noisy) | GCFL+, FedDAG |
| v2 | **Homophily confidence** via masked label propagation to decide whether a hospital should aggregate with others at all | AdaFGL |
| v2 | **Privacy hardening**: differential privacy on the fingerprint; quantify what the vector reveals | — |

---

## 6. Evaluation plan (when wired into aggregation)

- Splits: **organism-community (5 hospitals)** and **specimen (3)** first — the two clean wins; then homophily/degree-skew as sensitivity.
- Metrics: **worst-hospital macro-F1** (headline), per-hospital deltas, weighted mean macro-F1, AUROC.
- Protocol: 3 seeds (42/43/44), matched budget, deterministic (client seeding fixed).
- Success criterion (Path-B scaffold): topology-aware best > FedAvg-best by **≥ 0.005** on worst-hospital.
- Sanity: `T → ∞` ≡ FedAvg; similarity matrix should show organism-community clients as mutually *dissimilar* and specimen clients as moderately similar.

---

## 7. References

- **FedGTA** — *Federated Graph Topology-aware Aggregation*, VLDB 2024.
- **AdaFGL** — *Adaptive Federated Graph Learning* (homophily confidence), 2024.
- **FedSSA** — *Heterogeneity-Aware Knowledge Sharing for Graph Federated Learning* (spectral energy, Grassmann/chordal).
- **FedStar** — federated graph learning with degree-based + random-walk structure embeddings.
- **GCFL / GCFL+** — *Federated Graph Classification over Non-IID Graphs* (gradient clustering; DTW).
- **FedSA-GCL** — soft-label feature matrix + cosine similarity.
- **FedDAG** — class-wise data + gradient similarity → hierarchical clustering.
- **OpenFGL** (VLDB 2025) and **FedGraphNN** (2021) — FGL benchmarks and split strategies.
- Classic structure similarity — graphlet spectrum kernels; Laplacian spectral distances (Wasserstein), heat/wave/commute-time; Weisfeiler–Lehman kernel.

---

## 8. v0.5 wiring — from fingerprint to aggregation (implemented)

How the v0 math reaches a live Flower run (driver computes, client forwards, server mixes):

1. **Driver** (`run_fedavg(..., strategy="topology")`): after partitioning, computes one
   `compute_fingerprint(sub)` per hospital from the frame + assignment and stores
   `{cid: [...]}` in `run_config.json` alongside `strategy`, `topo_temperature`,
   `topo_mode`. (The client only sees its saved graph — no timestamps/names — so the
   driver, which has the frame, does the computing. Identical data either way.)
2. **Client** (`client_app.fit`): reads its own vector and packs
   `metrics["topology_fingerprint"] = json.dumps([...])`. Absent vector → the server
   falls back for that client.
3. **Server** (`server_app` + fresh `federated/strategy.py`): `TopologyAwareStrategy`
   subclasses `FedAvg`, overrides only `aggregate_fit` — decode payloads, build
   `TopologyFingerprint`s, `aggregation_weights(...)`, weighted-average the parameters.
   Missing/malformed payloads → size-proportional (FedAvg) fallback with a warning.
   Everything else (sampling, evaluation, initial weights, seeds) is inherited
   unchanged, so the comparison is apples-to-apples. Default strategy stays FedAvg.

Fingerprint hygiene (v0.1, in `topology.select_features`, on by default): drop the
`resistance_rate` column (it duplicates `history_prior_rate` ~1:1) and any column whose
coefficient of variation is below 0.01 (near-constant backbone stats whose noise
z-scoring would amplify to full votes). If nothing survives, weights fall back to
uniform. `temperature=inf` always reproduces FedAvg exactly.

Evaluation driver: `notebooks/07_topology_comparison.ipynb` runs organism-community +
specimen × {FedAvg, topology} at matched protocols, 3 seeds, and prints the Path-B
verdict (topology-aware beats FedAvg-best by ≥ 0.005 on worst-hospital or mean).

---

## Appendix — plain-word glossary

- **Fingerprint** — a small numeric summary of one hospital's graph/data; the only thing shared with the server.
- **Homophily** — whether a bug's resistance outcomes are *consistent* (clustered) across its tested drugs, or scattered.
- **Hubness** — how many *different* drugs a patient's bugs were tested against (breadth of drug repertoire).
- **Cosine similarity** — the angle between two vectors; 1 = same direction, 0 = unrelated, −1 = opposite. Scale-invariant, so good for mixing differently-scaled features after z-scoring.
- **Z-score** — subtract the mean, divide by the standard deviation, so each feature contributes fairly.
- **Softmax temperature `T`** — controls how "peaked" the weights are. Small `T` = a few clients dominate; large `T` = everyone equal (plain FedAvg).
- **Consensus vs distinctiveness weighting** — favouring clients that look like the average, vs favouring the odd ones out.
- **Grassmann manifold / chordal distance** — a way to compare subspaces (used by FedSSA); only relevant if we add spectral features later.
- **DTW (dynamic time warping)** — compares two sequences that may be shifted in time; used by GCFL+ to compare gradient histories.
