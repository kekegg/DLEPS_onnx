# DLEPS — ONNX build

**Deep Learning based Efficacy Prediction System** for drug discovery.

This repository is the **ONNX build** of DLEPS. It is derived from
[`kekegg/DLEPS_py3`](https://github.com/kekegg/DLEPS_py3) (the Python 3 port of
[`kekegg/DLEPS`](https://github.com/kekegg/DLEPS), which still targets Python 2) with a
single, deliberately small change: **the model inference runs on ONNX Runtime instead of
Keras/TensorFlow.**

* **No TensorFlow, no Keras, no `keras.backend`** — the inference path imports only
  `onnxruntime`, `numpy`, `pandas`, `h5py`, `rdkit` and `nltk`.
* **No `vae.hdf5`** — the exported graph already contains the whole inference stack
  (grammar encoder → VAE sampling → expression head), so the separate Keras grammar
  model is no longer needed at prediction time.
* **Same numbers.** The ONNX graph carries the original Keras weights bit for bit; the
  end-to-end enrichment scores are identical to the Keras build (see
  [Verification](#verification)).

> **If you use DLEPS — the code, the model weights, or any drug-efficacy predictions it
> produces — please cite the paper below.**
> Academic/research use only: commercial use of the code, the weights, or the predictions
> requires prior written permission. See [LICENSE](LICENSE) · [LICENSE.zh-CN.md](LICENSE.zh-CN.md).

## 📖 Citation

> Zhu J#, Wang J#, Wang X#, Gao M, Guo B, Gao M, Liu J, Yu Y, Wang L, Kong W, An Y, Liu Z,
> Sun X, Huang Z, Zhou H, Zhang N, Zheng R, Xie Z\*. Prediction of drug efficacy from
> transcriptional profiles with deep learning. *Nat Biotechnol*. 2021;39(11):1444–1452.
> [doi:10.1038/s41587-021-00946-z](https://doi.org/10.1038/s41587-021-00946-z)

**BibTeX**

```bibtex
@article{zhu2021prediction,
  title   = {Prediction of drug efficacy from transcriptional profiles with deep learning},
  author  = {Zhu, Jie and Wang, Jingxiang and Wang, Xin and Gao, Mingjing and
             Guo, Bingbing and Gao, Miaomiao and Liu, Jiarui and Yu, Yanqiu and
             Wang, Liang and Kong, Weikaixin and An, Yongpan and Liu, Zurui and
             Sun, Xinpei and Huang, Zhuo and Zhou, Hong and Zhang, Ning and
             Zheng, Ruimao and Xie, Zhengwei},
  journal = {Nature Biotechnology},
  year    = {2021},
  volume  = {39},
  number  = {11},
  pages   = {1444--1452},
  doi     = {10.1038/s41587-021-00946-z},
  url     = {https://doi.org/10.1038/s41587-021-00946-z}
}
```

**Machine-readable metadata:** [`citation.json`](citation.json) · [`CITATION.cff`](CITATION.cff) · [`citation.bib`](citation.bib) · [`CITATION.md`](CITATION.md)

**Cited by** — these indexes maintain independent citation graphs, so the numbers legitimately differ. Cite the index you actually consulted, together with its access date (retrieved 2026-10-21).

| Index | Count | Link |
|---|---:|---|
| Google Scholar | 222 | [cites](https://scholar.google.com.hk/scholar?oi=bibs&hl=en&cites=13650249607434417486&as_sdt=5) |
| OpenAlex | 192 | [W3192398644](https://openalex.org/W3192398644) |
| Crossref | 186 | [DOI](https://doi.org/10.1038/s41587-021-00946-z) |
| Semantic Scholar | 160 | [corpus 235470977](https://www.semanticscholar.org/paper/235470977) |

---

# What changed in the ONNX build

Only the **model inference** was touched. Everything downstream of the model — the
one-hot encoding, the dense gene-weight matrix, the connectivity score and the
enrichment score — is byte-identical to `DLEPS_py3`.

| File | Change |
|---|---|
| `code/DLEPS/dleps_predictor.py` | Keras model → `onnxruntime.InferenceSession`; removed the TensorFlow imports and the `sampling()` helper; `train()` now raises `NotImplementedError` |
| `code/DLEPS/molecule_vae.py` | `import models.model_zinc` moved into `ZincGrammarModel.__init__` |

Everything else — `utils.py`, `vectorized_cmap.py`, `zinc_grammar.py`, `driv_DLEPS.py`,
`run_script`, the notebooks — is unchanged from `DLEPS_py3`.

## Why `molecule_vae.py` had to change

`utils.py` needs `get_zinc_tokenizer()` from `molecule_vae.py`, and `molecule_vae.py`
imported `models.model_zinc` at module level. `models/model_zinc.py` imports Keras **and**
creates TensorFlow variables at import time (`masks_K = K.variable(G.masks)`), so the
Keras dependency reached all the way into the one-hot encoder.

Deferring that single import into `ZincGrammarModel.__init__` — where the grammar model
is actually instantiated — cuts the chain:

```
utils.py → molecule_vae.py → (nltk, numpy, zinc_grammar)      # no Keras
                 └→ models/model_zinc.py → Keras               # only when a grammar model is built
```

`models/model_zinc.py` and `models/model_zinc_str.py` still require Keras; they are only
needed for **training** (see [Training](#training)).

## The exported graph

The ONNX file is a `keras2onnx` export of the trained Keras model, and it is a **single
merged graph**:

```
one-hot (N, 277, 76)
  → conv_1 → conv_2 → conv_3 → flatten_1 → dense_1
  → z_mean / z_log_var
  → sampling  (reparameterisation, ε ~ N(0, 0.01))
  → dense_3 → dense_4 → dense_5 → dense_6 → dense_7
expression (N, 978)
```

`277 × 76` is the grammar one-hot encoding produced by `utils.to1hot()` — the same
function the Keras build used, unchanged. Dropout layers are absent, as usual at
inference time.

> **Note on the sampling step.** The VAE head is *stochastic*: `sampling()` adds
> `exp(z_log_var / 2) · ε` with `ε ~ N(0, 0.01)`. Two runs of the **same** model on the
> **same** molecule therefore differ slightly — this is a property of the original model,
> not of the ONNX port. Measured run-to-run spread over 8 repeats: **0.265** for the ONNX
> graph vs **0.274** for the Keras model.

## Ensembles

`DLEPS` accepts a list of ONNX files and averages their outputs, matching the ensemble
rule used by the AIDock server:

```python
from dleps_predictor import DLEPS

predictor = DLEPS(
    up_name='../../data/HUA_up',
    down_name='../../data/HUA_down',
    onnx_dir='../../data/onnx',
    onnx_files=['dleps_0b_0.73.h5.onnx', 'dleps_1_0.726.h5.onnx'],
)
```

Only `dleps_0b_0.73.h5.onnx` is shipped here. The sibling exports
(`dleps_1_0.726`, `dleps_2_0.723`, `dleps_3_0.728`, `transformer_dleps_1b/2b/3b`) come
from the same training run and can be dropped into `data/onnx/` if you want the
ensemble.

---

# Verification

Three independent checks, all reproducible.

### 1. The weights are bit-for-bit identical

`dleps_0b_0.73.h5.onnx` carries exactly the weights of the Keras checkpoint
`DLEPS_30000_tune_gvae10000.h5` that the Keras build loads — **22/22 tensors identical**:

```bash
python scripts/verify_onnx_weights.py \
    --h5   path/to/DLEPS_30000_tune_gvae10000.h5 \
    --onnx data/onnx/dleps_0b_0.73.h5.onnx
```

```
Keras tensor              ONNX initializer        shape            max|diff|    identical
conv_1_2                  conv_1/kernel:0          (9, 76, 9)       0.000e+00    True
conv_2_2                  conv_2/kernel:0          (9, 9, 9)        0.000e+00    True
conv_3_2                  conv_3/kernel:0          (10, 9, 11)      0.000e+00    True
dense_1_2                 dense_1/kernel:0         (2510, 435)      0.000e+00    True
z_mean_2                  z_mean/kernel:0          (435, 56)        0.000e+00    True
z_log_var_2               z_log_var/kernel:0       (435, 56)        0.000e+00    True
dense_3 … dense_7         dense_3 … dense_7        (…, 1024/978)    0.000e+00    True
--------------------------------------------------------------------------
22/22 tensors are bit-for-bit identical
```

(The only transformation is the Conv1D kernel layout: Keras stores
`(kernel_size, in, filters)`, ONNX wants `(filters, in, kernel_size)` — a transpose, which
the script applies before comparing.)

### 2. The graphs compute the same thing

With the sampling noise switched off on both sides (`ε = 0`), the ONNX graph and the
Keras model agree to float32 rounding:

| | max abs difference |
|---|---:|
| ONNX vs Keras, expression output `(N, 978)` | **1.43 × 10⁻⁶** |
| ONNX vs Keras, enrichment scores (full pipeline) | **0.0** (bit-identical) |

### 3. The sampling noise matches too

| | max spread over 8 repeats |
|---|---:|
| Keras model | 0.274 |
| ONNX graph | 0.265 |

---

# Setup

```bash
conda create -n dleps-onnx python=3.11 -y
conda activate dleps-onnx

pip install -r requirements.txt
# or, with the conda-flavoured RDKit:
conda install -c conda-forge rdkit
```

```
Python 3.13.7 · onnxruntime 1.30.0 · NumPy 2.4.2 · pandas 3.0.5
scikit-learn 1.8.0 · RDKit 2025.03.4 · NLTK 3.9.1
```

The supporting files and sample inputs live in the `data/` folder; results are
written to the `results/` folder.

# Run the model

**Input files**

1. A CSV with the chemical SMILES in the column whose header is `string SMILES`;
   the other columns are copied to the output and an efficacy-score column is
   appended.
2. The upregulated gene signatures as ENTREZGENE_ACC, one per line, no header.
3. The downregulated gene signatures, same format.

Gene-name conversion: https://biit.cs.ut.ee/gprofiler/convert

A sample command:

```bash
cd code/DLEPS
python driv_DLEPS.py \
  --input=../../data/Brief_Targetmol_natural_product_2719 \
  --output=../../results/np2719_Browning.csv \
  --upset=../../data/BROWNING_up \
  --downset=../../data/BROWNING_down \
  --reverse=False
```

Batch jobs are in `run_script`.

**Options**

```
'--input',      default=INPUTFILE,  'Brief format of chemicals: contains SMILES column.'
'--use_onehot', default=True,       'If use pre-stored one hot array to save time.'
'--use_l12k',   default=None,       'Use pre-calculated L12k'
'--upset',      default=None,       'Up set of genes'
'--downset',    default=None,       'Down set of genes.'
'--reverse',    default=True,       'If the drug Reverse the Up / Down set of genes.'
'--output',     default='out.csv',  'Output file name.'
```

Jupyter users may run `DLEPS_tutorial.ipynb` for a more interactive workflow.

## Model weights

The **ONNX model ships with this repository**:

```
data/onnx/dleps_0b_0.73.h5.onnx      # 21 MB, the full inference graph
```

Two supporting files are **not** in this repository (they are too large for git and
unchanged from the original project). Download them from the project drive and place
them in `data/`:

```
denseweight.h5                        # 92 MB, maps the 978 predicted genes onto 12,328 genes
SMILE_train_demo.h5                  # only needed for training
SMILE_test_demo.h5
RNA_train_demo.h5
RNA_test_demo.h5
```

https://drive.google.com/drive/folders/1Fye5Bn_ZkT_wjZJIWTcj4RqGXZq-nd4L

`benchmark.csv` and `gene_info.txt` are in the `data/` folder.

## Training

The exported graph is inference-only and cannot be trained further. Training still
requires the Keras build:

* [`kekegg/DLEPS_py3`](https://github.com/kekegg/DLEPS_py3) — Python 3 + Keras
* `code/DLEPS/Training.ipynb` in this repository — the training notebook, kept for
  reference. It needs the Keras build to run; the ONNX build deliberately does not
  carry TensorFlow.
* `code/DLEPS/models/model_zinc.py` — the grammar VAE definition (Keras).

To produce a new ONNX graph, train with the Keras build and export it with
`keras2onnx`; then check the result with `scripts/verify_onnx_weights.py`.

## Repository layout

```
code/DLEPS/            # the model (Python 3, ONNX inference)
├── driv_DLEPS.py      # CLI entry point
├── dleps_predictor.py # DLEPS class: predict / cal_expr / comp_cs
├── molecule_vae.py    # grammar VAE helpers (training only)
├── utils.py           # SMILES → one-hot, fingerprint helpers
├── vectorized_cmap.py # connectivity-score helpers
├── zinc_grammar.py    # zinc-grammar productions
└── models/            # model definitions (Keras — training only)
code/Preprocess/       # R + notebook preprocessing
code/analysis_plot/    # analysis notebooks
data/                  # inputs, gene signatures, benchmark, ONNX weights
scripts/
└── verify_onnx_weights.py   # ONNX ↔ Keras weight comparison
```

# License

**Academic Research Use License 1.0** — see [LICENSE](LICENSE) (authoritative)
and [LICENSE.zh-CN.md](LICENSE.zh-CN.md) (Chinese translation; the English
version prevails). The code, the weights and any predictions produced by this
software are restricted to academic research, education and other non-commercial
scholarly use; commercial use requires prior written permission.

This software is a research tool, not a medical device.
