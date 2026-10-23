# DLEPS — Python 3 port

**Deep Learning based Efficacy Prediction System** for drug discovery.

This repository is the **Python 3 port** of [`kekegg/DLEPS`](https://github.com/kekegg/DLEPS),
which still targets Python 2. The original remains the canonical reference for the
method and the paper; this repository is the maintained, installable version.

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

# What changed in the Python 3 port

The bulk of the conversion is mechanical and was produced by **`2to3`**
(`python -m lib2to3 -w -n code/DLEPS/`). Two categories of change were needed on
top of it, because `2to3` only rewrites *syntax* — it knows nothing about
library APIs.

## 1. Mechanical (`2to3`)

| File | Change |
|---|---|
| `code/DLEPS/dleps_predictor.py` | dropped the now-redundant `from __future__ import print_function` / `division` |
| `code/DLEPS/driv_DLEPS.py` | `print(x)` → `print((x))` |
| `code/DLEPS/utils.py` | `print(x)` → `print((x))` |
| `code/DLEPS/molecule_vae.py` | `map(str, rhs[ix])[::-1]` → `list(map(str, rhs[ix]))[::-1]` — **required**: in Python 3 `map` returns an iterator, which cannot be sliced |
| `code/DLEPS/molecule_vae.py` | builtin `reduce` → `from functools import reduce` |

## 2. Library-API migration (not covered by `2to3`)

The original code targets `tensorflow>=1.15` / `keras>=2.3`. On a modern
TensorFlow stack the following modules no longer exist:

| Before (Keras 2.3) | After (Keras 2.15) |
|---|---|
| `from keras import objectives` | `from keras import losses as objectives` |
| `from keras.layers.core import ...` | `from keras.layers import ...` |
| `from keras.layers.wrappers import TimeDistributed` | `from keras.layers import TimeDistributed` |
| `from keras.layers.recurrent import GRU` | `from keras.layers import GRU` |
| `from keras.layers.convolutional import Convolution1D` | `from keras.layers import Conv1D as Convolution1D` |

The `objectives` → `losses` swap is **semantically exact**, not an
approximation: Keras 2.3.1 defined

```python
# keras/losses.py, Keras 2.3.1
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    ...
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)
```

and Keras 2.15 defines `keras.losses.binary_crossentropy` with the same
`mean(..., axis=-1)` reduction. The call sites are unchanged.

## 3. Division semantics — checked, unchanged

`2to3` cannot prove whether `a / b` was meant as integer or true division, so
every `/` was audited individually with `tokenize`. All eight are safe:

* `dleps_predictor.py` already imported `from __future__ import division`, so
  its two divisions were true division under Python 2 as well;
* `utils.py` forces float with `i * 1.0 / len(smiles)`;
* `models/model_zinc*.py` divide Keras tensors (always float);
* `vectorized_cmap.py` divides pandas Series (always float).

No `//` was introduced, because none of them were integer divisions.

## 4. Notebooks

All seven notebooks already carried Python 3 metadata (`python3` kernel,
`language_info.version = 3.7.3`) and contain no Python 2 syntax, so they were
copied unchanged.

## Verified

```
Python 3.9.7 · TensorFlow 2.15.0 · Keras 2.15.0 · NumPy 1.26.4
pandas 2.2.3 · scikit-learn 1.6.1 · RDKit 2023.03.3 · NLTK 3.9.1
```

* All seven modules import cleanly under Python 3.
* `python driv_DLEPS.py --help` runs.
* The VAE graph builds: encoder 1,102,985 params (output `(None, 2)`),
  decoder 3,821,723 params (output `(None, 277, 89)`), autoencoder
  4,924,708 params.

---

# Setup

```bash
conda create -n dleps-py3 python=3.9 -y
conda activate dleps-py3

pip install -r requirements.txt
# or, with the conda-flavoured RDKit:
conda install -c conda-forge rdkit
```

> `lib2to3` was removed from Python 3.13, so if you want to re-run the
> conversion yourself use Python ≤ 3.12 (`python -m lib2to3 -w -n code/DLEPS/`).

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

The weights are **not** in this repository. Download them from the project
drive and place them in `data/`:

```
denseweight.h5
DLEPS_30000_tune_gvae10000.h5
SMILE_train_demo.h5
SMILE_test_demo.h5
RNA_train_demo.h5
RNA_test_demo.h5
```

https://drive.google.com/drive/folders/1Fye5Bn_ZkT_wjZJIWTcj4RqGXZq-nd4L

`benchmark.csv` is in the `data/` folder.

## Repository layout

```
code/DLEPS/            # the model (Python 3)
├── driv_DLEPS.py      # CLI entry point
├── dleps_predictor.py # DLEPS class: predict / cal_expr / comp_cs
├── molecule_vae.py    # MoleculeVAE (zinc-grammar VAE)
├── utils.py           # SMILES → one-hot, fingerprint helpers
├── vectorized_cmap.py # connectivity-score helpers
├── zinc_grammar.py    # zinc-grammar productions
└── models/            # model definitions (Keras)
code/Preprocess/       # R + notebook preprocessing
code/analysis_plot/    # analysis notebooks
data/                  # inputs, gene signatures, benchmark
```

# License

**Academic Research Use License 1.0** — see [LICENSE](LICENSE) (authoritative)
and [LICENSE.zh-CN.md](LICENSE.zh-CN.md) (Chinese translation; the English
version prevails). The code, the weights and any predictions produced by this
software are restricted to academic research, education and other non-commercial
scholarly use; commercial use requires prior written permission.

This software is a research tool, not a medical device.
