#!/usr/bin/env python
########################################################
# All rights reserved.
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

"""Local DLEPS scoring on the ONNX build.

This is the engine behind the ``calculate_dleps_score`` tool that the drug
discovery skills use.  It runs the **DLEPS_onnx** graph
(``data/onnx/dleps_0b_0.73.h5.onnx``) through ONNX Runtime, so no
TensorFlow and no Keras are needed -- only ``onnxruntime``, ``numpy``,
``pandas``, ``h5py``, ``rdkit`` and ``nltk``.

Two entry points are provided:

* :class:`DlepsScorer` -- the reusable API.
* ``python scripts/dleps_score.py --disease "Gout" --smiles "CCO,CC(=O)O"``
  -- a thin command line front end.

The prediction itself is the *unchanged* DLEPS model: one-hot grammar
encoding -> ONNX graph -> 978 gene expression deltas -> enrichment score
against the up/down signature of a disease.  A score above 0.2 indicates a
candidate that is likely to reverse the disease signature.

Data files
----------
``data/onnx/*.onnx``
    Shipped with the repository.
``data/denseweight.h5``
    The dense gene-weight matrix (978 predicted genes -> 12,328 genes).
    ~92 MB, therefore **not** in git.  Resolved from, in order:
    ``$DLEPS_DENSEWEIGHT``, ``data/denseweight.h5``,
    ``~/.cache/dleps/denseweight.h5``.
``data/<signature>``
    The up / down gene signatures of every supported disease, listed in
    ``data/diseases.json``.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / 'code' / 'DLEPS'
DATA_DIR = REPO_ROOT / 'data'

#: A candidate counts as "effective" above this reversal score.
EFFECTIVE_THRESHOLD = 0.2

#: Modules the DLEPS code imports.  They are purged from ``sys.modules``
#: before loading so a foreign module of the same name cannot be picked up.
_RELATED_MODULES = ('dleps_predictor', 'molecule_vae', 'utils',
                    'vectorized_cmap', 'zinc_grammar',
                    'models', 'models.model_zinc')


# --------------------------------------------------------------------------
# data / disease resolution
# --------------------------------------------------------------------------

def resolve_denseweight(override=None):
    """Return the path of ``denseweight.h5`` or raise a helpful error."""
    candidates = []
    if override:
        candidates.append(Path(override))
    env = os.environ.get('DLEPS_DENSEWEIGHT')
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(DATA_DIR / 'denseweight.h5')
    candidates.append(Path(os.path.expanduser('~/.cache/dleps/denseweight.h5')))
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        'denseweight.h5 not found. Looked at:\n  ' + '\n  '.join(str(c) for c in candidates)
        + '\nDownload it from https://drive.google.com/drive/folders/1Fye5Bn_ZkT_wjZJIWTcj4RqGXZq-nd4L'
          '\nand put it in data/ (or set $DLEPS_DENSEWEIGHT).')


def load_diseases(path=None):
    """Load the disease -> signature mapping (``data/diseases.json``)."""
    p = Path(path) if path else DATA_DIR / 'diseases.json'
    with open(p, encoding='utf-8') as f:
        return json.load(f)['diseases']


def _norm(name):
    return ''.join(ch for ch in str(name).lower() if ch.isalnum() or ch in ' -_')


def resolve_disease(name, diseases=None):
    """Map a free-form disease name onto ``(key, spec)``.

    Matching order: exact key -> case/space-insensitive key -> alias.
    """
    diseases = diseases if diseases is not None else load_diseases()
    if name in diseases:
        return name, diseases[name]
    target = _norm(name)
    for key, spec in diseases.items():
        if _norm(key) == target:
            return key, spec
    for key, spec in diseases.items():
        for alias in spec.get('aliases', []):
            if _norm(alias) == target:
                return key, spec
    raise KeyError(
        'Unknown disease %r. Available: %s'
        % (name, ', '.join('%s (%s)' % (k, v.get('label', '')) for k, v in diseases.items())))


def signature_paths(disease_key, spec, data_dir=None):
    """Resolve the up/down signature files of a disease.

    ``spec['down']`` may be ``None``: the score is then the enrichment of the
    "up" set alone.  That is what the upstream ``run_script`` does for the
    expert browning screen (``--upset=Browning_Expert_UP`` with no
    ``--downset``).
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    up = data_dir / spec['up']
    down = data_dir / spec['down'] if spec.get('down') else None
    missing = [str(p) for p in (up, down) if p is not None and not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Signature file(s) missing for %s: %s' % (disease_key, ', '.join(missing)))
    return up, down


def signature_direction(spec):
    """Return the ``reverse`` flag of a disease spec.

    ``True``  -- the signature states a disease state; a positive score means
                 the compound reverses it.
    ``False`` -- the signature states a therapeutic phenotype; a positive
                 score means the compound mimics it (browning).
    """
    return bool(spec.get('reverse', True))


# --------------------------------------------------------------------------
# gene list helpers
# --------------------------------------------------------------------------

def load_gene_table(data_dir=None):
    """Return the 12,328-gene table (``pr_gene_id`` -> gene record)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / 'gene_info.txt'
    if not path.exists():
        raise FileNotFoundError('gene_info.txt not found at %s' % path)
    import pandas as pd
    return pd.read_table(str(path), header=0)


def load_gene_correlations(data_dir=None):
    """Return ``{gene_id: correlation}`` for the 12,328 predicted genes.

    The correlation is the per-gene quality of the DLEPS prediction; the
    AIDock server exposes it as the "Gene Correlation Cutoff" (0 - 0.8), where
    0 keeps every gene and 0.68376 keeps only the well-predicted ones.
    The values live in ``data/corr_genes_12328.csv`` (the same table as
    ``gene_info.txt`` plus the correlation column).
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / 'corr_genes_12328.csv'
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(str(path))
    col = '0' if '0' in df.columns else df.columns[-1]
    return {int(g): float(c) for g, c in zip(df['pr_gene_id'], df[col])}


def read_signature(path):
    """Read a signature file and return the list of gene ids (ints)."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                out.append(int(line))
            except ValueError:
                continue
    return out


def check_genes(gene_ids, data_dir=None):
    """Split gene ids into (present in the 12,328 list, not present).

    This is what the AIDock client's "Check the genes" button does: it shows
    the submitted ids that the model can actually use and the ones it has
    never heard of.
    """
    table = load_gene_table(data_dir)
    known = set(int(g) for g in table['pr_gene_id'])
    present, missing = [], []
    for g in gene_ids:
        try:
            gi = int(g)
        except (TypeError, ValueError):
            missing.append(str(g))
            continue
        (present if gi in known else missing).append(gi)
    return present, missing


def filter_signature_by_correlation(path, cutoff, correlations, data_dir=None):
    """Write a copy of a signature file keeping only well-predicted genes.

    Mirrors the AIDock server: a cutoff of 0 (or outside 0 - 0.8) keeps every
    gene, otherwise only the genes whose prediction correlation is above the
    cutoff survive.  Returns the path of the (possibly temporary) file that
    should be handed to the model.
    """
    path = Path(path)
    if not (0 < cutoff < 0.8) or not correlations:
        return path
    genes = read_signature(path)
    kept = [g for g in genes if correlations.get(g, 0.0) > cutoff]
    if not kept:
        raise ValueError(
            'Gene Correlation Cutoff %.5f removed every gene of %s; lower it.'
            % (cutoff, path.name))
    tmp = tempfile.NamedTemporaryFile('w', suffix='_' + path.name,
                                       prefix='dleps_geneCorr_', delete=False,
                                       encoding='utf-8')
    try:
        tmp.write('\n'.join(str(g) for g in kept) + '\n')
    finally:
        tmp.close()
    return Path(tmp.name)


# --------------------------------------------------------------------------
# scorer
# --------------------------------------------------------------------------

class DlepsScorer(object):
    """Score candidate molecules against a disease signature with DLEPS_onnx."""

    def __init__(self, repo_root=None, onnx_dir=None, onnx_files=None,
                 denseweight=None, reverse=True, setmean=False):
        self.repo_root = Path(repo_root).resolve() if repo_root else REPO_ROOT
        self.code_dir = self.repo_root / 'code' / 'DLEPS'
        self.data_dir = self.repo_root / 'data'
        self.onnx_dir = Path(onnx_dir) if onnx_dir else self.data_dir / 'onnx'
        self.onnx_files = onnx_files
        self.denseweight = resolve_denseweight(denseweight)
        self.reverse = reverse
        self.setmean = setmean
        self._dleps_cls = None
        self._correlations = None

    # -- plumbing ----------------------------------------------------------

    @contextlib.contextmanager
    def _in_code_dir(self):
        """DLEPS resolves a few data files relative to the working directory."""
        old = os.getcwd()
        os.chdir(self.code_dir)
        try:
            yield
        finally:
            os.chdir(old)

    def _load_dleps_class(self):
        """Import ``dleps_predictor`` from *this* checkout, in isolation.

        ``utils``, ``molecule_vae`` and friends are very generic module names.
        If another DLEPS checkout (or any other package) already put a module
        of that name into ``sys.modules``, a plain ``import`` would silently
        pick up the wrong one and score against a foreign model.  So the
        related names are purged before the import and restored afterwards --
        the freshly imported module objects stay alive through the references
        the loaded module holds.
        """
        if self._dleps_cls is not None:
            return self._dleps_cls

        saved = {}
        for name in _RELATED_MODULES:
            if name in sys.modules:
                saved[name] = sys.modules.pop(name)
        try:
            if str(self.code_dir) not in sys.path:
                sys.path.insert(0, str(self.code_dir))
            with self._in_code_dir():
                import dleps_predictor  # noqa: F401
            self._dleps_cls = sys.modules['dleps_predictor'].DLEPS
        finally:
            for name in _RELATED_MODULES:
                sys.modules.pop(name, None)
            sys.modules.update(saved)
        return self._dleps_cls

    def _make_predictor(self, up_path, down_path, reverse=None, setmean=None):
        """Instantiate the DLEPS class with the dense weights we resolved.

        The published ``dleps_predictor.py`` hard-codes
        ``../../data/denseweight.h5``; the subclass below points it at the
        file we actually found, so the model code stays untouched.

        ``reverse`` decides the direction of the score (see
        :func:`signature_direction`).  Note that the upstream ``DLEPS``
        forces ``reverse = False`` when no "down" signature is given, so a
        down-less signature is always scored in the "mimic" direction.

        ``setmean`` subtracts the set mean of the RNA levels of the current
        input; the AIDock client offers it for unbalanced output.
        """
        import h5py

        base = self._load_dleps_class()
        denseweight = self.denseweight
        reverse = self.reverse if reverse is None else bool(reverse)
        setmean = self.setmean if setmean is None else bool(setmean)

        class _LocalDLEPS(base):
            def _get_W(self):
                with h5py.File(denseweight, 'r') as hf:
                    return np.array(hf.get('W'))

        with self._in_code_dir():
            return _LocalDLEPS(reverse=reverse,
                               setmean=setmean,
                               up_name=str(up_path),
                               down_name=str(down_path) if down_path is not None else None,
                               onnx_dir=str(self.onnx_dir),
                               onnx_files=self.onnx_files)

    # -- public API --------------------------------------------------------

    def available_diseases(self):
        """Return the disease mapping that this installation can serve."""
        diseases = load_diseases()
        out = []
        for key, spec in diseases.items():
            up = self.data_dir / spec['up']
            down = self.data_dir / spec['down'] if spec.get('down') else None
            out.append({
                'disease': key,
                'label': spec.get('label', ''),
                'aliases': spec.get('aliases', []),
                'available': up.exists() and (down is None or down.exists()),
                'reverse': signature_direction(spec),
                'direction': 'reverse the disease state' if signature_direction(spec)
                             else 'mimic the phenotype (browning)',
                'up_file': spec['up'],
                'down_file': spec.get('down'),
                'note': spec.get('note', ''),
            })
        return out

    def score(self, smiles_list, disease_name=None, threshold=EFFECTIVE_THRESHOLD,
              n_repeats=1, up_file=None, down_file=None, reverse=None,
              setmean=None, gene_corr=0.0, abs_score=False,
              save_onehot=None, load_onehot=None):
        """Score every SMILES against ``disease_name``.

        Parameters
        ----------
        smiles_list :
            The candidate SMILES.
        disease_name :
            Disease key or alias, e.g. ``"Gout"`` / ``"痛风"``.  Ignored when
            ``up_file``/``down_file`` are given.
        threshold :
            Score above which a candidate counts as effective.
        n_repeats :
            The DLEPS head is stochastic (it samples ``eps ~ N(0, 0.01)`` in the
            VAE reparameterisation), so two runs on the same molecule differ.
            With ``n_repeats > 1`` the ONNX graph is executed that many times
            and the **expression outputs are averaged** before scoring, which
            removes most of that jitter. Use 3-5 for screening decisions.
        up_file, down_file :
            Optional custom signature files (one ENTREZGENE_ACC per line).
            They override the disease mapping, so any signature can be used.
        reverse :
            Optional direction override.  ``None`` (the default) uses the
            direction declared for the disease in ``data/diseases.json``; for a
            custom signature it means the upstream default, ``True``.
            ``True``  -- the signature states a disease state, a positive score
                         means the compound reverses it.
            ``False`` -- the signature states a therapeutic phenotype, a
                         positive score means the compound mimics it.
        setmean :
            Subtract the set mean of the RNA levels of the current input.
            The AIDock client offers this "for unbalanced output".
        gene_corr :
            Gene Correlation Cutoff, 0 - 0.8.  0 keeps every gene; 0.68376
            keeps only the well-predicted ones.  Same rule as the AIDock
            server.
        abs_score :
            Report the absolute value of the score, ignoring its sign.
        save_onehot, load_onehot :
            Cache / reuse the one-hot encoding of the input SMILES, which
            makes a repeated run over the same set much faster.

        Returns
        -------
        list of dict
            ``[{'smiles': str, 'cs_score': float, 'effective': bool,
            'expr_std': float}, ...]`` in the same order as the input.
        """
        if isinstance(smiles_list, str):
            smiles_list = [s.strip() for s in smiles_list.split(',') if s.strip()]
        smiles_list = [str(s).strip() for s in smiles_list if str(s).strip()]
        if not smiles_list:
            return []

        if up_file and down_file:
            # Resolve to absolute paths: DLEPS reads the signature files while
            # the working directory is code/DLEPS/, so a path relative to the
            # repository root would not be found.
            up_path = Path(up_file).expanduser().resolve()
            down_path = Path(down_file).expanduser().resolve()
            reverse = self.reverse if reverse is None else bool(reverse)
            missing = [str(p) for p in (up_path, down_path) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    'signature file(s) not found: %s' % ', '.join(missing))
        else:
            if not disease_name:
                raise ValueError('disease_name is required (or give up_file/down_file)')
            key, spec = resolve_disease(disease_name)
            up_path, down_path = signature_paths(key, spec, self.data_dir)
            reverse = signature_direction(spec) if reverse is None else bool(reverse)

        # Gene Correlation Cutoff: keep only the well-predicted genes.
        if gene_corr:
            if self._correlations is None:
                self._correlations = load_gene_correlations(self.data_dir)
            up_path = filter_signature_by_correlation(up_path, gene_corr,
                                                      self._correlations, self.data_dir)
            if down_path is not None:
                down_path = filter_signature_by_correlation(down_path, gene_corr,
                                                            self._correlations, self.data_dir)

        predictor = self._make_predictor(up_path, down_path, reverse=reverse,
                                         setmean=setmean)

        n_repeats = max(1, int(n_repeats))
        runs = []
        if n_repeats > 1:
            original = predictor._predict_onnx

            def _averaged(onehot):
                vals = [original(onehot) for _ in range(n_repeats)]
                runs.append(np.stack(vals))
                return np.mean(vals, axis=0)

            predictor._predict_onnx = _averaged

        scores = predictor.predict(smiles_list,
                                   save_onehot=save_onehot,
                                   load_onehot=load_onehot)

        expr_std = 0.0
        if runs:
            stack = runs[0]                      # (n_repeats, n_mol, 978)
            expr_std = float(np.mean(np.std(stack, axis=0)))

        results = []
        for smi, sc in zip(smiles_list, scores):
            sc = float(sc)
            if abs_score:
                sc = abs(sc)
            results.append({
                'smiles': smi,
                'cs_score': sc,
                'effective': bool(sc > threshold),
                'expr_std': expr_std,
            })
        return results


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Score candidate molecules with DLEPS (ONNX build).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='examples:\n'
               '  python scripts/dleps_score.py --list-diseases\n'
               '  python scripts/dleps_score.py --disease "Gout" --smiles "CCO,CC(=O)Oc1ccccc1C(=O)O"\n'
               '  python scripts/dleps_score.py --disease Aging --smiles-file mols.txt --json out.json\n'
               '  python scripts/dleps_score.py --up-file up.txt --down-file down.txt --smiles "CCO"\n')
    ap.add_argument('--disease', help='disease name (see --list-diseases)')
    ap.add_argument('--up-file', dest='up_file',
                    help='custom up-regulated signature file (one ENTREZGENE_ACC per line); '
                         'overrides the disease mapping')
    ap.add_argument('--down-file', dest='down_file',
                    help='custom down-regulated signature file; overrides the disease mapping')
    ap.add_argument('--smiles', help='comma separated SMILES')
    ap.add_argument('--smiles-file', help='file with one SMILES per line')
    ap.add_argument('--json', help='write the result to this JSON file')
    ap.add_argument('--threshold', type=float, default=EFFECTIVE_THRESHOLD,
                    help='score above which a candidate counts as effective (default: %(default)s)')
    ap.add_argument('--repeats', type=int, default=1,
                    help='run the ONNX graph N times and average the expression outputs, '
                         'which removes the jitter of the model\'s stochastic sampling '
                         '(default: %(default)s; use 3-5 for screening decisions)')
    ap.add_argument('--list-diseases', action='store_true',
                    help='list the supported diseases and exit')
    ap.add_argument('--setmean', action='store_true',
                    help='subtract the set mean of the RNA levels of the current input '
                         '(for unbalanced output)')
    ap.add_argument('--abs', dest='abs_score', action='store_true',
                    help='report the absolute value of the ES score, ignoring its sign')
    ap.add_argument('--gene-corr', type=float, default=0.0,
                    help='Gene Correlation Cutoff, 0 - 0.8. 0 keeps every gene; '
                         '0.68376 keeps only the well-predicted ones (default: %(default)s)')
    ap.add_argument('--save-onehot',
                    help='save the one-hot encoding of the input SMILES to this path '
                         '(reused automatically on the next run)')
    ap.add_argument('--load-onehot',
                    help='reuse a previously saved one-hot encoding instead of recomputing it')
    direction = ap.add_mutually_exclusive_group()
    direction.add_argument('--reverse', dest='direction', action='store_const',
                           const=True,
                           help='the signature states a disease state: a positive score '
                                'means the compound reverses it (overrides the disease entry)')
    direction.add_argument('--no-reverse', dest='direction', action='store_const',
                           const=False,
                           help='the signature states a therapeutic phenotype: a positive '
                                'score means the compound mimics it (overrides the disease entry)')
    ap.set_defaults(direction=None)
    args = ap.parse_args(argv)

    if args.list_diseases:
        for d in DlepsScorer().available_diseases():
            flag = 'ok ' if d['available'] else 'MISS'
            print('%-5s %-38s %-46s %-9s %s / %s'
                  % (flag, d['disease'], d['label'],
                     'reverse' if d['reverse'] else 'mimic',
                     d['up_file'], d['down_file'] or '-'))
        return 0

    if not args.disease and not (args.up_file and args.down_file):
        ap.error('--disease is required (or --up-file together with --down-file; '
                 'or use --list-diseases)')
    if not args.smiles and not args.smiles_file:
        ap.error('--smiles or --smiles-file is required')

    if args.smiles_file:
        with open(args.smiles_file, encoding='utf-8') as f:
            smiles = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    else:
        smiles = [s.strip() for s in args.smiles.split(',') if s.strip()]

    scorer = DlepsScorer()
    try:
        results = scorer.score(smiles, args.disease, threshold=args.threshold,
                               n_repeats=args.repeats,
                               up_file=args.up_file, down_file=args.down_file,
                               reverse=args.direction,
                               setmean=args.setmean, gene_corr=args.gene_corr,
                               abs_score=args.abs_score,
                               save_onehot=args.save_onehot,
                               load_onehot=args.load_onehot)
    except KeyError as e:
        ap.error(str(e.args[0]) if e.args else 'unknown disease')
    except FileNotFoundError as e:
        ap.error(str(e))
    except ValueError as e:
        ap.error(str(e))

    print('%-52s %10s  %s' % ('SMILES', 'cs_score', 'verdict'))
    for r in results:
        print('%-52s %10.4f  %s'
              % (r['smiles'][:52], r['cs_score'],
                 'effective' if r['effective'] else ''))
    hits = sum(1 for r in results if r['effective'])
    print('\n%d/%d candidate(s) above %.2f' % (hits, len(results), args.threshold))
    if args.repeats > 1:
        print('expression jitter over %d run(s): std = %.4f' % (args.repeats,
                                                                results[0]['expr_std']))

    if args.json:
        payload = {
            'disease': args.disease,
            'threshold': args.threshold,
            'n_candidates': len(results),
            'n_effective': hits,
            'pred_scores': [{'smiles': r['smiles'], 'cs_score': r['cs_score']} for r in results],
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print('wrote %s' % args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
