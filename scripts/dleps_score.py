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
    """Resolve the up/down signature files of a disease."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    up = data_dir / spec['up']
    down = data_dir / spec['down']
    missing = [str(p) for p in (up, down) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Signature file(s) missing for %s: %s' % (disease_key, ', '.join(missing)))
    return up, down


# --------------------------------------------------------------------------
# scorer
# --------------------------------------------------------------------------

class DlepsScorer(object):
    """Score candidate molecules against a disease signature with DLEPS_onnx."""

    def __init__(self, repo_root=None, onnx_dir=None, onnx_files=None,
                 denseweight=None, reverse=True):
        self.repo_root = Path(repo_root).resolve() if repo_root else REPO_ROOT
        self.code_dir = self.repo_root / 'code' / 'DLEPS'
        self.data_dir = self.repo_root / 'data'
        self.onnx_dir = Path(onnx_dir) if onnx_dir else self.data_dir / 'onnx'
        self.onnx_files = onnx_files
        self.denseweight = resolve_denseweight(denseweight)
        self.reverse = reverse
        self._dleps_cls = None

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

    def _make_predictor(self, up_path, down_path):
        """Instantiate the DLEPS class with the dense weights we resolved.

        The published ``dleps_predictor.py`` hard-codes
        ``../../data/denseweight.h5``; the subclass below points it at the
        file we actually found, so the model code stays untouched.
        """
        import h5py

        base = self._load_dleps_class()
        denseweight = self.denseweight

        class _LocalDLEPS(base):
            def _get_W(self):
                with h5py.File(denseweight, 'r') as hf:
                    return np.array(hf.get('W'))

        with self._in_code_dir():
            return _LocalDLEPS(reverse=self.reverse,
                               up_name=str(up_path),
                               down_name=str(down_path),
                               onnx_dir=str(self.onnx_dir),
                               onnx_files=self.onnx_files)

    # -- public API --------------------------------------------------------

    def available_diseases(self):
        """Return the disease mapping that this installation can serve."""
        diseases = load_diseases()
        out = []
        for key, spec in diseases.items():
            up = self.data_dir / spec['up']
            down = self.data_dir / spec['down']
            out.append({
                'disease': key,
                'label': spec.get('label', ''),
                'aliases': spec.get('aliases', []),
                'available': up.exists() and down.exists(),
                'n_up': len(spec['up']),
                'up_file': spec['up'],
                'down_file': spec['down'],
            })
        return out

    def score(self, smiles_list, disease_name=None, threshold=EFFECTIVE_THRESHOLD,
              n_repeats=1, up_file=None, down_file=None):
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
            up_path, down_path = Path(up_file), Path(down_file)
            missing = [str(p) for p in (up_path, down_path) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    'signature file(s) not found: %s' % ', '.join(missing))
        else:
            if not disease_name:
                raise ValueError('disease_name is required (or give up_file/down_file)')
            key, spec = resolve_disease(disease_name)
            up_path, down_path = signature_paths(key, spec, self.data_dir)
        predictor = self._make_predictor(up_path, down_path)

        n_repeats = max(1, int(n_repeats))
        runs = []
        if n_repeats > 1:
            original = predictor._predict_onnx

            def _averaged(onehot):
                vals = [original(onehot) for _ in range(n_repeats)]
                runs.append(np.stack(vals))
                return np.mean(vals, axis=0)

            predictor._predict_onnx = _averaged

        scores = predictor.predict(smiles_list)

        expr_std = 0.0
        if runs:
            stack = runs[0]                      # (n_repeats, n_mol, 978)
            expr_std = float(np.mean(np.std(stack, axis=0)))

        results = []
        for smi, sc in zip(smiles_list, scores):
            sc = float(sc)
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
               '  python scripts/dleps_score.py --disease Aging --smiles-file mols.txt --json out.json\n')
    ap.add_argument('--disease', help='disease name (see --list-diseases)')
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
    args = ap.parse_args(argv)

    if args.list_diseases:
        for d in DlepsScorer().available_diseases():
            flag = 'ok ' if d['available'] else 'MISS'
            print('%-5s %-38s %-46s %s / %s'
                  % (flag, d['disease'], d['label'], d['up_file'], d['down_file']))
        return 0

    if not args.disease:
        ap.error('--disease is required (or use --list-diseases)')
    if not args.smiles and not args.smiles_file:
        ap.error('--smiles or --smiles-file is required')

    if args.smiles_file:
        with open(args.smiles_file, encoding='utf-8') as f:
            smiles = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    else:
        smiles = [s.strip() for s in args.smiles.split(',') if s.strip()]

    scorer = DlepsScorer()
    results = scorer.score(smiles, args.disease, threshold=args.threshold,
                           n_repeats=args.repeats)

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
