#!/usr/bin/env python
########################################################
# All rights reserved.
# Author: XIE Zhengwei @ Beijing Gigarmaceuticals Tech Co., Ltd
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

"""Local MCP server exposing DLEPS (ONNX build) as ``calculate_dleps_score``.

The drug-discovery skill suite normally calls ``calculate_dleps_score`` on the
remote SCP Hub (``https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool``),
which needs an ``SCP_HUB_API_KEY`` and uploads the input files to somebody
else's machine.  This server exposes **the same tool, same name, same
argument names and the same return shape**, but runs the DLEPS_onnx graph
locally:

* no API key, no network, no file upload;
* no TensorFlow / Keras -- ONNX Runtime only;
* the structures never leave the machine.

Start it over stdio::

    python scripts/dleps_mcp_server.py

and register it as an MCP server, e.g.::

    mcp create dleps-onnx --transport stdio \\
        --command python3 --args /abs/path/to/DLEPS_onnx/scripts/dleps_mcp_server.py
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dleps_score import (EFFECTIVE_THRESHOLD, DlepsScorer,  # noqa: E402
                          load_diseases, resolve_disease)

SERVER_NAME = 'dleps-onnx'

TOOL_DESCRIPTION = (
    'Enter a list of candidate small molecules. Based on the input disease name, '
    'identify upregulated and downregulated genes associated with the disease state, '
    'and predict a reversal score for each small molecule. Generally, a score above '
    '0.2 indicates effectiveness, with higher scores being better. Runs the DLEPS '
    'ONNX graph locally -- no upload, no API key, no TensorFlow.'
)

INPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'smiles_list': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': 'List of input SMILES strings, '
                           '(e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])',
        },
        'disease_name': {
            'type': 'string',
            'description': 'Supported diseases, e.g., "Aging", "Gout", "Pulmonary fibrosis", '
                           '"Non-alcoholic fatty liver disease", "Obesity".',
        },
        'up_file': {
            'type': 'string',
            'description': 'Optional path to a custom up-regulated gene signature file '
                           '(one ENTREZGENE_ACC per line). Overrides the disease mapping.',
        },
        'down_file': {
            'type': 'string',
            'description': 'Optional path to a custom down-regulated gene signature file. '
                           'Overrides the disease mapping.',
        },
        'n_repeats': {
            'type': 'integer',
            'minimum': 1,
            'default': 1,
            'description': 'Run the ONNX graph this many times and average the expression '
                           'outputs. The DLEPS head samples eps ~ N(0, 0.01), so a single run '
                           'is jittery; 3-5 is recommended for screening decisions. '
                           '(default: 1)',
        },
    },
    'required': ['smiles_list', 'disease_name'],
}

_scorer = None


def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = DlepsScorer()
    return _scorer


def _disease_catalog():
    out = []
    for d in get_scorer().available_diseases():
        out.append('%s (%s)%s' % (d['disease'], d['label'],
                                  '' if d['available'] else '  [signature file missing]'))
    return '; '.join(out)


def calculate_dleps_score(smiles_list, disease_name, up_file=None, down_file=None,
                          n_repeats=1):
    """The tool body -- kept importable so it can be unit-tested directly."""
    scorer = get_scorer()
    n_repeats = max(1, int(n_repeats or 1))
    try:
        if not up_file or not down_file:
            # Surface an unknown / not-installed disease before doing any work.
            resolve_disease(disease_name)
    except KeyError as e:
        return {'status': 'error',
                'msg': str(e.args[0]) + ' | available: ' + _disease_catalog(),
                'pred_scores': []}

    try:
        results = scorer.score(smiles_list, disease_name,
                               up_file=up_file, down_file=down_file,
                               n_repeats=n_repeats)
    except Exception as e:  # noqa: BLE001 -- report, never crash the server
        return {'status': 'error', 'msg': '%s: %s' % (type(e).__name__, e), 'pred_scores': []}

    n_effective = sum(1 for r in results if r['effective'])
    target = disease_name if not (up_file and down_file) else '%s + %s' % (up_file, down_file)
    msg = ('%d candidate(s) scored against %s; %d above %.2f'
           % (len(results), target, n_effective, EFFECTIVE_THRESHOLD))
    if n_repeats > 1 and results:
        msg += ' (expression jitter std %.4f over %d run(s))' % (results[0]['expr_std'], n_repeats)
    return {
        'status': 'success',
        'msg': msg,
        'pred_scores': [{'smiles': r['smiles'], 'cs_score': r['cs_score']} for r in results],
    }


def build_server():
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools():
        return [Tool(name='calculate_dleps_score',
                     description=TOOL_DESCRIPTION,
                     inputSchema=INPUT_SCHEMA)]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name != 'calculate_dleps_score':
            payload = {'status': 'error', 'msg': 'unknown tool %r' % name, 'pred_scores': []}
        else:
            arguments = arguments or {}
            payload = calculate_dleps_score(
                arguments.get('smiles_list') or [],
                arguments.get('disease_name') or '',
                arguments.get('up_file'),
                arguments.get('down_file'),
                arguments.get('n_repeats', 1))
        return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False))]

    return server


async def run_stdio():
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(argv=None):
    ap = argparse.ArgumentParser(description='MCP stdio server for DLEPS (ONNX build).')
    ap.add_argument('--check', action='store_true',
                    help='self-check: print the disease catalog and exit')
    args = ap.parse_args(argv)

    if args.check:
        for d in get_scorer().available_diseases():
            print('%-5s %-42s %s' % ('ok' if d['available'] else 'MISS',
                                     d['disease'], d['label']))
        return 0

    try:
        asyncio.run(run_stdio())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
