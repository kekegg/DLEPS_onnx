#!/usr/bin/env python
########################################################
# All rights reserved.
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

"""Check that an exported ONNX graph carries exactly the weights of its Keras
checkpoint.

The DLEPS models were originally trained with Keras/TensorFlow and are shipped
as ``.h5`` checkpoints.  For the ONNX build the same checkpoints were converted
with keras2onnx, which stores every weight as an ONNX initializer.

This script compares the two weight sets tensor by tensor:

  * the Keras ``Conv1D`` kernels are stored as ``(kernel_size, in, filters)``
    while ONNX wants ``(filters, in, kernel_size)`` -- those are transposed;
  * every other tensor must match bit for bit.

Usage
-----
    python scripts/verify_onnx_weights.py \
        --h5   data/onnx/dleps_0b_0.73.h5 \
        --onnx data/onnx/dleps_0b_0.73.h5.onnx

Exit code is 0 when every tensor matches, 1 otherwise.
"""

import argparse
import sys

import numpy as np

# Keras layer name -> (ONNX initializer name, axes permutation or None)
# The Conv1D kernels need a transpose, everything else is stored as-is.
WEIGHT_MAP = [
    ('model_weights/conv_1/conv_1_2/kernel:0', 'conv_1/kernel:0', (2, 1, 0)),
    ('model_weights/conv_1/conv_1_2/bias:0', 'conv_1/bias:0', None),
    ('model_weights/conv_2/conv_2_2/kernel:0', 'conv_2/kernel:0', (2, 1, 0)),
    ('model_weights/conv_2/conv_2_2/bias:0', 'conv_2/bias:0', None),
    ('model_weights/conv_3/conv_3_2/kernel:0', 'conv_3/kernel:0', (2, 1, 0)),
    ('model_weights/conv_3/conv_3_2/bias:0', 'conv_3/bias:0', None),
    ('model_weights/dense_1/dense_1_2/kernel:0', 'dense_1/kernel:0', None),
    ('model_weights/dense_1/dense_1_2/bias:0', 'dense_1/bias:0', None),
    ('model_weights/z_mean/z_mean_2/kernel:0', 'z_mean/kernel:0', None),
    ('model_weights/z_mean/z_mean_2/bias:0', 'z_mean/bias:0', None),
    ('model_weights/z_log_var/z_log_var_2/kernel:0', 'z_log_var/kernel:0', None),
    ('model_weights/z_log_var/z_log_var_2/bias:0', 'z_log_var/bias:0', None),
    ('model_weights/dense_3/dense_3/kernel:0', 'dense_3/kernel:0', None),
    ('model_weights/dense_3/dense_3/bias:0', 'dense_3/bias:0', None),
    ('model_weights/dense_4/dense_4/kernel:0', 'dense_4/kernel:0', None),
    ('model_weights/dense_4/dense_4/bias:0', 'dense_4/bias:0', None),
    ('model_weights/dense_5/dense_5/kernel:0', 'dense_5/kernel:0', None),
    ('model_weights/dense_5/dense_5/bias:0', 'dense_5/bias:0', None),
    ('model_weights/dense_6/dense_6/kernel:0', 'dense_6/kernel:0', None),
    ('model_weights/dense_6/dense_6/bias:0', 'dense_6/bias:0', None),
    ('model_weights/dense_7/dense_7/kernel:0', 'dense_7/kernel:0', None),
    ('model_weights/dense_7/dense_7/bias:0', 'dense_7/bias:0', None),
]


def load_onnx_initializers(path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(path)
    return {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--h5', required=True, help='Keras checkpoint (.h5)')
    parser.add_argument('--onnx', required=True, help='Exported ONNX graph (.onnx)')
    args = parser.parse_args()

    import h5py

    init = load_onnx_initializers(args.onnx)
    n_ok = 0
    n_bad = 0
    print('%-46s %-24s %-16s %-12s %s' % ('Keras tensor', 'ONNX initializer', 'shape', 'max|diff|', 'identical'))
    print('-' * 118)

    with h5py.File(args.h5, 'r') as f:
        for h5_path, onnx_name, perm in WEIGHT_MAP:
            if h5_path not in f:
                print('%-46s MISSING in %s' % (h5_path.split('/')[-2], args.h5))
                n_bad += 1
                continue
            a = np.array(f[h5_path])
            if perm is not None:
                a = np.transpose(a, perm)
            b = init.get(onnx_name)
            if b is None:
                print('%-46s MISSING in %s' % (onnx_name, args.onnx))
                n_bad += 1
                continue
            if a.shape != b.shape:
                print('%-46s %-24s SHAPE %s vs %s' % (h5_path.split('/')[-2], onnx_name, a.shape, b.shape))
                n_bad += 1
                continue
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            same = bool(np.array_equal(a, b))
            n_ok += 1 if same else 0
            n_bad += 0 if same else 1
            print('%-46s %-24s %-16s %-12.3e %s'
                  % (h5_path.split('/')[-2], onnx_name, str(a.shape), diff, same))

    print('-' * 118)
    print('%d/%d tensors are bit-for-bit identical' % (n_ok, n_ok + n_bad))
    if n_bad:
        print('FAILED: %d tensor(s) differ' % n_bad)
        return 1
    print('OK: the ONNX graph carries exactly the weights of the Keras checkpoint.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
