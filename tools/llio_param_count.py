"""Print "<params> <fp32_mib>" for an LLIO-Net variant.

Usage: python tools/llio_param_count.py [512|256|128]

Used by train_llio128.sh so the parameter count of the variant being trained is
echoed before training starts, rather than scrolling past inside the trainer's
own log line.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))

from model_llio1d import LLIO1D  # noqa: E402

feature_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 512
n = LLIO1D(feature_dim=feature_dim).get_num_params()
print("{:,} {:.2f}".format(n, 4 * n / 1024 ** 2))
