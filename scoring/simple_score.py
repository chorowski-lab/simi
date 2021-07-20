import sys
from pathlib import Path

from fit import (score_cpc_quantizations,
                 score_cpc_quantizations_matching_sentpieces_with_phones)

gt = sys.argv[1]
quantized = sys.argv[2]
shift = int(sys.argv[3])

quant_dir = (Path(quantized) / '..').resolve().name
acc = score_cpc_quantizations_matching_sentpieces_with_phones(
    gt, quantized, shift=shift, per_ignore_short_blocks=0, print_sample=1)

print(f'PER: {acc:.2f} % ({quant_dir})')
