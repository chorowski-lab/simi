import sys

from fit import score_cpc_quantizations


gt = sys.argv[1]
quantized = sys.argv[2]
shift = int(sys.argv[3])

acc = score_cpc_quantizations(gt, quantized, shift=shift)
print(f'Framewise accuracy: {acc:.2f} %')
