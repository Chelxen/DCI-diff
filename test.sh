#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Use: bash test.sh --checkpoint experiments/<GE_run>/checkpoint/best_gen.pth --angle 90
# Add --use_dc to use the corresponding GE geometry for data consistency.
python test.py --config config/GE_test.json --dataset config/GE.yml "$@"
