#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Train GE uncertainty model, then initialize GSR from its saved weights.
# Use: bash train.sh --dataset config/GE_multi_angle.yml for both angles.
python train.py -uncertainty --config config/GE_train_u.json --dataset config/GE.yml "$@"
python train.py --config config/GE_train.json --dataset config/GE.yml "$@"
