#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/multinode_2x4.json}"
TENSOR_ELEMENTS="${2:-1000000}"

python -m precisionflow_lab connect --manifest "$CONFIG" --anonymize-hostnames
python -m precisionflow_lab inspect "$CONFIG" --tensor-elements "$TENSOR_ELEMENTS"
