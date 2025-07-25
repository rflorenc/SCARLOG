#!/usr/bin/env bash
# OpenReasoning-Nemotron test script for AssureMOSS dataset
# Usage: bash benchmark-test-openreason.sh [nlines]

set -euo pipefail
IFS=$'\n\t'

# Configuration
NLINES=${1:-1000}

HF_TOKEN_FILE="../hf_token.txt"
if [[ ! -f $HF_TOKEN_FILE ]]; then
  echo "Hugging Face token file not found: $HF_TOKEN_FILE"
  exit 1
fi
export HUGGINGFACE_HUB_TOKEN="$(<"$HF_TOKEN_FILE")"

LOGDIR="./bench_logs/openreason"
mkdir -p "$LOGDIR"

timestamp() { date '+%d_%H%M'; }

# AssureMOSS datasets
AMK_BENIGN="../datasets/intermediate_tasks/task1_systems/AssureMOSS Kubernetes Run-time Monitoring Dataset_1_all/elastic_may2021_benign_data.csv"
AMK_MALICIOUS="../datasets/intermediate_tasks/task1_systems/AssureMOSS Kubernetes Run-time Monitoring Dataset_1_all/elastic_may2021_malicious_data.csv"

run_and_log() {
  local tag="$1"; shift
  local log="$LOGDIR/${tag}__$(timestamp).log"
  echo "Running $tag -> $log"
  if "$@" &> "$log"; then
    echo "$tag finished OK"
  else
    echo "$tag failed (see $log)"
    return 1
  fi
}

echo "Running SCARLOG OpenReasoning-Nemotron test - AssureMOSS dataset (nlines=$NLINES)"
echo "Using test_openreasoning config (2 models: 1.5B and 7B)"

TEST_CONFIG="test_openreasoning"

echo "=== Test 1: Labeled AssureMOSS with OpenReasoning-Nemotron models ==="
echo "Using malicious dataset (contains both benign and malicious samples)"
run_and_log "openreason_assuremoss_${NLINES}" \
  python ../run_benchmark.py --test-config "$TEST_CONFIG" "$AMK_MALICIOUS" "$NLINES" assuremoss

echo ""
echo "OpenReasoning-Nemotron test completed successfully!"
echo "Results saved in output_results/"
echo "Logs available in $LOGDIR/"
echo ""
echo "To analyze results, run:"
echo "python ../tools/analyze_reasoning_results.py -d ../output_results --auto-discover"