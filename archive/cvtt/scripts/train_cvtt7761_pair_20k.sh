#!/usr/bin/env bash
# Sequential on one 5090: never continue to the second run after a failed first run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for method in rvld gavd; do
    run="cvtt7761_${method}_20000_20261001"
    bash scripts/train_cvtt7761_students_5090.sh "$method" 20000 64 "$run" \
        > "logs/$run.console.log" 2>&1
    echo "$method completed successfully"
done
