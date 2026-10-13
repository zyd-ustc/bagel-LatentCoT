#!/usr/bin/env bash
# Run on the local Mac. An optional argument selects a specific remote run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOST=${REVIEW_HOST:-modelarts-job}
REMOTE_RUN=${1:-}
if [[ -z "$REMOTE_RUN" ]]; then
  REMOTE_RUN=$(ssh "$HOST" 'python3 -' <<'PY'
from pathlib import Path
runs = [p for p in Path('/root/outputs').glob('memory16_t2i_R234_early1020_*') if (p/'index.html').is_file()]
if not runs:
    raise SystemExit('No completed comparison page found')
print(max(runs, key=lambda p: (p/'index.html').stat().st_mtime))
PY
  )
fi
LOCAL_RUN="${ARTIFACTS_DIR:-$ROOT/../artifacts}/$(basename "$REMOTE_RUN")"
mkdir -p "$LOCAL_RUN"
scp -r "${HOST}:${REMOTE_RUN}/." "$LOCAL_RUN/"
[[ -f "$LOCAL_RUN/index.html" ]] || { echo 'The remote run has no index.html' >&2; exit 1; }
open "$LOCAL_RUN/index.html"
