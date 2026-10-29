#!/usr/bin/env bash
# train_db.sh settings, with rendered-depth geometry adaptation enabled.
# Default: unfinished scenes under ~/AD/dataset/db, sequentially.
# A final model plus finite ours_40000 metrics marks a scene complete.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/AD/dataset/db}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nvme1n1/zhutianchi/miniconda/envs/octree-gs/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/db}"
EXP_NAME="${EXP_NAME:-geometry_adaptive}"
PORT="${PORT:-$((10000 + $$ % 40000))}"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_$$"
dry_run=false
all_scenes=false
scenes=()

usage() {
    cat <<'EOF'
Usage: bash train_db_geometry.sh [--dry-run] [--all | SCENE ...]
  No scenes: consider every scene under DATA_ROOT; skip completed scenes.
  --all:     same as the default; completed scenes are still skipped.
  SCENE ...: consider only these scenes; completed scenes are still skipped.
  --dry-run: show skipped scenes and pending commands without training.
Examples:
  bash train_db_geometry.sh
  bash train_db_geometry.sh drjohnson
  bash train_db_geometry.sh --all --dry-run
Environment overrides: DATA_ROOT, PYTHON_BIN, OUTPUT_ROOT, EXP_NAME, PORT.
GPU selection uses train.py's existing automatic selection (--gpu -1).
Each scene runs 40000 iterations, then renders and evaluates its test split.
Completion is checked in OUTPUT_ROOT/<scene>/EXP_NAME/* using the final model
and finite SSIM/PSNR/LPIPS in results.json under ours_40000.
Incomplete runs restart training in a new directory; this is not checkpoint resume.
Geometry uses rendered depth only; no external depth files are needed.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --all) all_scenes=true ;;
        -h|--help) usage; exit 0 ;;
        -*) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
        *) scenes+=("$arg") ;;
    esac
done

cd -- "$PROJECT_DIR"
[[ -d "$DATA_ROOT" ]] || { printf 'Missing dataset: %s\n' "$DATA_ROOT" >&2; exit 1; }
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
command -v "$PYTHON_BIN" >/dev/null || { printf 'Python not found: %s\n' "$PYTHON_BIN" >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && ((10#$PORT >= 1024 && 10#$PORT <= 65535)) || { printf 'Invalid PORT\n' >&2; exit 2; }
if "$all_scenes"; then
    ((${#scenes[@]} == 0)) || { printf 'Use --all or scene names, not both.\n' >&2; exit 2; }
fi
if ((${#scenes[@]} == 0)); then
    for path in "$DATA_ROOT"/*; do
        [[ -d "$path" ]] && scenes+=("${path##*/}")
    done
fi
((${#scenes[@]} > 0)) || { printf 'No scenes found.\n' >&2; exit 1; }

# Check every run of the same experiment; a directory alone is not completion.
completed_run() {
    "$PYTHON_BIN" - "$OUTPUT_ROOT/$1/$EXP_NAME" <<'PY'
import json
import math
import sys
from pathlib import Path

experiment = Path(sys.argv[1])
if not experiment.exists():
    sys.exit(0)
required = ('point_cloud.ply', 'point_cloud.geometry.pt',
            'opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt')
for run in sorted(experiment.iterdir(), reverse=True):
    if not run.is_dir():
        continue
    try:
        model = run / 'point_cloud' / 'iteration_40000'
        if not all((model / name).is_file() and (model / name).stat().st_size > 0
                   for name in required):
            continue
        with (run / 'results.json').open() as stream:
            results = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        continue
    metrics = results.get('ours_40000') if isinstance(results, dict) else None
    if not isinstance(metrics, dict):
        continue
    if all(type(metrics.get(key)) in (int, float) and math.isfinite(metrics[key])
           for key in ('SSIM', 'PSNR', 'LPIPS')):
        print(run)
        break
PY
}

pending=()
skipped=0
for scene in "${scenes[@]}"; do
    [[ "$scene" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { printf 'Invalid scene name: %s\n' "$scene" >&2; exit 2; }
    finished="$(completed_run "$scene")"
    if [[ -n "$finished" ]]; then
        printf 'SKIP completed: %s (%s)\n' "$scene" "$finished"
        skipped=$((skipped + 1))
    else
        pending+=("$scene")
    fi
done
printf 'Selected: %s | Completed: %s | Pending: %s\n' "${#scenes[@]}" "$skipped" "${#pending[@]}"
if ((${#pending[@]} == 0)); then
    printf 'Nothing to train: all selected scenes have final evaluation results.\n'
    exit 0
fi

# Validate all pending scenes before starting the first training job.
for scene in "${pending[@]}"; do
    for required in images sparse; do
        [[ -d "$DATA_ROOT/$scene/$required" ]] || { printf 'Missing directory: %s\n' "$DATA_ROOT/$scene/$required" >&2; exit 1; }
    done
done

for scene in "${pending[@]}"; do
    model_path="$OUTPUT_ROOT/$scene/$EXP_NAME/$RUN_ID"
    cmd=("$PYTHON_BIN" -u train.py
        --eval -s "$DATA_ROOT/$scene" -m "$model_path"
        --gpu -1 --port "$PORT" --iterations 40000
        -r -1 --ratio 1 --appearance_dim 0
        --fork 2 --base_layer 12 --visible_threshold 0.9
        --dist2level round --update_ratio 0.2
        --progressive --levels -1 --init_level -1 --dist_ratio 0.999
        --extra_ratio 0.25 --extra_up 0.01
        --geometry_adaptive --geometry_warmup 1000
        --geometry_interval 500 --geometry_views 4
        --geometry_max_anchors 16384 --geometry_chunk 256
    )
    printf '\nScene: %s\n' "$scene"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if "$dry_run"; then
        continue
    fi
    mkdir -p -- "$model_path"
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\ncd -- %q\n' "$PROJECT_DIR"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$model_path/command.sh"
    "${cmd[@]}" 2>&1 | tee "$model_path/console.log"
done
