#!/usr/bin/env bash
# Train one scene with rendered-depth geometry adaptation.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/AD/dataset/db}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nvme1n1/zhutianchi/miniconda/envs/octree-gs/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/db}"
EXP_NAME="${EXP_NAME:-geometry_adaptive}"
PORT="${PORT:-$((10000 + $$ % 40000))}"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_$$"
dry_run=false
scenes=()

usage() {
    cat <<'EOF'
Usage: bash single_train_geo.sh [--dry-run] SCENE
Examples:
  bash single_train_geo.sh drjohnson
  bash single_train_geo.sh boats --dry-run
  DATA_ROOT=/path/to/dataset bash single_train_geo.sh playroom

SCENE is a directory name under DATA_ROOT (default: ~/AD/dataset/db).
Runs 40000 training iterations, then renders and evaluates the test split.
Every invocation starts fresh in a new output directory, even for a completed
scene. Existing results are preserved. This script does not resume checkpoints.
Geometry uses rendered depth. GPU selection is automatic (--gpu -1).
Environment overrides: DATA_ROOT, PYTHON_BIN, OUTPUT_ROOT, EXP_NAME, PORT.
--dry-run validates paths and prints the command without starting training.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        -*) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
        *) scenes+=("$arg") ;;
    esac
done
if ((${#scenes[@]} != 1)); then
    printf 'Provide exactly one scene name.\n' >&2
    usage >&2
    exit 2
fi
scene="${scenes[0]}"
[[ "$scene" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { printf 'Invalid scene name: %s\n' "$scene" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && ((10#$PORT >= 1024 && 10#$PORT <= 65535)) || { printf 'Invalid PORT\n' >&2; exit 2; }

cd -- "$PROJECT_DIR"
[[ -f train.py ]] || { printf 'Missing train.py in %s\n' "$PROJECT_DIR" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { printf 'Python not found: %s\n' "$PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { printf 'Missing dataset: %s\n' "$DATA_ROOT" >&2; exit 1; }
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
for required in images sparse; do
    [[ -d "$DATA_ROOT/$scene/$required" ]] || { printf 'Missing directory: %s\n' "$DATA_ROOT/$scene/$required" >&2; exit 1; }
done

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
printf 'Scene: %s\nOutput: %s\n' "$scene" "$model_path"
printf '%q ' "${cmd[@]}"
printf '\n'
if "$dry_run"; then
    exit 0
fi

mkdir -p -- "$OUTPUT_ROOT/$scene/$EXP_NAME"
mkdir -- "$model_path"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd -- %q\n' "$PROJECT_DIR"
    printf '%q ' "${cmd[@]}"
    printf '\n'
} > "$model_path/command.sh"
"${cmd[@]}" 2>&1 | tee "$model_path/console.log"
