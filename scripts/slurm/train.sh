#!/bin/bash
# Example Slurm job: train VideoGAN on the golf dataset with one A100.
#
# Written for UF HiPerGator; change the partition, GPU type and limits for other clusters.
# Submit from the repository root:
#
#   VENV=/path/to/venv GOLF_ROOT=/path/to/golf sbatch scripts/slurm/train.sh
#
# Submitting the same command again continues from the run's last checkpoint, which is how a
# run longer than the time limit is finished. To get email updates, add
# "#SBATCH --mail-type=END,FAIL" and "#SBATCH --mail-user=<your address>" below.
#
#SBATCH --job-name=videogan-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=100:00:00

set -eo pipefail

: "${VENV:?set VENV to the virtual environment that has videogan installed}"
: "${GOLF_ROOT:?set GOLF_ROOT to the folder with golf.txt and frames-stable-many/}"
RUN_DIR=${RUN_DIR:-runs/golf}

# shellcheck source=/dev/null
source "$VENV/bin/activate"
echo "host $(hostname -s), started $(date)"

resume=()
if [[ -f "$RUN_DIR/checkpoints/last.pt" ]]; then
    resume=(--resume "$RUN_DIR/checkpoints/last.pt")
fi

python -m videogan.train \
    --data-root "$GOLF_ROOT" \
    --out-dir "$RUN_DIR" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    "${resume[@]}"
