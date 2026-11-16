#!/bin/bash
# Example Slurm job: fine-tune the GAN's discriminator on UCF-101 with one A100.
#
# Written for UF HiPerGator; change the partition, GPU type and limits for other clusters.
# Submit from the repository root with the split folder as the argument:
#
#   export VENV=/path/to/venv UCF101_ROOT=/path/to/UCF-101 GAN_CHECKPOINT=runs/golf/checkpoints/last.pt
#   sbatch scripts/slurm/finetune.sh splits/kayaking-yoyo    # Kayaking vs. YoYo
#   sbatch scripts/slurm/finetune.sh splits/ucf101           # all 101 classes
#
# Results go to runs/<split folder name> unless RUN_DIR is set. Leave GAN_CHECKPOINT unset to
# train the same network from random weights, for comparison.
#
#SBATCH --job-name=videogan-finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

set -eo pipefail

: "${VENV:?set VENV to the virtual environment that has videogan installed}"
: "${UCF101_ROOT:?set UCF101_ROOT to the extracted UCF-101 folder}"
split_dir=${1:-splits/kayaking-yoyo}
RUN_DIR=${RUN_DIR:-runs/$(basename "$split_dir")}

# shellcheck source=/dev/null
source "$VENV/bin/activate"
echo "host $(hostname -s), started $(date)"

init=()
if [[ -n "${GAN_CHECKPOINT:-}" ]]; then
    init=(--gan-checkpoint "$GAN_CHECKPOINT")
fi

python -m videogan.finetune \
    --data-root "$UCF101_ROOT" \
    --split-dir "$split_dir" \
    --out-dir "$RUN_DIR" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    "${init[@]}"
