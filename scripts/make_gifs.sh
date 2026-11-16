#!/bin/bash
# Convert sample videos to looping GIFs for the README, enlarged 3x without smoothing.
#
#   scripts/make_gifs.sh docs/figures genvideos/epoch3_iter30187_video.mp4 ...
#
# Each GIF is named after its input (epoch3_iter30187_video.mp4 -> golf_epoch3.gif).
# Requires ffmpeg on PATH; python -m videogan.sample writes GIFs of new samples without it.
set -euo pipefail

if ! command -v ffmpeg >/dev/null; then
    echo "make_gifs.sh needs the ffmpeg command on PATH" >&2
    exit 1
fi

out_dir=$1
shift
mkdir -p "$out_dir"
for video in "$@"; do
    name=$(basename "$video" .mp4)
    name=golf_${name%%_*}
    ffmpeg -v error -y -i "$video" \
        -vf "scale=iw*3:ih*3:flags=neighbor,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=none" \
        -loop 0 "$out_dir/$name.gif"
    echo "$out_dir/$name.gif"
done
