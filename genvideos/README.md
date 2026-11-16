# Samples from the 2022 training run

Written at the end of each of the ten epochs of the 2022 run (iteration 30187 is the last
iteration of an epoch):

- `epochN_iter30187_video.mp4`: one generated video, 32 frames of 64 x 64 pixels at 25 frames
  per second, taken from the generator's output for the epoch's last training batch.
- `epochN_iter30187_bg.jpg`: the background stream of a different sample from the same batch
  (the 2022 code kept the batch's last background, not the one that belongs to the video).

The GIFs at the top of the main README were made from epochs 3, 6 and 9 with
[`scripts/make_gifs.sh`](../scripts/make_gifs.sh).
