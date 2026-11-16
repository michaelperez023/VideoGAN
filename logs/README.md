# Training log of the 2022 run

`golf_training_2022.csv` holds the losses of the 2022 GAN training run on the golf dataset. They
come from the jobs' standard output, which had one line per iteration (20 clips), averaged over
windows of 500 iterations; the last window of each complete epoch has the remaining 188
iterations (195 for the partial epoch 8). The columns are the same as in the `train_log.csv`
that `videogan.train` writes:

| Column | Meaning |
|---|---|
| `epoch` | epoch, counting from 0 |
| `step` | iterations completed since the start of training |
| `epoch_progress` | position in training, in epochs |
| `loss_d` | discriminator loss |
| `loss_g` | generator loss, including the mask penalty |
| `d_real` | the discriminator's mean probability that real clips are real |
| `d_fake` | the discriminator's mean probability that generated clips are real |

Epochs 0 to 7 and 9 are complete. Epoch 8 has only its first 9,195 iterations: the first job
reached the cluster's 100-hour limit there, and the log of the job that ran epoch 8 again was not
kept. [`docs/figures/training_loss.png`](../docs/figures/training_loss.png) is drawn from this
file with [`scripts/plot_training_log.py`](../scripts/plot_training_log.py).
