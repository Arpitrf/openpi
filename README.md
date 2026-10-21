# openpi: π₀.₅-DROID co-training

Fine-tunes `pi05_droid` (joint velocity actions) on a 50/50 mix of DROID and our own dataset.
Config: `pi05_droid_cotrain` in [src/openpi/training/config.py](src/openpi/training/config.py#L923).
The original openpi README is in [docs/openpi_README.md](docs/openpi_README.md).

## Setup

```bash
uv venv --python 3.11      # tensorflow-cpu 2.15 (RLDS loader) needs Python 3.11
uv sync --group rlds
```

**W&B:** put your own credentials in `.env` in the repo root (it's gitignored):

```bash
WANDB_API_KEY=<your key>   # from https://wandb.ai/authorize
WANDB_ENTITY=<user or team>
```

Don't run `wandb login`. It writes the shared `~/.netrc` and changes the login for everyone on this account.

## Data

| Dataset | Location | Weight |
|---|---|---|
| DROID 1.0.1 (RLDS) | `/data/shared/droid/1.0.1` | 1.0 for now, 0.5 once our data is added |
| Our dataset | `/data/shared/<name>/<version>` | 0.5 |

To add our dataset, uncomment its entry in `pi05_droid_cotrain` and set both weights to 0.5. It must be **RLDS in the DROID layout**:
- **Fields:** `action_dict/{joint_velocity, gripper_position}`, `observation/{exterior_image_1_left, exterior_image_2_left, wrist_image_left, joint_position, gripper_position}`, `language_instruction{,_2,_3}`, `traj_metadata/episode_metadata/{file_path, recording_folderpath}`.
- **Each `file_path` must contain `success`,** or the loader silently drops the episode.

## Train

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OPENPI_DATA_HOME=/data/users/arpit/.cache/openpi \
  uv run --env-file .env scripts/train.py pi05_droid_cotrain --exp-name=pi05_droid_finetune 2>&1 | tee pi05_droid_finetune.log
```

Multi-GPU: list the GPUs and scale the global batch (32 per GPU):

```bash
CUDA_VISIBLE_DEVICES=0,2,3,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OPENPI_DATA_HOME=/data/users/arpit/.cache/openpi \
  uv run --env-file .env scripts/train.py pi05_droid_cotrain --exp-name=<run_name> --batch-size 128 2>&1 | tee <run_name>.log
```

Useful flags:

| Flag | Effect |
|---|---|
| `--resume` | Continue from the latest checkpoint of `--exp-name` |
| `--overwrite` | Delete that run's checkpoints and start over |
| `--fsdp-devices N` | Split the model across N GPUs to save memory |
| `--project-name X` | W&B project (default is set in `TrainConfig`) |
| `--checkpoint-base-dir P` | Save checkpoints somewhere other than `./checkpoints` |

Run inside `tmux` so the job survives disconnects.

## Outputs

- **Checkpoints:** `checkpoints/pi05_droid_cotrain/<run_name>/<step>/`, about 42 GB each. One is saved every 1k steps; the latest and every 5k-th are kept.
- **Logs:** W&B (`wandb.ai/<entity>/<project>`) and `<run_name>.log`.

## Notes

- **GPUs are shared.** Check `nvidia-smi` first and pick GPUs with under ~5 GB in use.
- **Memory:** full fine-tuning at batch 32 per GPU only fits with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. It runs out of memory at the default 0.75 or with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. If memory is short, use `--fsdp-devices 2` or `--ema-decay None`.
- **Caches:** set `OPENPI_DATA_HOME` to the path above. The home disk is nearly full, and checkpoints download there by default.
- **Speed:** about 1.6 s per step on one H100 at batch 32. Startup takes about 4 minutes (idle-filter table, shuffle buffer, compilation).
