#!/usr/bin/env -S uv run --group launch python
import argparse
import sys
from enum import StrEnum
from pathlib import Path

import sky

NAME = "luolamies"
ROOT = Path(__file__).parent
DATA_FILES = ["tokenizer.json", "corpus.npy", "wiki.parquet"]
VERDA_BUCKET = "luolamies-checkpoints"
VERDA_GPUS = ["H100", "H200", "A100"]
DISK_SIZE = 50

PATH = 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'

SETUP = f"""
set -euo pipefail
{PATH}

command -v cargo >/dev/null ||
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
command -v uv >/dev/null ||
  curl -LsSf https://astral.sh/uv/install.sh | sh

UV_PYTHON_PREFERENCE=only-managed uv sync --frozen --no-group launch
"""

RUN = f"""
set -euo pipefail
{PATH}

ln -sfn "$HOME/data" data
UV_PYTHON_PREFERENCE=only-managed uv run --frozen --no-group launch main.py
"""


def base_task():
    return sky.Task(
        NAME,
        setup=SETUP,
        run=RUN,
        workdir=str(ROOT),
        file_mounts={f"~/data/{f}": str(ROOT / "data" / f) for f in DATA_FILES},
    )


def verda_task():
    task = base_task()
    task.set_resources(
        [
            sky.Resources(
                infra="verda",
                accelerators=f"{gpu}:1",
                use_spot=True,
                disk_size=DISK_SIZE,
            )
            for gpu in VERDA_GPUS
        ]
    )
    task.update_envs({"CHECKPOINT_FILE": "/checkpoints/checkpoint.pt"})
    task.set_storage_mounts(
        {
            "/checkpoints": sky.Storage(
                source=f"verda://{VERDA_BUCKET}",
                mode=sky.StorageMode.MOUNT_CACHED,
            )
        }
    )
    return task


def dryrun(task: sky.Task):
    sky.stream_and_get(sky.launch(task, cluster_name=NAME, down=True, dryrun=True))


def run_job(task: sky.Task):
    job_ids, _ = sky.stream_and_get(sky.jobs.launch(task, name=NAME))
    return sky.jobs.tail_logs(job_id=job_ids[0], follow=True)


parser = argparse.ArgumentParser()
parser.add_argument("--dryrun", action="store_true")


def main():
    args = parser.parse_args()
    if args.dryrun:
        dryrun(verda_task())
    else:
        sys.exit(run_job(verda_task()))


if __name__ == "__main__":
    main()
