#!/usr/bin/env -S uvx --from skypilot[gcp]==0.13.0 python
from enum import StrEnum
from enum import Enum
import subprocess
import sys
import argparse
from pathlib import Path

import sky

NAME = "luolamies"
ROOT = Path(__file__).parent
DATA_FILES = ["tokenizer.json", "corpus.npy", "wiki.parquet"]
CHECKPOINT_BUCKET = "luolamies-checkpoints-505220"
GCP_REGIONS = ["europe-west4", "europe-west1", "europe-west3"]
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

UV_PYTHON_PREFERENCE=only-managed uv sync
"""

RUN = f"""
set -euo pipefail
{PATH}

ln -sfn "$HOME/data" data
UV_PYTHON_PREFERENCE=only-managed uv run main.py
"""


def base_task():
    return sky.Task(
        NAME,
        setup=SETUP,
        run=RUN,
        workdir=str(ROOT),
        file_mounts={f"~/data/{f}": str(ROOT / "data" / f) for f in DATA_FILES},
    )


def gcp_task():
    task = base_task()
    task.set_resources(
        [
            sky.Resources(
                infra=f"gcp/{region}",
                accelerators="L4:1",
                use_spot=True,
                disk_size=DISK_SIZE,
            )
            for region in GCP_REGIONS
        ]
    )
    task.update_envs({"CHECKPOINT_FILE": "/checkpoints/checkpoint.pt"})
    task.set_storage_mounts(
        {
            "/checkpoints": sky.Storage(
                name=CHECKPOINT_BUCKET,
                stores=[sky.StoreType.GCS],
                mode=sky.StorageMode.MOUNT_CACHED,
            )
        }
    )
    return task


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
    return task


def run(task: sky.Task, down: bool, dryrun: bool):
    request = sky.launch(task, cluster_name=NAME, down=down, dryrun=dryrun)
    job_id, _ = sky.stream_and_get(request)
    if dryrun:
        return None
    return sky.tail_logs(NAME, job_id, follow=True)


def fetch_checkpoint_and_down():
    subprocess.run(
        ["rsync", "-avz", f"{NAME}:sky_workdir/checkpoint.pt", str(ROOT / "data")],
        check=True,
    )
    sky.stream_and_get(sky.down(NAME))


class Cloud(StrEnum):
    Verda = "verda"
    GCP = "gcp"


parser = argparse.ArgumentParser()

parser.add_argument("cloud", type=Cloud, choices=list(Cloud))
parser.add_argument("--dryrun", action="store_true")


def main():
    args = parser.parse_args()
    cloud = args.cloud
    dryrun = args.dryrun

    if cloud == Cloud.GCP:
        run(gcp_task(), down=True, dryrun=dryrun)
    elif cloud == Cloud.Verda:
        exit_code = run(verda_task(), down=False, dryrun=dryrun)
        if exit_code == 0:
            fetch_checkpoint_and_down()
    else:
        sys.exit("usage: launch.py gcp|verda [--dryrun]")


if __name__ == "__main__":
    main()
