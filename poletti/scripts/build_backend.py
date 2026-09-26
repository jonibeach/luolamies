import subprocess
from maturin import *
import maturin


def gen():
    subprocess.run(["cargo", "run", "--bin", "stub_gen"], check=True)


def build_wheel(*args, **kwargs):
    gen()
    return maturin.build_wheel(*args, **kwargs)


def build_editable(*args, **kwargs):
    gen()
    return maturin.build_editable(*args, **kwargs)
