#!/usr/bin/env python3
"""Run VariDP's train/eval.py unchanged, with ManiSkill's renderer disabled.

VariDP's eval.py calls gym.make without render_backend, so ManiSkill creates a CUDA
Vulkan render device even for state observations. WSL2 has no NVIDIA Vulkan driver
and fails with 'Failed to find a supported physical device "cuda:0"'. This wrapper
adds render_backend="none" to every gym.make call that does not set one, then runs
eval.py as __main__ with the given arguments. Run it from the VariDP checkout with
VariDP's own environment:

    uv run python /path/to/dp-manip/scripts/smoke/varidp_eval.py train/eval.py --ckpt ... --sim-backend cpu --num-envs 1
"""

import functools
import runpy
import sys

import gymnasium

_make = gymnasium.make


@functools.wraps(_make)
def _make_without_renderer(*args, **kwargs):
    kwargs.setdefault("render_backend", "none")
    return _make(*args, **kwargs)


if __name__ == "__main__":
    gymnasium.make = _make_without_renderer
    script, *arguments = sys.argv[1:]
    sys.argv = [script, *arguments]
    runpy.run_path(script, run_name="__main__")
