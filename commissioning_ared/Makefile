# Run from commissioning_ared/. On Windows, GNU make is required for these
# targets; the same commands are listed in the README for PowerShell.

PYTHON ?= python
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif

.PHONY: env test toy parking sweep

env:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install -U pip
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

test:
	$(VENV_PY) -m pytest

# Scaled NICE, one seed, cold start vs farthest-first handoff.
toy:
	$(VENV_PY) -m src.run_experiment --method cold_ared --data NICE --stream stationary --t-frac 0.2 --B0 20 --kappa 1.0 --seed 0 --n-rel-classes 4 --l-buf-size 200 --nice-base 400 --query-rule global_fft --out results/toy.jsonl --overwrite
	$(VENV_PY) -m src.run_experiment --method farpoint_then_ared --data NICE --stream stationary --t-frac 0.2 --B0 20 --kappa 1.0 --seed 0 --n-rel-classes 4 --l-buf-size 200 --nice-base 400 --query-rule global_fft --out results/toy.jsonl
	$(VENV_PY) -m src.run_experiment --method cold_ared --data NICE --stream late_arrival --t-frac 0.2 --B0 20 --kappa 1.0 --seed 0 --n-rel-classes 4 --l-buf-size 200 --nice-base 400 --out results/toy.jsonl
	$(VENV_PY) -m src.run_experiment --method farpoint_then_ared --data NICE --stream late_arrival --t-frac 0.2 --B0 20 --kappa 1.0 --seed 0 --n-rel-classes 4 --l-buf-size 200 --nice-base 400 --query-rule global_fft --out results/toy.jsonl
	$(VENV_PY) -m src.plotting --jsonl results/toy.jsonl --out results/figures --summary results/summary.csv

# PARKING_LOT_DINO smoke test. Exits with a path listing when the latent cache is absent.
parking:
	$(VENV_PY) -m src.run_sweep --config experiments/configs/parking_smoke.yaml --out results/parking_smoke

# Small NICE matrix (all four methods, both stream constructions).
sweep:
	$(VENV_PY) -m src.run_sweep --config experiments/configs/nice_small.yaml --out results/nice_small
