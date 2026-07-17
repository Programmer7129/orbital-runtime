PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

.PHONY: venv install test demo clean

venv:
	python3 -m venv .venv

install:
	$(PIP) install -e ".[dev]" -q

test:
	$(PYTEST)

# M1 demo: an unprotected nanoGPT run at elevated upset rate dies or silently
# diverges; a clean baseline (--rate 0) is run first for comparison.
demo:
	$(PY) -m orbital_runtime.run --workload nanogpt --orbits 2 --rate 0 --protect off --seed 1337 --tag baseline
	-$(PY) -m orbital_runtime.run --workload nanogpt --orbits 2 --rate 3e-6 --protect off --seed 1337 --tag unprotected

clean:
	rm -rf .venv build *.egg-info runs/
