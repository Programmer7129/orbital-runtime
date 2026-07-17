PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

CORPUS := demo/workloads/nanogpt/data/input.txt
SHAKESPEARE_URL := https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

.PHONY: venv install data test demo clean

venv:
	python3 -m venv .venv

install:
	$(PIP) install -e ".[dev]" -q

# The corpus is vendored in-repo; this only re-fetches it if it goes missing.
data: $(CORPUS)
$(CORPUS):
	@mkdir -p $(dir $(CORPUS))
	curl -sS -f -o $(CORPUS) $(SHAKESPEARE_URL)

test: data
	$(PYTEST)

# M1 demo: a clean baseline, then the same run under radiation. The
# unprotected run dies (NaN) or finishes silently degraded -- either way it
# is corrupted, which is the M1 deliverable. `-` on the second line because
# a dead run exits non-zero and that is the POINT.
#
# The rate is elevated above the calibrated 1e-9..1e-7 flight band because
# this model holds 7.8e7 resident bits against an H100's 6.4e11 -- four
# orders of magnitude fewer bits to hit. The band itself is asserted against
# real H100 bit counts in tests/test_flux.py. M4 produces headline numbers
# at realistic scale on a rented GPU.
DEMO_RATE ?= 3e-5
DEMO_SEED ?= 1337
DEMO_STEPS ?= 200

demo: data
	$(PY) -m orbital_runtime.run --workload nanogpt --orbits 2 --rate 0 \
		--protect off --seed $(DEMO_SEED) --steps $(DEMO_STEPS) --tag baseline
	-$(PY) -m orbital_runtime.run --workload nanogpt --orbits 2 --rate $(DEMO_RATE) \
		--protect off --seed $(DEMO_SEED) --steps $(DEMO_STEPS) --tag unprotected

clean:
	rm -rf build *.egg-info runs/ .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
