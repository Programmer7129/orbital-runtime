PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

CORPUS := demo/workloads/nanogpt/data/input.txt
SHAKESPEARE_URL := https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

.PHONY: venv install data test demo bench clean

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

# The three-way story: a clean baseline, the same run under radiation
# unprotected (dies), and the same run protected (survives).
#
# `-` on the unprotected line because a dead run exits non-zero, and that is
# the POINT.
#
# The rate is elevated above the calibrated 1e-9..1e-7 flight band because
# this model holds 7.8e7 resident bits against an H100's 6.4e11 -- four
# orders of magnitude fewer bits to hit. The band itself is asserted against
# real H100 bit counts in tests/test_flux.py. M4 produces headline numbers
# at realistic scale on a rented GPU.
#
# 3e-6 is the honest demo rate: the unprotected run dies, and the protected
# run recovers a model matching the clean baseline (val 2.4288 vs 2.4304).
# At 3e-5 the unprotected run dies sooner (step 38, more dramatic) but the
# protected model ends degraded (~3.37) -- protection still converts death
# into survival, but sub-detection-floor strikes accumulate. Set
# DEMO_RATE=3e-5 to see that regime.
DEMO_RATE ?= 3e-6
DEMO_SEED ?= 1337
DEMO_STEPS ?= 200
DEMO_ARGS = --workload nanogpt --orbits 2 --seed $(DEMO_SEED) --steps $(DEMO_STEPS)

demo: data
	$(PY) -m orbital_runtime.run $(DEMO_ARGS) --rate 0 --protect off --tag baseline
	-$(PY) -m orbital_runtime.run $(DEMO_ARGS) --rate $(DEMO_RATE) --protect off --tag unprotected
	$(PY) -m orbital_runtime.run $(DEMO_ARGS) --rate $(DEMO_RATE) --protect on --tag protected

bench: data
	$(PY) -m bench.overhead --json bench/results/overhead.json
	$(PY) -m bench.detect_eval --json bench/results/detect-eval.json

clean:
	rm -rf build *.egg-info runs/ .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
