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
# run recovers a model matching the clean baseline (val 2.4304 vs 2.4304, to
# four decimals). At 3e-5 the unprotected run dies sooner (more dramatic) but
# the protected model ends degraded (~3.37) -- protection still converts death
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

# Regenerate the committed bench artifacts with the EXACT flags that produced
# them (review item 19). Each JSON also records its own model size + config +
# torch version in a `config` block, so a committed file is self-describing and
# cannot be misread as a different scale (review items 18/20).
#
# Portable + deterministic here: detect-eval is seeded and reproduces bit-for-bit;
# overhead-cpu is a timing measurement, so it reproduces up to this machine's
# noise floor (reported as an A/A control inside the run). The MPS, large-model,
# and CUDA (L4) artifacts need specific hardware -- their exact invocations are
# recorded below and in each file's `config` block, not run by the portable target.
bench: data
	$(PY) -m bench.overhead --device cpu --steps 120 --repeats 5 --json bench/results/overhead-cpu.json
	$(PY) -m bench.detect_eval --seeds 12 --rate 5e-4 --steps 120 --device cpu --json bench/results/detect-eval.json

# MPS (dev Mac) artifacts -- run on a machine with MPS:
#   overhead-mps.json:       bench.overhead --device mps --steps 120 --repeats 5 --json bench/results/overhead-mps.json
#   overhead-mps-large.json: bench.overhead --device mps --steps 40 --repeats 3 <LARGE MODEL FLAGS> --json bench/results/overhead-mps-large.json
# NOTE: the original 10.7M "large" run predates the in-JSON config block, so its
# exact --n-layer/--n-embd were not recorded -- the very gap this metadata closes.
# NVIDIA L4 (M4b, torch 2.7.0+cu128) artifacts, GPU gone -- recorded for provenance:
#   overhead_l4.json:          bench.overhead --device cuda --n-layer 12 --n-embd 768 --block-size 64 ...
#   detect_eval_l4.json:       bench.detect_eval --device cuda --n-layer 12 --n-embd 768 --block-size 64 --seeds 6 --rate 1e-7 ...
#   protect_overhead_l4.json:  bench.protect_overhead_calibrated --device cuda (INDICATIVE; superseded by the controlled script)

clean:
	rm -rf build *.egg-info runs/ .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
