"""Is bf16 ABFT a fixable tolerance-model bug, or a real sensitivity floor?

The tier detects nothing in bf16. `detect/abft.py` bounds rounding noise at
`safety * eps * sqrt(K) * scale`, keyed to the L1 magnitude of the summed
terms, and in bf16 that evaluates to more than the whole row being checked.
Two explanations fit that fact and they have opposite consequences.

  1. A tolerance-model BUG. `_verify` reduces both sides in fp32 (it casts
     explicitly, and says why), then sizes the bound with `x_.dtype`, which is
     bf16. The arithmetic is fp32 and the bound is bf16. If the bound simply
     names the wrong epsilon, the fix is one line.

  2. A real FLOOR. `y` is the module's actual output and is STORED in bf16, so
     it genuinely carries bf16 representation error of order eps_bf16 * |y|
     per element, whatever precision the comparison is done in. Corruption
     smaller than that is indistinguishable from rounding, by construction,
     and no tolerance formula recovers it.

Both can be true at once, which is the answer this probe exists to settle
rather than assume. It records the per-row residual and every scale term for
clean and injected forwards, then scores candidate bounds offline on the SAME
records, so the candidates are compared on identical data rather than on
separate runs.

Each candidate is scored on two numbers, never one. A bound that catches
everything by firing constantly is not a fix, so the clean-run false-positive
rate is reported beside the detection rate, and the empirical clean residual
distribution is printed against each threshold so the margin is visible
instead of argued about.

Run: python -m bench.abft_tolerance_probe [--dtype bfloat16] [--trials 25]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn

from bench.sdc_campaign import ActivationInjector, DTYPES, bit_field, bit_width
from orbital_runtime.detect.abft import AbftTier, DEFAULT_SAFETY_FACTOR
from orbital_runtime.inject.memory import flip_bit
from orbital_runtime.rng import stream
from orbital_runtime.workload import get_workload
from demo.workloads.nanogpt.data import get_batch

EPS = {dt: float(torch.finfo(dt).eps) for dt in (torch.float32, torch.bfloat16, torch.float16)}


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #
# The math below is lifted verbatim from AbftTier._verify. It is duplicated
# rather than monkeypatched so every intermediate can be kept, and it is
# asserted against the real tier's own ratio in `check_fidelity` so the
# duplicate cannot drift from the thing it claims to measure.


def verify_terms(module: nn.Linear, x: torch.Tensor, y: torch.Tensor,
                 s: torch.Tensor) -> dict:
    x_ = x.detach()
    y_ = y.detach()
    if module.bias is not None:
        y_ = y_ - module.bias
    xf = x_.to(torch.float32)
    yf = y_.to(torch.float32)
    lhs = yf.sum(dim=-1)
    rhs = xf @ s
    residual = (lhs - rhs).abs()
    l1_lhs = yf.abs().sum(dim=-1)
    l1_rhs = xf.abs() @ s.abs()
    return {
        "residual": residual.reshape(-1).double().numpy(),
        "l1_lhs": l1_lhs.reshape(-1).double().numpy(),
        "l1_rhs": l1_rhs.reshape(-1).double().numpy(),
        "abs_lhs": lhs.abs().reshape(-1).double().numpy(),
        "k": int(x_.shape[-1]),
        "dtype": x_.dtype,
        "y_dtype": y_.dtype,
        # Raw, unfloored, exactly as AbftTier passes it to _tolerance_terms.
        "l1_lhs_raw": l1_lhs.reshape(-1).double().numpy(),
    }


def trusted_checksum(weight: torch.Tensor) -> torch.Tensor:
    """Byte-identical to AbftTier.refresh_checksums().

    The order matters and is easy to get wrong: the tier sums in the WEIGHT's
    dtype and casts afterwards, so in bf16 the reference checksum itself
    carries bf16 accumulation error. Casting first and summing in fp32 gives a
    cleaner anchor than the tier actually has, and shifted the measured ratio
    by 6% before this was matched. `check_fidelity()` asserts the match.
    """
    return weight.detach().sum(dim=0).to(torch.float32)


def check_fidelity() -> None:
    """Assert the duplicated math above reproduces the real tier's ratio."""
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16):
        lin = nn.Linear(64, 32).to(dt)
        x = torch.randn(8, 64).to(dt)
        tier = AbftTier(nn.Sequential(lin), base_sample_rate=1.0,
                        saa_sample_rate=1.0, adaptive=False)
        tier.refresh_checksums()
        with torch.no_grad():
            y = lin(x)
            tier._pending = []
            tier._verify("0", lin, x, y)
            real = float(tier._pending[0][1])
            rec = verify_terms(lin, x, y, trusted_checksum(lin.weight))
        # Against the bound the tier CURRENTLY implements. When abft.py moved
        # from the single-term bound to `_tolerance_terms`, this check failed
        # rather than silently comparing the tier against a formula it no
        # longer used. That is the point of it.
        tol = candidates(rec, DEFAULT_SAFETY_FACTOR)["d_two_term"]
        mine = float((rec["residual"] / tol).max())
        rel = abs(real - mine) / max(real, 1e-30)
        if rel > 1e-6:
            raise AssertionError(
                f"probe drifted from AbftTier for {dt}: tier={real:.6e} "
                f"probe={mine:.6e} rel={rel:.2e}"
            )


# --------------------------------------------------------------------------- #
# Candidate bounds
# --------------------------------------------------------------------------- #


def candidates(rec: dict, safety: float) -> dict[str, np.ndarray]:
    """Per-row tolerance under each candidate, on one recorded check."""
    k = rec["k"]
    dt = rec.get("y_dtype", rec["dtype"])
    eps_store = EPS[dt]
    eps_acc = EPS[torch.float32]
    scale = np.maximum(rec["l1_lhs"], np.maximum(rec["l1_rhs"], 1e-8))
    result = np.maximum(rec["abs_lhs"], 1e-8)
    l1y = np.maximum(rec["l1_lhs"], 1e-8)
    return {
        # (a) the bound that shipped BEFORE this investigation: storage eps,
        #     sqrt(K), L1-scaled. Kept to reproduce the 0/N bf16 baseline.
        "a_current": safety * eps_store * math.sqrt(k) * scale,
        # (b) accumulation eps instead, everything else unchanged.
        "b_eps_fp32": safety * eps_acc * math.sqrt(k) * scale,
        # (c) storage eps, keyed to |result| rather than L1. This is the
        #     pre-M4c form the L1 scaling replaced; included because the
        #     question is whether L1 is what makes it catastrophic.
        "c_result_keyed": safety * eps_store * math.sqrt(k) * result,
        # (d) explicit two-term bound. Term one is the bf16 representation
        #     error of the stored output, keyed to L1(y) with NO sqrt(K),
        #     because an L1 sum is already the worst case over K terms and
        #     multiplying the two double-counts. Term two is the fp32
        #     accumulation error of the comparison itself.
        # Mirrors AbftTier._tolerance_terms exactly: raw l1_lhs (not floored)
        # for the storage term, floored `scale` for the accumulation term.
        # check_fidelity() asserts this against the live tier.
        "d_two_term": safety * (
            eps_store * rec.get("l1_lhs_raw", l1y) + eps_acc * math.sqrt(k) * scale
        ),
        # (e) storage term only, to isolate how much of (a) is the sqrt(K)
        #     and safety factors stacked on an already-worst-case scale.
        "e_store_only": eps_store * l1y,
    }


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


def linear_modules(model) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def run(args) -> int:
    check_fidelity()
    dtype = DTYPES[args.dtype]
    width = bit_width(dtype)
    torch.set_num_threads(args.threads or 4)

    w = get_workload("nanogpt", seed=args.seed, device="cpu", n_layer=args.n_layer,
                     n_head=args.n_head, n_embd=args.n_embd,
                     batch_size=args.batch_size, block_size=args.block_size)
    model = w.model
    model.train()
    for step in range(args.warmup):
        loss = w.loss_for_step(step)
        w.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        w.optimizer.step()
    model = model.to(dtype)
    w.model = model
    model.eval()

    rng = stream(args.seed, "abft_probe.eval")
    batches = [get_batch(w.dataset.val, batch_size=args.batch_size,
                         block_size=args.block_size, rng=rng, device="cpu")
               for _ in range(args.eval_batches)]

    names = linear_modules(model)
    lookup = dict(model.named_modules())
    checksums = {n: trusted_checksum(lookup[n].weight) for n in names}

    captured: list[dict] = []

    def hook_for(name):
        def hook(mod, inp, out):
            captured.append(verify_terms(mod, inp[0], out, checksums[name]))
            return out
        return hook

    def forward_all():
        handles = [lookup[n].register_forward_hook(hook_for(n)) for n in names]
        try:
            with torch.no_grad():
                for x, y in batches:
                    model(x, y)
        finally:
            for h in handles:
                h.remove()

    # --- clean phase ------------------------------------------------------ #
    print(f"probing {args.dtype} on cpu, {args.clean_passes} clean passes, "
          f"{args.trials} injected trials per bit\n")
    clean: list[list[dict]] = []
    for _ in range(args.clean_passes):
        captured.clear()
        forward_all()
        clean.append(list(captured))
    n_clean_checks = sum(len(c) for c in clean)

    # --- injected phase --------------------------------------------------- #
    bits = ([int(b) for b in args.bits.split(",")] if args.bits
            else [width - 2, width - 3, width // 2, 0])
    injected: dict[int, list[list[dict]]] = {b: [] for b in bits}
    irng = np.random.default_rng(args.seed)
    for bit in bits:
        for _ in range(args.trials):
            inj = ActivationInjector(model, names, dtype)
            inj.arm(names[int(irng.integers(0, len(names)))], float(irng.random()), bit)
            captured.clear()
            with inj:
                forward_all()
            injected[bit].append(list(captured))

    # --- scoring ---------------------------------------------------------- #
    def fires(records: list[dict], key: str) -> bool:
        for rec in records:
            tol = candidates(rec, args.safety)[key]
            if np.any(rec["residual"] > tol):
                return True
        return False

    keys = ["a_current", "b_eps_fp32", "c_result_keyed", "d_two_term", "e_store_only"]
    label = {
        "a_current": "(a) current: eps_store, sqrt(K), L1",
        "b_eps_fp32": "(b) eps_fp32, sqrt(K), L1",
        "c_result_keyed": "(c) eps_store, sqrt(K), |result|",
        "d_two_term": "(d) two-term: eps_store*L1(y) + eps_fp32*sqrt(K)*L1",
        "e_store_only": "(e) eps_store*L1(y) only, no safety",
    }

    print(f"{'candidate':<52}{'clean FP':>10}", end="")
    for b in bits:
        print(f"{'bit'+str(b):>9}", end="")
    print()
    print("-" * (62 + 9 * len(bits)))
    for key in keys:
        fp = sum(1 for c in clean if fires(c, key))
        print(f"{label[key]:<52}{fp:>4}/{len(clean):<5}", end="")
        for b in bits:
            hit = sum(1 for r in injected[b] if fires(r, key))
            print(f"{hit:>5}/{len(injected[b]):<3}", end="")
        print()

    # --- the margin, empirically ----------------------------------------- #
    print(f"\nclean residual against each bound, over {n_clean_checks} checks")
    print(f"{'candidate':<52}{'max resid/tol':>15}{'headroom':>12}")
    print("-" * 79)
    for key in keys:
        worst = 0.0
        for c in clean:
            for rec in c:
                tol = candidates(rec, args.safety)[key]
                worst = max(worst, float((rec["residual"] / tol).max()))
        print(f"{label[key]:<52}{worst:>15.3e}{1.0/worst if worst else float('inf'):>11.1f}x")

    print(f"\ntolerance as a multiple of a row's L1 magnitude ({args.dtype}, "
          f"safety={args.safety}):")
    for rec in clean[0][:1]:
        k = rec["k"]
        for key in keys:
            tol = candidates(rec, args.safety)[key]
            mult = float(np.median(tol / np.maximum(rec["l1_lhs"], 1e-8)))
            print(f"  {label[key]:<50} {mult:.3e} x L1   (K={k})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--clean-passes", type=int, default=25)
    p.add_argument("--bits", type=str, default="")
    p.add_argument("--safety", type=float, default=DEFAULT_SAFETY_FACTOR)
    p.add_argument("--warmup", type=int, default=120)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# Safety calibration
# --------------------------------------------------------------------------- #
# THE RULE IS FIXED HERE, BEFORE ANY NUMBER IS COMPUTED, and it reads only
# clean runs. Detection rates on injected trials are never consulted while
# choosing `safety`. Fitting the constant to the detection numbers we are about
# to publish would make those numbers meaningless.
#
#   safety := the smallest value in LADDER such that, over the CALIBRATION
#             split of clean checks only, max(residual / bound) <= TARGET.
#
# The bound is linear in safety, so max_ratio(s) = max_ratio(1) / s and the
# rule reduces to s >= max_ratio(1) / TARGET. That makes it mechanical: there
# is no choice left to make once the clean residuals are in.
#
# TARGET = 0.5 leaves a factor of two of headroom over the worst clean residual
# actually observed. It is a margin against clean runs we did not sample, not a
# tuning knob.

CALIBRATION_TARGET = 0.5
CALIBRATION_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def worst_clean_ratio(clean: list[list[dict]], key: str, safety: float) -> float:
    worst = 0.0
    for passes in clean:
        for rec in passes:
            tol = candidates(rec, safety)[key]
            worst = max(worst, float((rec["residual"] / tol).max()))
    return worst


def calibrate(clean: list[list[dict]], key: str) -> tuple[int, float]:
    """Smallest ladder value meeting CALIBRATION_TARGET on clean data alone."""
    at_one = worst_clean_ratio(clean, key, 1.0)
    for s in CALIBRATION_LADDER:
        if at_one / s <= CALIBRATION_TARGET:
            return s, at_one
    return CALIBRATION_LADDER[-1], at_one
