"""Outcome campaign: what a single bit flip DOES to an undefended run.

Every other bench in this repo measures a detector. This one deliberately
turns the detectors off and measures the fault itself, because the claim the
product rests on is not "we catch faults" -- it is "faults that nobody
catches produce answers that look fine and are wrong".

The three outcomes
------------------
Each trial injects exactly ONE bit flip into a warm, real workload and sorts
the result into one bucket. The taxonomy is the standard one from the GPU
fault-injection literature (NVBitFI; SASSIFI before it), stated in the terms
this repo can actually observe:

  MASKED    the observable output is bit-identical to the golden run. The
            fault happened and left no trace. Nothing to detect, nothing to
            fix.

  DETECTED  the run announced its own failure. Two sub-cases, reported
            separately because they are not the same event:
              * crash     -- the workload raised.
              * nonfinite -- the output contains NaN or +/-Inf.
            This is the DUE-equivalent bucket. On silicon a DUE is raised by
            ECC or by an Xid; in a software-only injector the analogue is a
            failure loud enough that a trivial screen catches it for free.
            `detect/guards.py` is exactly that trivial screen, which is why
            this bucket is the one already solved.

  SDC       the run completed, returned no error, and every number in the
            output is finite -- and the output is wrong. This is the number
            the campaign exists to produce.

SDC is split again, because "wrong" spans six orders of magnitude:

  sdc            the observable differs from golden at all (bitwise).
  sdc_critical   the model's PREDICTED TOKENS changed. Not a float in the
                 last mantissa place: a different answer. This is the
                 subset a user would see.

Both are reported. Quoting only the strict number would overstate the harm;
quoting only the critical number would understate the corruption.

Why the golden comparison is exact
----------------------------------
No tolerance is used anywhere. The observable is (float64 loss, argmax token
ids) and equality is bitwise. That is only legitimate if an uninjected run is
reproducible bit-for-bit, so the campaign refuses to start until it has
proved that: `--control` trials run the whole harness with NO injection and
must all come back MASKED. A control arm that reports a single non-masked
trial invalidates the campaign, and the run aborts rather than publishing.

Bit position is the point
-------------------------
The headline SDC rate depends entirely on which bits you flip, so a single
blended rate is close to meaningless on its own. `--sweep bits` holds the bit
fixed and varies the site, giving the outcome mix for each of the 32 fp32
bit positions. The contrast between the exponent field and the low mantissa
is the actual finding; the blended rate is a function of whatever bit prior
you assume, and this repo has two (uniform, and the LSB-biased measured
prior in `inject/gpu_model.py`). `--sweep uniform --bit-model {uniform,gpu}`
reports it under both, so the assumption is visible instead of buried.

Run:
    python -m bench.sdc_campaign --sweep bits --trials 100
    python -m bench.sdc_campaign --sweep uniform --trials 2000 --bit-model gpu
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from demo.workloads.nanogpt.data import get_batch
from orbital_runtime.inject.gpu_model import sample_bit_position
from orbital_runtime.inject.memory import flip_bit
from orbital_runtime.rng import stream
from orbital_runtime.workload import get_workload, resolve_device

# Outcome labels.
MASKED = "masked"
CRASH = "crash"
NONFINITE = "nonfinite"
SDC = "sdc"

DETECTED = (CRASH, NONFINITE)

# fp32 field boundaries. Bit 0 is the LSB of the mantissa.
FP32_MANTISSA_BITS = 23
FP32_SIGN_BIT = 31

# Injection sites.
TARGET_WEIGHT = "weight"
TARGET_OPTIMIZER = "optimizer"
TARGET_ACTIVATION = "activation"
TARGET_GRADIENT = "gradient"


def bit_field(bit: int) -> str:
    """Which IEEE-754 field a bit index lands in."""
    if bit == FP32_SIGN_BIT:
        return "sign"
    if bit >= FP32_MANTISSA_BITS:
        return "exponent"
    return "mantissa"


# --------------------------------------------------------------------------- #
# The observable
# --------------------------------------------------------------------------- #
# What the "output of the run" means, made concrete. A campaign that compares
# weights instead of outputs would call every flip an SDC by construction --
# the flipped bit is still sitting there. The observable has to be what the
# job RETURNS.


@dataclass(frozen=True)
class Observable:
    """The result a caller of this workload would receive."""

    loss: float  # mean cross-entropy, float64
    tokens: bytes  # argmax token ids over every position, int32 little-endian
    finite: bool  # every logit and the loss are finite

    def key(self) -> tuple:
        # Compare the loss by its exact bit pattern rather than by ==, so
        # that 0.0/-0.0 are distinguished and NaN compares equal to NaN
        # (== would call two identical NaN outputs "different" and inflate
        # the SDC count).
        return (struct.pack("<d", self.loss), self.tokens)

    def identical_to(self, other: Observable) -> bool:
        return self.key() == other.key()

    def tokens_changed(self, other: Observable) -> int:
        a = np.frombuffer(self.tokens, dtype=np.int32)
        b = np.frombuffer(other.tokens, dtype=np.int32)
        if a.shape != b.shape:
            return max(a.size, b.size)
        return int((a != b).sum())


@torch.no_grad()
def observe(model, batches) -> Observable:
    """Run the model over the fixed eval batches and capture what it returns."""
    was_training = model.training
    model.eval()
    losses = []
    toks = []
    all_finite = True
    for x, y in batches:
        logits, loss = model(x, y)
        if not bool(torch.isfinite(logits).all()):
            all_finite = False
        losses.append(loss.double())
        toks.append(logits.argmax(dim=-1).to(torch.int32).reshape(-1).cpu())
    total = torch.stack(losses).mean()
    lossf = float(total.item())
    if not math.isfinite(lossf):
        all_finite = False
    if was_training:
        model.train()
    return Observable(
        loss=lossf,
        tokens=torch.cat(toks).numpy().tobytes(),
        finite=all_finite,
    )


# --------------------------------------------------------------------------- #
# Injection sites
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Site:
    """One strikeable tensor."""

    name: str
    kind: str
    tensor: torch.Tensor


def weight_sites(model) -> list[Site]:
    # named_parameters() yields the tied lm_head/wte matrix once, so a tied
    # model cannot be struck twice in one trial.
    return [
        Site(n, TARGET_WEIGHT, p.data)
        for n, p in model.named_parameters()
        if p.dtype == torch.float32 and p.numel() > 0
    ]


def optimizer_sites(model, optimizer) -> list[Site]:
    names = {id(p): n for n, p in model.named_parameters()}
    out: list[Site] = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st:
                continue
            for key, val in st.items():
                if (
                    isinstance(val, torch.Tensor)
                    and val.dtype == torch.float32
                    and val.numel() > 0
                ):
                    out.append(
                        Site(f"{names.get(id(p), '?')}.{key}", TARGET_OPTIMIZER, val)
                    )
    return out


def pick_site(sites: list[Site], rng: np.random.Generator) -> tuple[Site, int]:
    """Uniform over resident ELEMENTS, not over tensors.

    Choosing a tensor first and an element second would over-weight the tiny
    tensors -- layer-norm gains are 384 elements, the embedding is 25 million,
    and treating them as equally likely targets misrepresents what a particle
    actually hits by five orders of magnitude.
    """
    sizes = np.array([s.tensor.numel() for s in sites], dtype=np.float64)
    flat = int(rng.integers(0, int(sizes.sum())))
    edges = np.cumsum(sizes)
    which = int(np.searchsorted(edges, flat, side="right"))
    offset = flat - int(edges[which - 1]) if which > 0 else flat
    return sites[which], offset


class ActivationInjector:
    """Flips one bit of one value in flight, at a chosen bit position.

    Distinct from `inject/compute.py`, which samples its own bit from the
    measured prior. The sweep needs the bit held fixed, and forcing that
    through the existing class would mean changing tested code to serve a
    bench.
    """

    def __init__(self, model, modules: list[str]):
        self.model = model
        self.names = modules
        self._handles: list = []
        self.fired: dict | None = None
        self._armed: tuple[str, int, int] | None = None  # (module, elem, bit)

    def module_names(self) -> list[str]:
        return list(self.names)

    def arm(self, module: str, elem_frac: float, bit: int) -> None:
        self._armed = (module, elem_frac, bit)
        self.fired = None

    def attach(self) -> ActivationInjector:
        lookup = dict(self.model.named_modules())
        for name in self.names:
            self._handles.append(lookup[name].register_forward_hook(self._hook(name)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> ActivationInjector:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    def _hook(self, name: str):
        def hook(_mod, _inp, output):
            if self._armed is None or self.fired is not None:
                return output
            want, elem_frac, bit = self._armed
            if name != want or not isinstance(output, torch.Tensor):
                return output
            if output.dtype != torch.float32 or output.numel() == 0:
                return output
            out = output if output.is_contiguous() else output.contiguous()
            idx = min(int(elem_frac * out.numel()), out.numel() - 1)
            corrupted = out.detach().clone()
            before, after = flip_bit(corrupted, idx, bit)
            self.fired = {
                "module": name,
                "index": idx,
                "bit": bit,
                "value_before": before,
                "value_after": after,
            }
            # Re-attach to the graph so the corrupted value propagates
            # forward exactly as a real strike would, without autograd trying
            # to differentiate the XOR.
            return output + (corrupted - out.detach())

        return hook


# --------------------------------------------------------------------------- #
# Trial record
# --------------------------------------------------------------------------- #


@dataclass
class Trial:
    outcome: str
    bit: int
    field: str
    target: str
    site: str
    value_before: float
    value_after: float
    rel_delta: float
    tokens_changed: int
    loss_delta: float
    critical: bool
    detail: str = ""

    def as_record(self) -> dict:
        return {
            "outcome": self.outcome,
            "bit": self.bit,
            "field": self.field,
            "target": self.target,
            "site": self.site,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "rel_delta": self.rel_delta if math.isfinite(self.rel_delta) else "inf",
            "tokens_changed": self.tokens_changed,
            "loss_delta": self.loss_delta if math.isfinite(self.loss_delta) else "inf",
            "critical": self.critical,
            "detail": self.detail,
        }


def rel_delta(before: float, after: float) -> float:
    if not math.isfinite(after):
        return math.inf
    if before == 0.0:
        return math.inf if after != 0.0 else 0.0
    return abs(after - before) / abs(before)


def classify(
    golden: Observable,
    got: Observable | None,
    exc: BaseException | None,
) -> tuple[str, bool]:
    """(outcome, critical). See the module docstring for the taxonomy."""
    if exc is not None:
        return CRASH, True
    assert got is not None
    if not got.finite:
        return NONFINITE, True
    if got.identical_to(golden):
        return MASKED, False
    return SDC, got.tokens_changed(golden) > 0


# --------------------------------------------------------------------------- #
# The campaign
# --------------------------------------------------------------------------- #


class Campaign:
    def __init__(self, args) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.workload = get_workload(
            "nanogpt",
            seed=args.seed,
            device=self.device,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
        self.model = self.workload.model
        self.optimizer = self.workload.optimizer

        # Warm the run up. A freshly initialised model is not a realistic
        # target: its weights are all ~N(0, 0.02), Adam has no state at all,
        # and the loss is uniform-random. Faults have to land in a model that
        # has learned something, or "the output changed" means nothing.
        self.model.train()
        for step in range(args.warmup):
            loss = self.workload.loss_for_step(step)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self.warm_loss = float(loss.item())

        # The eval batches are fixed for the whole campaign, so every trial
        # and the golden run see byte-identical input. Any difference in the
        # output is then attributable to the injected bit and nothing else.
        eval_rng = stream(args.seed, "sdc_campaign.eval")
        self.batches = [
            get_batch(
                self.workload.dataset.val,
                batch_size=args.batch_size,
                block_size=args.block_size,
                rng=eval_rng,
                device=self.device,
            )
            for _ in range(args.eval_batches)
        ]

        # The state every trial starts from. Optimizer state is snapshotted
        # too: in train mode a flip in an Adam moment is the whole point, and
        # a trial that left the moments dirty would contaminate the next one.
        self.warm_model = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        self.warm_opt = copy.deepcopy(self.optimizer.state_dict())

        # In train mode the observable is taken AFTER the run finishes its
        # remaining work, so the golden has to be measured the same way.
        self.golden = self.measure()
        self.restore()
        self.trials: list[Trial] = []

    # -- state handling ---------------------------------------------------- #

    def restore(self) -> None:
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                v.copy_(self.warm_model[k])
        self.optimizer.load_state_dict(copy.deepcopy(self.warm_opt))

    def train_steps(self, k: int, grad_hook=None) -> None:
        """Continue the real training loop for k steps from the warm point."""
        self.model.train()
        for i in range(k):
            loss = self.workload.loss_for_step(self.args.warmup + i)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if i == 0 and grad_hook is not None:
                grad_hook()
            self.optimizer.step()

    def measure(self, grad_hook=None) -> Observable:
        """Finish the job, then look at what it returned."""
        if self.args.mode == "train":
            self.train_steps(self.args.train_steps, grad_hook=grad_hook)
        return observe(self.model, self.batches)

    # -- preflight --------------------------------------------------------- #

    def control(self, n: int) -> list[str]:
        """Run the whole harness with NO injection. Must be all-masked."""
        out = []
        for _ in range(n):
            got = self.measure()
            self.restore()
            outcome, _ = classify(self.golden, got, None)
            out.append(outcome)
        return out

    # -- one trial --------------------------------------------------------- #

    def sites_for(self, target: str) -> list[Site]:
        if target == TARGET_WEIGHT:
            return weight_sites(self.model)
        if target == TARGET_OPTIMIZER:
            return optimizer_sites(self.model, self.optimizer)
        if target == "all":
            return weight_sites(self.model) + optimizer_sites(
                self.model, self.optimizer
            )
        raise ValueError(target)

    def run_stored(self, target: str, bit: int, rng: np.random.Generator) -> Trial:
        """Flip one bit of stored state, finish the job, observe, restore."""
        sites = self.sites_for(target)
        site, index = pick_site(sites, rng)
        before, after = flip_bit(site.tensor, index, bit)
        exc: BaseException | None = None
        got: Observable | None = None
        try:
            got = self.measure()
        except BaseException as e:  # noqa: BLE001 - the point is to catch anything
            exc = e
        finally:
            self.restore()
        outcome, critical = classify(self.golden, got, exc)
        return Trial(
            outcome=outcome,
            bit=bit,
            field=bit_field(bit),
            target=site.kind,
            site=site.name,
            value_before=before,
            value_after=after if math.isfinite(after) else math.inf,
            rel_delta=rel_delta(before, after),
            tokens_changed=got.tokens_changed(self.golden) if got else -1,
            loss_delta=abs(got.loss - self.golden.loss) if got else math.inf,
            critical=critical,
            detail=type(exc).__name__ if exc else "",
        )

    def run_activation(self, bit: int, rng: np.random.Generator) -> Trial:
        """Flip one bit of a value in flight."""
        names = self._activation_modules()
        mod = names[int(rng.integers(0, len(names)))]
        inj = ActivationInjector(self.model, names)
        inj.arm(mod, float(rng.random()), bit)
        exc: BaseException | None = None
        got: Observable | None = None
        with inj:
            try:
                got = self.measure()
            except BaseException as e:  # noqa: BLE001
                exc = e
            finally:
                self.restore()
        fired = inj.fired or {}
        outcome, critical = classify(self.golden, got, exc)
        before = fired.get("value_before", 0.0)
        after = fired.get("value_after", 0.0)
        return Trial(
            outcome=outcome,
            bit=bit,
            field=bit_field(bit),
            target=TARGET_ACTIVATION,
            site=fired.get("module", "<never fired>"),
            value_before=before,
            value_after=after if math.isfinite(after) else math.inf,
            rel_delta=rel_delta(before, after),
            tokens_changed=got.tokens_changed(self.golden) if got else -1,
            loss_delta=abs(got.loss - self.golden.loss) if got else math.inf,
            critical=critical,
            detail=type(exc).__name__ if exc else "",
        )

    def _activation_modules(self) -> list[str]:
        # Every nn.Linear output: the tensor-core / GEMM output surface, which
        # is what Tung et al. instrumented.
        return [
            n
            for n, m in self.model.named_modules()
            if isinstance(m, torch.nn.Linear)
        ]

    def run_gradient(self, bit: int, rng: np.random.Generator) -> Trial:
        """Flip one bit of a gradient between backward() and step().

        The narrowest window in the whole loop and the one with the most
        leverage: the corrupted value is written into the weights by the very
        next optimizer step, and from there into both Adam moments, where it
        keeps acting on every future step through the momentum term.
        """
        record: dict = {}

        def hook() -> None:
            grads = [
                Site(n, TARGET_GRADIENT, p.grad.data)
                for n, p in self.model.named_parameters()
                if p.grad is not None and p.grad.dtype == torch.float32
            ]
            if not grads:
                return
            site, index = pick_site(grads, rng)
            before, after = flip_bit(site.tensor, index, bit)
            record.update(site=site.name, before=before, after=after)

        exc: BaseException | None = None
        got: Observable | None = None
        try:
            got = self.measure(grad_hook=hook)
        except BaseException as e:  # noqa: BLE001
            exc = e
        finally:
            self.restore()
        outcome, critical = classify(self.golden, got, exc)
        before = record.get("before", 0.0)
        after = record.get("after", 0.0)
        return Trial(
            outcome=outcome,
            bit=bit,
            field=bit_field(bit),
            target=TARGET_GRADIENT,
            site=record.get("site", "<no grad>"),
            value_before=before,
            value_after=after if math.isfinite(after) else math.inf,
            rel_delta=rel_delta(before, after),
            tokens_changed=got.tokens_changed(self.golden) if got else -1,
            loss_delta=abs(got.loss - self.golden.loss) if got else math.inf,
            critical=critical,
            detail=type(exc).__name__ if exc else "",
        )

    def trial(self, target: str, bit: int, rng: np.random.Generator) -> Trial:
        if target == TARGET_ACTIVATION:
            return self.run_activation(bit, rng)
        if target == TARGET_GRADIENT:
            return self.run_gradient(bit, rng)
        return self.run_stored(target, bit, rng)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


def summarise(trials: list[Trial]) -> dict:
    c = Counter(t.outcome for t in trials)
    n = len(trials)
    crit = sum(1 for t in trials if t.outcome == SDC and t.critical)
    detected = c[CRASH] + c[NONFINITE]
    # Magnitude, over the SILENT failures only. Averaging the detected ones in
    # would report an infinity and say nothing about the quiet case, which is
    # the case that matters: two bits can both be 100% SDC and differ by ten
    # orders of magnitude in how wrong they make the answer.
    quiet = [t for t in trials if t.outcome == SDC]
    deltas = [t.loss_delta for t in quiet if math.isfinite(t.loss_delta)]
    return {
        "trials": n,
        "masked": c[MASKED],
        "crash": c[CRASH],
        "nonfinite": c[NONFINITE],
        "detected": detected,
        "sdc": c[SDC],
        "sdc_critical": crit,
        "masked_pct": 100.0 * c[MASKED] / n if n else 0.0,
        "detected_pct": 100.0 * detected / n if n else 0.0,
        "sdc_pct": 100.0 * c[SDC] / n if n else 0.0,
        "sdc_critical_pct": 100.0 * crit / n if n else 0.0,
        "sdc_loss_delta_median": _median(deltas),
        "sdc_loss_delta_max": max(deltas) if deltas else 0.0,
        "sdc_tokens_changed_median": _median(
            [float(t.tokens_changed) for t in quiet]
        ),
    }


def table(rows: list[tuple[str, dict]], label: str) -> str:
    head = (
        f"{label:<12} {'n':>6} {'masked':>14} {'detected':>14} "
        f"{'SDC':>14} {'SDC-critical':>14} {'loss delta':>12}"
    )
    lines = [head, "-" * len(head)]
    for name, s in rows:
        lines.append(
            f"{name:<12} {s['trials']:>6} "
            f"{s['masked']:>6} {s['masked_pct']:>6.1f}% "
            f"{s['detected']:>6} {s['detected_pct']:>6.1f}% "
            f"{s['sdc']:>6} {s['sdc_pct']:>6.1f}% "
            f"{s['sdc_critical']:>6} {s['sdc_critical_pct']:>6.1f}% "
            f"{s['sdc_loss_delta_median']:>12.2e}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", choices=("bits", "uniform"), default="bits")
    p.add_argument("--trials", type=int, default=100, help="per bit, or total")
    p.add_argument(
        "--target",
        choices=(
            TARGET_WEIGHT,
            TARGET_OPTIMIZER,
            TARGET_ACTIVATION,
            TARGET_GRADIENT,
            "all",
        ),
        default=TARGET_WEIGHT,
    )
    p.add_argument("--mode", choices=("inference", "train"), default="inference")
    p.add_argument(
        "--train-steps",
        type=int,
        default=10,
        help="train mode: steps run after the injection, before observing",
    )
    p.add_argument("--bit-model", choices=("uniform", "gpu"), default="uniform")
    p.add_argument("--bits", type=str, default="", help="comma list, default 0..31")
    p.add_argument("--control", type=int, default=8)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--json", type=str, default="")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    # Adam's moments and the gradients are not read by a forward pass, so
    # striking them in inference mode is guaranteed to be masked -- a 100%
    # masked row that describes the harness, not the hardware. Refuse rather
    # than publish a number that means nothing.
    if args.mode == "inference" and args.target in (
        TARGET_OPTIMIZER,
        TARGET_GRADIENT,
        "all",
    ):
        p.error(
            f"--target {args.target} has no effect on an inference-only "
            f"observable (optimizer state and gradients are never read by a "
            f"forward pass). Use --mode train."
        )

    t0 = time.perf_counter()
    print(
        f"building + warming workload ({args.warmup} steps, {args.device}), "
        f"mode={args.mode}"
        + (f" (+{args.train_steps} steps after injection)" if args.mode == "train" else "")
        + " ..."
    )
    camp = Campaign(args)
    params = sum(x.numel() for x in camp.model.parameters())
    resident = sum(
        s.tensor.numel()
        for s in weight_sites(camp.model) + optimizer_sites(camp.model, camp.optimizer)
    )
    print(
        f"  {params/1e6:.2f}M params, {resident/1e6:.2f}M resident fp32 elements "
        f"({resident*32/1e9:.2f}e9 bits)"
    )
    print(f"  warm train loss {camp.warm_loss:.6f}, golden eval loss {camp.golden.loss:.9f}")

    # Preflight. Without a clean A/A control the exact comparison is not
    # trustworthy and no SDC number from this harness means anything.
    ctrl = camp.control(args.control)
    bad = [o for o in ctrl if o != MASKED]
    print(f"  control ({args.control} uninjected runs): {len(ctrl)-len(bad)}/{len(ctrl)} masked")
    if bad:
        print(
            f"ABORT: the uninjected workload is not reproducible on this device "
            f"({Counter(bad)}). Exact golden comparison is invalid here; "
            f"use --device cpu."
        )
        return 2

    rng = stream(args.seed, "sdc_campaign.trials")
    per_bit: dict[int, list[Trial]] = defaultdict(list)

    if args.sweep == "bits":
        bits = (
            [int(b) for b in args.bits.split(",") if b.strip() != ""]
            if args.bits
            else list(range(32))
        )
        total = len(bits) * args.trials
        print(f"\nsweeping {len(bits)} bit positions x {args.trials} trials = {total}")
        done = 0
        for bit in bits:
            for _ in range(args.trials):
                t = camp.trial(args.target, bit, rng)
                camp.trials.append(t)
                per_bit[bit].append(t)
                done += 1
            s = summarise(per_bit[bit])
            print(
                f"  bit {bit:>2} ({bit_field(bit):<8}) "
                f"masked {s['masked_pct']:>5.1f}%  "
                f"detected {s['detected_pct']:>5.1f}%  "
                f"SDC {s['sdc_pct']:>5.1f}%  "
                f"SDC-crit {s['sdc_critical_pct']:>5.1f}%   [{done}/{total}]"
            )
    else:
        print(f"\n{args.trials} trials, bit drawn from the {args.bit_model!r} model")
        for i in range(args.trials):
            bit = (
                int(rng.integers(0, 32))
                if args.bit_model == "uniform"
                else sample_bit_position(rng, 32)
            )
            t = camp.trial(args.target, bit, rng)
            camp.trials.append(t)
            per_bit[bit].append(t)
            if (i + 1) % 200 == 0:
                s = summarise(camp.trials)
                print(
                    f"  [{i+1}/{args.trials}] masked {s['masked_pct']:.1f}%  "
                    f"detected {s['detected_pct']:.1f}%  SDC {s['sdc_pct']:.1f}%  "
                    f"SDC-crit {s['sdc_critical_pct']:.1f}%"
                )

    overall = summarise(camp.trials)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 76)
    print(f"OVERALL  ({args.target} target, {args.mode} mode, {args.sweep} sweep)")
    print("=" * 76)
    print(table([("all bits", overall)], "set"))
    if len(per_bit) > 1:
        print()
        by_field: dict[str, list[Trial]] = defaultdict(list)
        for t in camp.trials:
            by_field[t.field].append(t)
        print(
            table(
                [(f, summarise(by_field[f])) for f in ("sign", "exponent", "mantissa") if f in by_field],
                "field",
            )
        )
        print()
        print(
            table(
                [(f"bit {b}", summarise(per_bit[b])) for b in sorted(per_bit)],
                "bit",
            )
        )

    # The magnitude of the quiet failures, which the counts alone hide.
    quiet = [t for t in camp.trials if t.outcome == SDC]
    if quiet:
        deltas = sorted(t.loss_delta for t in quiet if math.isfinite(t.loss_delta))
        crit = [t for t in quiet if t.critical]
        print(f"\nSDC magnitude ({len(quiet)} silent failures)")
        print(f"  eval-loss delta   median {deltas[len(deltas)//2]:.3e}   max {deltas[-1]:.3e}")
        if crit:
            tc = sorted(t.tokens_changed for t in crit)
            ntok = len(np.frombuffer(camp.golden.tokens, dtype=np.int32))
            print(
                f"  tokens changed    median {tc[len(tc)//2]} of {ntok}   "
                f"max {tc[-1]} of {ntok}  ({100.0*tc[-1]/ntok:.1f}%)"
            )
    print(f"\n{len(camp.trials)} trials in {elapsed:.1f}s")

    if args.json:
        out = {
            "config": {
                "sweep": args.sweep,
                "mode": args.mode,
                "train_steps": args.train_steps if args.mode == "train" else 0,
                "target": args.target,
                "bit_model": args.bit_model,
                "trials_per_bit": args.trials,
                "warmup_steps": args.warmup,
                "eval_batches": args.eval_batches,
                "device": str(camp.device),
                "seed": args.seed,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "params": int(params),
                "resident_fp32_elements": int(resident),
                "torch": torch.__version__,
                "elapsed_s": elapsed,
            },
            "golden": {"loss": camp.golden.loss, "warm_train_loss": camp.warm_loss},
            "control": {"n": len(ctrl), "masked": len(ctrl) - len(bad)},
            "overall": overall,
            "by_bit": {str(b): summarise(per_bit[b]) for b in sorted(per_bit)},
            "by_field": {
                f: summarise([t for t in camp.trials if t.field == f])
                for f in ("sign", "exponent", "mantissa")
                if any(t.field == f for t in camp.trials)
            },
            "trials": [t.as_record() for t in camp.trials],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
