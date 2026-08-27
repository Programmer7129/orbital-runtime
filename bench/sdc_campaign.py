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
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.integrity import IntegrityTier
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

# Float formats. Bit 0 is always the LSB of the mantissa, and the sign bit is
# always the top bit, but everything between moves with the format. bf16 is the
# case that matters commercially: it keeps all 8 exponent bits of fp32 and pays
# for them out of the mantissa, so HALF of a bf16 word is exponent or sign
# against a quarter of an fp32 word. A bit-position table is meaningless without
# the format printed next to it.
FORMATS: dict[torch.dtype, dict] = {
    torch.float32: {"name": "fp32", "width": 32, "mantissa": 23, "exponent": 8},
    torch.bfloat16: {"name": "bf16", "width": 16, "mantissa": 7, "exponent": 8},
    torch.float16: {"name": "fp16", "width": 16, "mantissa": 10, "exponent": 5},
}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# Injection sites.
TARGET_WEIGHT = "weight"
TARGET_OPTIMIZER = "optimizer"
TARGET_ACTIVATION = "activation"
TARGET_GRADIENT = "gradient"

# Arms.
ARM_UNDEFENDED = "undefended"
ARM_DEFENDED = "defended"

# Defended-arm outcomes, on top of the three shared ones.
REPAIRED = "repaired"
CAUGHT = "caught"


def bit_field(bit: int, dtype: torch.dtype = torch.float32) -> str:
    """Which IEEE-754 field a bit index lands in, for this format."""
    f = FORMATS[dtype]
    if bit == f["width"] - 1:
        return "sign"
    if bit >= f["mantissa"]:
        return "exponent"
    return "mantissa"


def format_name(dtype: torch.dtype) -> str:
    return FORMATS[dtype]["name"]


def bit_width(dtype: torch.dtype) -> int:
    return FORMATS[dtype]["width"]


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
        logits, _ = model(x, y)
        if not bool(torch.isfinite(logits).all()):
            all_finite = False
        # Reduce the loss in fp32 even when the model is bf16. Production
        # mixed precision does exactly this: the matmuls run narrow, the
        # softmax and the loss reduction run in fp32. Taking the loss in
        # native bf16 instead makes it too coarse to resolve the corruption
        # at all, so every bf16 SDC reports a loss delta of exactly zero and
        # the number describes the observable rather than the fault.
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1)
        )
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


def weight_sites(model, dtype: torch.dtype = torch.float32) -> list[Site]:
    # named_parameters() yields the tied lm_head/wte matrix once, so a tied
    # model cannot be struck twice in one trial.
    #
    # Sites are restricted to the campaign format. A bf16 campaign that also
    # struck fp32 tensors would mix two bit layouts into one bit-position
    # table, where bit 14 means "exponent" in one row and "mantissa" in the
    # other. The table would be arithmetic over two different alphabets.
    return [
        Site(n, TARGET_WEIGHT, p.data)
        for n, p in model.named_parameters()
        if p.dtype == dtype and p.numel() > 0
    ]


def optimizer_sites(model, optimizer, dtype: torch.dtype = torch.float32) -> list[Site]:
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
                    and val.dtype == dtype
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
    fmt: str = "fp32"
    arm: str = ARM_UNDEFENDED
    fired_tier: str = ""  # which detector tier fired, defended arm only
    repaired: bool = False

    def as_record(self) -> dict:
        return {
            "outcome": self.outcome,
            "bit": self.bit,
            "field": self.field,
            "fmt": self.fmt,
            "arm": self.arm,
            "fired_tier": self.fired_tier,
            "repaired": self.repaired,
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


def classify_defended(
    golden: Observable,
    got: Observable | None,
    exc: BaseException | None,
    fired: bool,
    repaired: bool,
) -> tuple[str, bool]:
    """The same taxonomy, plus the two outcomes only a defended run can have.

    Order matters and is chosen so the defended arm can never flatter itself:

      * A crash or a non-finite output is still reported as such, even if a
        tier also fired. Those are failures the free screen already caught, so
        crediting the detector for them would inflate the delta between the
        arms with work it did not do.
      * `repaired` requires BOTH that a tier fired and that the output came
        back bit-identical to the golden run. A repair that leaves the answer
        wrong is not a repair, and it is reported as `caught`.
      * `caught` means a tier fired and the output is still wrong. The fault
        was seen but not undone. That is a real outcome and it is not an SDC,
        because the run is no longer silent.
      * `sdc` in this arm means every tier stayed quiet and the answer is
        wrong. This is the number the defended arm exists to shrink.
    """
    if exc is not None:
        return CRASH, True
    assert got is not None
    if not got.finite:
        return NONFINITE, True
    identical = got.identical_to(golden)
    if fired:
        if identical and repaired:
            return REPAIRED, False
        return CAUGHT, (not identical) and got.tokens_changed(golden) > 0
    if identical:
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

        # Warm up in fp32 whatever the campaign format is, then cast. Adam is
        # not stable in bf16 with no fp32 master copy, so training the warmup
        # in bf16 would measure an optimizer problem rather than a fault
        # problem. Casting afterwards is also what deployment actually does:
        # train in fp32 or mixed precision, then serve the weights narrow.
        self.dtype = DTYPES[args.dtype]
        self.fmt = format_name(self.dtype)
        self.width = bit_width(self.dtype)
        if self.dtype is not torch.float32:
            self.model = self.model.to(self.dtype)
            self.workload.model = self.model

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

        # The defended arm. Both tiers run at their ceiling: the integrity tier
        # scans every step (no flux model, so scan_interval() is 1) and ABFT
        # samples every Linear (rate 1.0). That measures what the detectors CAN
        # see, not what the shipping configuration sees. The shipping sample
        # rates are lower and cost 25.7% overhead, which this campaign does not
        # measure and does not claim.
        self.integrity: IntegrityTier | None = None
        self.abft: AbftTier | None = None
        # A monotonic step counter for the tiers. IntegrityTier.check_now()
        # rate-limits itself against the last step it scanned, and reset() does
        # NOT clear that bookkeeping, so passing step=0 every trial makes every
        # trial after the first silently skip its scan and report zero
        # detections. Found by the defended arm catching 0% of flips that an
        # exact integer checksum cannot possibly miss.
        self._tier_step = 0
        if args.arm in (ARM_DEFENDED, "paired"):
            self.integrity = IntegrityTier(
                model=self.model, optimizer=self.optimizer, repair=True
            )
            self.abft = AbftTier(
                self.model,
                base_sample_rate=1.0,
                saa_sample_rate=1.0,
                adaptive=False,
                rng=np.random.default_rng(args.seed),
            )

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
            sites = weight_sites(self.model, self.dtype)
        elif target == TARGET_OPTIMIZER:
            sites = optimizer_sites(self.model, self.optimizer, self.dtype)
        elif target == "all":
            sites = weight_sites(self.model, self.dtype) + optimizer_sites(
                self.model, self.optimizer, self.dtype
            )
        else:
            raise ValueError(target)
        if not sites:
            raise RuntimeError(
                f"no {self.fmt} tensors to strike for target {target!r}. "
                f"Adam keeps its moments in the dtype they were created in, so "
                f"a --dtype bfloat16 campaign has fp32 optimizer state and no "
                f"bf16 optimizer target. Use --target weight."
            )
        return sites

    # -- the defended arm --------------------------------------------- #

    def arm_defenses(self) -> None:
        """Snapshot the trusted state. Must run BEFORE the flip."""
        self._tier_step += 1
        if self.integrity is not None:
            self.integrity.reset()
            self.integrity.refresh()
        if self.abft is not None:
            self.abft.reset()
            self.abft.refresh_checksums()

    def check_stored(self) -> tuple[bool, str, bool]:
        """Verify stored state after the flip. Returns (fired, tier, repaired)."""
        if self.integrity is None:
            return False, "", False
        before = self.integrity.stats.repaired
        verdict = self.integrity.check_now(step=self._tier_step)
        repaired = self.integrity.stats.repaired > before
        # A repair counts as seeing the fault even when the tier decides the
        # residual is too small to escalate. Reporting only escalations would
        # undercount the detector against itself.
        fired = bool(verdict.triggered) or repaired
        return fired, (verdict.tier if verdict.triggered else "integrity"), repaired

    def run_stored(self, target: str, bit: int, rng: np.random.Generator,
                   arm: str = ARM_UNDEFENDED) -> Trial:
        """Flip one bit of stored state, finish the job, observe, restore."""
        sites = self.sites_for(target)
        site, index = pick_site(sites, rng)
        defended = arm == ARM_DEFENDED
        if defended:
            self.arm_defenses()
        before, after = flip_bit(site.tensor, index, bit)
        fired, tier_name, repaired = (
            self.check_stored() if defended else (False, "", False)
        )
        exc: BaseException | None = None
        got: Observable | None = None
        try:
            got = self.measure()
        except BaseException as e:  # noqa: BLE001 - the point is to catch anything
            exc = e
        finally:
            self.restore()
        outcome, critical = (
            classify_defended(self.golden, got, exc, fired, repaired)
            if defended
            else classify(self.golden, got, exc)
        )
        return Trial(
            outcome=outcome,
            bit=bit,
            field=bit_field(bit, self.dtype),
            target=site.kind,
            site=site.name,
            value_before=before,
            value_after=after if math.isfinite(after) else math.inf,
            rel_delta=rel_delta(before, after),
            tokens_changed=got.tokens_changed(self.golden) if got else -1,
            loss_delta=abs(got.loss - self.golden.loss) if got else math.inf,
            critical=critical,
            detail=type(exc).__name__ if exc else "",
            fmt=self.fmt,
            arm=arm,
            fired_tier=tier_name if fired else "",
            repaired=repaired,
        )

    def run_activation(self, bit: int, rng: np.random.Generator,
                       arm: str = ARM_UNDEFENDED) -> Trial:
        """Flip one bit of a value in flight."""
        names = self._activation_modules()
        mod = names[int(rng.integers(0, len(names)))]
        inj = ActivationInjector(self.model, names)
        inj.arm(mod, float(rng.random()), bit)
        defended = arm == ARM_DEFENDED
        exc: BaseException | None = None
        got: Observable | None = None
        tier_fired, tier_name = False, ""
        if defended:
            # ABFT is the only tier that can see this. A corrupted activation
            # leaves the stored weights bit-identical, so a state checksum is
            # structurally blind to it.
            self.arm_defenses()
            assert self.abft is not None
            self.abft.arm()
            # Hook order is load-bearing. Forward hooks fire in registration
            # order, so the injector must attach FIRST: it stands in for a
            # fault inside the GEMM, and ABFT has to see the corrupted output
            # the way the hardware would hand it over. Attaching ABFT first
            # makes it read the clean tensor and pass every time, which reads
            # as "ABFT detects nothing" and is a bug in the harness, not a
            # result about the detector.
            with inj:
                with self.abft:
                    try:
                        got = self.measure()
                    except BaseException as e:  # noqa: BLE001
                        exc = e
            verdict = self.abft.observe(step=self._tier_step)
            tier_fired = bool(verdict.triggered)
            tier_name = verdict.tier if tier_fired else ""
            self.restore()
        else:
            with inj:
                try:
                    got = self.measure()
                except BaseException as e:  # noqa: BLE001
                    exc = e
                finally:
                    self.restore()
        fired = inj.fired or {}
        outcome, critical = (
            classify_defended(self.golden, got, exc, tier_fired, False)
            if defended
            else classify(self.golden, got, exc)
        )
        before = fired.get("value_before", 0.0)
        after = fired.get("value_after", 0.0)
        return Trial(
            outcome=outcome,
            bit=bit,
            field=bit_field(bit, self.dtype),
            target=TARGET_ACTIVATION,
            site=fired.get("module", "<never fired>"),
            value_before=before,
            value_after=after if math.isfinite(after) else math.inf,
            rel_delta=rel_delta(before, after),
            tokens_changed=got.tokens_changed(self.golden) if got else -1,
            loss_delta=abs(got.loss - self.golden.loss) if got else math.inf,
            critical=critical,
            detail=type(exc).__name__ if exc else "",
            fmt=self.fmt,
            arm=arm,
            fired_tier=tier_name,
            repaired=False,
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
            field=bit_field(bit, self.dtype),
            target=TARGET_GRADIENT,
            site=record.get("site", "<no grad>"),
            value_before=before,
            value_after=after if math.isfinite(after) else math.inf,
            rel_delta=rel_delta(before, after),
            tokens_changed=got.tokens_changed(self.golden) if got else -1,
            loss_delta=abs(got.loss - self.golden.loss) if got else math.inf,
            critical=critical,
            detail=type(exc).__name__ if exc else "",
            fmt=self.fmt,
            arm=ARM_UNDEFENDED,
        )

    def trial(self, target: str, bit: int, rng: np.random.Generator,
              arm: str = ARM_UNDEFENDED) -> Trial:
        if target == TARGET_ACTIVATION:
            return self.run_activation(bit, rng, arm)
        if target == TARGET_GRADIENT:
            # No defended arm here, deliberately. The integrity tier must run
            # BEFORE optimizer.step(), but a gradient fault only reaches the
            # state THROUGH that step. Checking after it compares legitimately
            # updated weights against a stale snapshot, which the tier's own
            # docstring records as having produced a 100% false positive rate.
            # A defended gradient number would measure that bug, not detection.
            return self.run_gradient(bit, rng)
        return self.run_stored(target, bit, rng, arm)


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
        "repaired": c[REPAIRED],
        "caught": c[CAUGHT],
        "repaired_pct": 100.0 * c[REPAIRED] / n if n else 0.0,
        "caught_pct": 100.0 * c[CAUGHT] / n if n else 0.0,
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


def measure_noise_band(args, n_seeds: int) -> dict:
    """Spread across independently seeded models trained to the same point.

    This is NOT run-to-run noise. On a fixed seed and device the uninjected
    workload here reproduces bit for bit, so run-to-run noise is exactly zero,
    which is why the golden comparison can be bitwise at all.

    What this measures is a different and larger quantity: how far apart two
    honestly trained models land. It is reported because a reader needs to know
    the scale a loss delta lives on. It is deliberately NOT used as the
    threshold for calling a trial corrupted. Doing that would compare a
    corrupted run against a DIFFERENTLY SEEDED model and let real corruption
    hide inside training variance. The corruption threshold stays what it was:
    the output changed, and the token-level subset of that.
    """
    losses, token_sets = [], []
    for k in range(n_seeds):
        sub = argparse.Namespace(**vars(args))
        sub.seed = args.seed + 1000 * (k + 1)
        sub.arm = ARM_UNDEFENDED
        c = Campaign(sub)
        losses.append(c.golden.loss)
        token_sets.append(c.golden.tokens)
        del c
    pair_loss, pair_tok = [], []
    ntok = len(np.frombuffer(token_sets[0], dtype=np.int32))
    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            pair_loss.append(abs(losses[i] - losses[j]))
            a = np.frombuffer(token_sets[i], dtype=np.int32)
            b = np.frombuffer(token_sets[j], dtype=np.int32)
            pair_tok.append(int((a != b).sum()))
    return {
        "seeds": n_seeds,
        "losses": losses,
        "loss_spread_min": min(pair_loss) if pair_loss else 0.0,
        "loss_spread_median": _median(pair_loss),
        "loss_spread_max": max(pair_loss) if pair_loss else 0.0,
        "tokens_differing_median": _median([float(x) for x in pair_tok]),
        "tokens_total": ntok,
    }


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
        "--arm",
        choices=(ARM_UNDEFENDED, ARM_DEFENDED, "paired"),
        default=ARM_UNDEFENDED,
        help="paired runs both arms on the same site and bit, trial for trial",
    )
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    p.add_argument(
        "--noise-band",
        type=int,
        default=0,
        help="train this many extra seeds to measure the seed-to-seed floor",
    )
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
    if args.arm in (ARM_DEFENDED, "paired") and args.target == TARGET_GRADIENT:
        p.error(
            "--target gradient has no defended arm. The integrity tier must run "
            "before optimizer.step(), but a gradient fault only reaches the "
            "state through that step. Checking after it compares legitimately "
            "updated weights against a stale snapshot, which the tier's own "
            "docstring records as a 100% false positive rate. Run this target "
            "undefended, or measure the defended arm on --target optimizer."
        )

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
        for s in weight_sites(camp.model, camp.dtype)
        + optimizer_sites(camp.model, camp.optimizer, camp.dtype)
    )
    print(
        f"  {params/1e6:.2f}M params, {resident/1e6:.2f}M resident {camp.fmt} "
        f"elements ({resident*camp.width/1e9:.2f}e9 bits)"
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

    noise = None
    if args.noise_band:
        print(f"\nmeasuring the seed-to-seed floor over {args.noise_band} extra seeds ...")
        noise = measure_noise_band(args, args.noise_band)
        print(
            f"  golden losses {['%.6f' % x for x in noise['losses']]}\n"
            f"  pairwise loss spread: median {noise['loss_spread_median']:.4e}  "
            f"max {noise['loss_spread_max']:.4e}\n"
            f"  pairwise token disagreement: median "
            f"{int(noise['tokens_differing_median'])} of {noise['tokens_total']}\n"
            f"  reported as scale only. It is NOT the corruption threshold; see "
            f"measure_noise_band.__doc__"
        )

    rng = stream(args.seed, "sdc_campaign.trials")
    per_bit: dict[int, list[Trial]] = defaultdict(list)
    per_bit_def: dict[int, list[Trial]] = defaultdict(list)
    defended: list[Trial] = []

    def one(bit: int) -> None:
        """Run this bit once per arm, on the SAME site.

        Pairing is what makes the two arms comparable, so it is enforced rather
        than assumed: the RNG state is rewound between arms and the two trials
        are asserted to have landed on the same tensor. An unpaired comparison
        would attribute site-to-site variance to the detector.
        """
        if args.arm != "paired":
            t = camp.trial(args.target, bit, rng, args.arm)
            (defended if args.arm == ARM_DEFENDED else camp.trials).append(t)
            (per_bit_def if args.arm == ARM_DEFENDED else per_bit)[bit].append(t)
            return
        state = rng.bit_generator.state
        a = camp.trial(args.target, bit, rng, ARM_UNDEFENDED)
        rng.bit_generator.state = state
        b = camp.trial(args.target, bit, rng, ARM_DEFENDED)
        if a.site != b.site:
            raise RuntimeError(
                f"pairing broke: undefended hit {a.site!r}, defended hit {b.site!r}"
            )
        camp.trials.append(a)
        per_bit[bit].append(a)
        defended.append(b)
        per_bit_def[bit].append(b)

    def progress(bit: int, done: int, total: int) -> None:
        u = summarise(per_bit[bit]) if per_bit[bit] else None
        d = summarise(per_bit_def[bit]) if per_bit_def[bit] else None
        head = f"  bit {bit:>2} ({bit_field(bit, camp.dtype):<8}) "
        if u and d:
            print(
                head + f"SDC undef {u['sdc_pct']:>5.1f}%  def {d['sdc_pct']:>5.1f}%  "
                f"(repaired {d['repaired_pct']:>5.1f}%  caught {d['caught_pct']:>5.1f}%)"
                f"   [{done}/{total}]"
            )
        else:
            x = u or d
            assert x is not None
            print(
                head + f"masked {x['masked_pct']:>5.1f}%  "
                f"detected {x['detected_pct']:>5.1f}%  SDC {x['sdc_pct']:>5.1f}%  "
                f"SDC-crit {x['sdc_critical_pct']:>5.1f}%   [{done}/{total}]"
            )

    if args.sweep == "bits":
        bits = (
            [int(b) for b in args.bits.split(",") if b.strip() != ""]
            if args.bits
            else list(range(camp.width))
        )
        total = len(bits) * args.trials
        print(
            f"\nsweeping {len(bits)} bit positions x {args.trials} trials = {total}"
            f"  [{camp.fmt}, arm={args.arm}]"
        )
        done = 0
        for bit in bits:
            for _ in range(args.trials):
                one(bit)
                done += 1
            progress(bit, done, total)
    else:
        print(
            f"\n{args.trials} trials, bit drawn from the {args.bit_model!r} model"
            f"  [{camp.fmt}, arm={args.arm}]"
        )
        for i in range(args.trials):
            bit = (
                int(rng.integers(0, camp.width))
                if args.bit_model == "uniform"
                else sample_bit_position(rng, camp.width)
            )
            one(bit)
            if (i + 1) % 200 == 0:
                x = summarise(camp.trials or defended)
                print(
                    f"  [{i+1}/{args.trials}] masked {x['masked_pct']:.1f}%  "
                    f"detected {x['detected_pct']:.1f}%  SDC {x['sdc_pct']:.1f}%  "
                    f"SDC-crit {x['sdc_critical_pct']:.1f}%"
                )

    undef_trials = camp.trials if camp.trials else defended
    overall = summarise(undef_trials)
    overall_def = summarise(defended) if defended else None
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 76)
    print(
        f"OVERALL  ({args.target} target, {args.mode} mode, {args.sweep} sweep, "
        f"{camp.fmt}, {str(camp.device)})"
    )
    print("=" * 76)
    if args.arm == "paired" and overall_def is not None:
        # The delta is the product. Print it as a delta, not as two tables the
        # reader has to subtract in their head.
        print(table([("undefended", overall), ("defended", overall_def)], "arm"))
        print()
        print(f"  SDC        {overall['sdc_pct']:.1f}%  ->  {overall_def['sdc_pct']:.1f}%"
              f"   ({overall['sdc']} -> {overall_def['sdc']} of {overall['trials']})")
        print(f"  SDC-crit   {overall['sdc_critical_pct']:.1f}%  ->  "
              f"{overall_def['sdc_critical_pct']:.1f}%"
              f"   ({overall['sdc_critical']} -> {overall_def['sdc_critical']})")
        print(f"  repaired   {overall_def['repaired_pct']:.1f}%   "
              f"caught {overall_def['caught_pct']:.1f}%")
        caught_frac = (
            100.0 * (overall["sdc"] - overall_def["sdc"]) / overall["sdc"]
            if overall["sdc"] else 0.0
        )
        print(f"  share of undefended SDC removed by the defended arm: {caught_frac:.1f}%")
    else:
        print(table([("all bits", overall)], "set"))
    if len(per_bit) > 1:
        print()
        by_field: dict[str, list[Trial]] = defaultdict(list)
        for t in undef_trials:
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
    quiet = [t for t in undef_trials if t.outcome == SDC]
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
    n_run = len(camp.trials) + len(defended)
    print(f"\n{n_run} trials in {elapsed:.1f}s")

    if args.json:
        out = {
            "config": {
                "sweep": args.sweep,
                "mode": args.mode,
                "arm": args.arm,
                "dtype": args.dtype,
                "format": format_name(DTYPES[args.dtype]),
                "bit_width": bit_width(DTYPES[args.dtype]),
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
            "noise_band": noise,
            "overall": overall,
            "overall_defended": overall_def,
            "by_bit": {str(b): summarise(per_bit[b]) for b in sorted(per_bit)},
            "by_bit_defended": {
                str(b): summarise(per_bit_def[b]) for b in sorted(per_bit_def)
            },
            "trials_defended": [t.as_record() for t in defended],
            "by_field": {
                f: summarise([t for t in undef_trials if t.field == f])
                for f in ("sign", "exponent", "mantissa")
                if any(t.field == f for t in undef_trials)
            },
            "trials": [t.as_record() for t in undef_trials],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
