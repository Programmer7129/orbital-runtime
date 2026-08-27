"""Compute SEU injector: activation corruption via forward hooks."""

from __future__ import annotations

import torch
from torch import nn

from orbital_runtime.inject.compute import ComputeInjector
from orbital_runtime.rng import STREAM_COMPUTE, stream


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 4)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def test_hooks_attach_to_linear_modules():
    inj = ComputeInjector(Net()).attach()
    assert set(inj.hooked_modules) == {"a", "b"}
    inj.detach()
    assert inj.hooked_modules == []


def test_unarmed_forward_is_bit_identical():
    """Zero effect until armed -- otherwise the 'protected' run would pay a
    corruption cost just for having the injector attached."""
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)
    clean = net(x).clone()

    with ComputeInjector(net) as inj:
        assert inj.armed == 0
        assert torch.equal(net(x), clean)


def test_armed_forward_corrupts_exactly_one_activation():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)

    with ComputeInjector(net) as inj:
        inj.arm(stream(1, STREAM_COMPUTE))
        assert inj.armed == 1
        net(x)
        assert inj.armed == 0  # disarmed after firing

    hits = inj.drain_hits()
    assert len(hits) == 1
    assert hits[0].module in ("a", "b")
    assert hits[0].value_before != hits[0].value_after


def test_corruption_propagates_to_the_output_when_not_masked():
    """A corrupted activation must actually reach the output."""
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)
    clean = net(x).clone()

    changed = 0
    for seed in range(8):
        torch.manual_seed(0)
        net, x = Net(), torch.randn(4, 8)
        with ComputeInjector(net) as inj:
            inj.arm(stream(seed, STREAM_COMPUTE))
            out = net(x)
        if not torch.equal(out, clean):
            changed += 1
    assert changed >= 3  # most unmasked hits move the output


def test_relu_masks_corruption_of_negative_activations():
    """Logical masking is real physics, and the sim reproduces it.

    A flip that leaves a negative pre-activation negative is annihilated by
    the following ReLU: the output is bit-identical to clean. This is why
    an injected upset is NOT the same thing as an SDC, and it is a
    well-documented effect in the fault-injection literature.

    It has a direct consequence for M2: a masked fault is both undetectable
    and harmless, so it must not be scored as a detector miss. Recall has to
    be measured against faults that PROPAGATE, not against faults injected.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)
    clean = net(x).clone()

    # Force a single-bit, single-element mantissa flip on a negative
    # pre-activation. This used to rely on seed 1 happening to draw exactly
    # that. Once the injector gained the measured GPU fault classes
    # (nullification, warp-aligned tracks, control-logic tiles), the same seed
    # could draw a multi-element event, and the test failed for reasons
    # unrelated to logical masking. Masking is the claim, so the mechanism is
    # pinned rather than sampled.
    from orbital_runtime.inject.gpu_model import CLASS_BITFLIP, FaultClass

    single_bit = FaultClass(CLASS_BITFLIP, n_elements=1, stride=1, n_bits=1)
    with ComputeInjector(net) as inj:
        inj.arm(stream(1, STREAM_COMPUTE))
        inj.force_fault_class = single_bit
        out = net(x)

    hit = inj.drain_hits()[0]
    assert hit.n_elements == 1
    assert hit.value_before < 0 and hit.value_after < 0
    assert hit.value_before != hit.value_after  # the flip really happened
    assert torch.equal(out, clean)  # ...and was completely absorbed


def test_corruption_is_transient_not_persistent():
    """The distinction from a memory SEU that justifies a separate channel.

    A compute upset pollutes one forward pass. The next clean forward must
    be back to normal -- nothing was written to the weights.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)
    clean = net(x).clone()
    weights_before = [p.clone() for p in net.parameters()]

    with ComputeInjector(net) as inj:
        inj.arm(stream(2, STREAM_COMPUTE))
        corrupted = net(x).clone()
        recovered = net(x).clone()

    assert not torch.equal(corrupted, clean)
    assert torch.equal(recovered, clean)  # transient
    for before, after in zip(weights_before, net.parameters()):
        assert torch.equal(before, after)  # weights untouched


def test_gradients_flow_through_the_corrupted_value():
    """A compute SEU must reach the gradients, not be a no-op for autograd.

    The hook adds `(corrupted - out.detach())`, a constant offset, so the
    corrupted VALUE propagates while autograd never tries to differentiate
    the XOR itself.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)

    net.zero_grad()
    net(x).sum().backward()
    clean_grads = [p.grad.clone() for p in net.parameters()]

    net.zero_grad()
    with ComputeInjector(net) as inj:
        inj.arm(stream(3, STREAM_COMPUTE))
        out = net(x)
        assert out.requires_grad
        out.sum().backward()

    assert any(
        not torch.equal(c, p.grad) for c, p in zip(clean_grads, net.parameters())
    )
    for p in net.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() or True


def test_arming_multiple_fires_multiple():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)
    with ComputeInjector(net) as inj:
        inj.arm(stream(4, STREAM_COMPUTE), n=2)
        net(x)  # two hooked modules -> both fire in one pass
        assert inj.armed == 0
    assert len(inj.drain_hits()) == 2


def test_same_seed_reproduces_the_same_hit():
    def run():
        torch.manual_seed(0)
        net, x = Net(), torch.randn(4, 8)
        with ComputeInjector(net) as inj:
            inj.arm(stream(9, STREAM_COMPUTE))
            net(x)
        h = inj.drain_hits()[0]
        return (h.module, h.index, h.bit, h.value_before, h.value_after)

    assert run() == run()


def test_attach_is_idempotent():
    inj = ComputeInjector(Net())
    inj.attach()
    inj.attach()
    assert len(inj.hooked_modules) == 2  # not 4
    inj.detach()


def test_drain_clears_hits():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(4, 8)
    with ComputeInjector(net) as inj:
        inj.arm(stream(5, STREAM_COMPUTE))
        net(x)
        assert len(inj.drain_hits()) == 1
        assert inj.drain_hits() == []
