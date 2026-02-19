import torch

from src.instrumentation.dormancy import (
    tau_dormant_mask,
    grad_quantile_dormant_mask,
    overlap_coefficient,
    dormancy_events,
)


def test_tau_dormant_mask_basic():
    # Construct activations: neuron0 always ~0, neuron1 active
    B = 100
    a0 = torch.zeros(B, 1)
    a1 = torch.ones(B, 1)
    acts = torch.cat([a0, a1], dim=1)  # [B,2]

    mask, score = tau_dormant_mask(acts, tau=0.5, eps=1e-8)
    assert mask.shape == (2,)
    assert bool(mask[0].item()) is True
    assert bool(mask[1].item()) is False
    assert score.numel() == 2


def test_grad_quantile_dormant_mask_selects_bottom_q():
    g = torch.tensor([0.0, 1.0, 2.0, 3.0])
    mask, score = grad_quantile_dormant_mask(g, q=0.25)  # bottom 1 element
    assert mask.sum().item() >= 1
    assert bool(mask[0].item()) is True  # smallest is dormant
    assert score.shape == g.shape


def test_overlap_and_events():
    prev = torch.tensor([True, False, True, False])
    curr = torch.tensor([True, True, False, False])

    ov = overlap_coefficient(prev, curr)
    # prev set size=2, curr set size=2, intersection size=1 => 1/min(2,2)=0.5
    assert abs(ov - 0.5) < 1e-6

    death, revival = dormancy_events(prev, curr)
    # death: active->dormant: index1 only => 1/4
    # revival: dormant->active: index2 only => 1/4
    assert abs(death - 0.25) < 1e-6
    assert abs(revival - 0.25) < 1e-6
