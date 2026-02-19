import torch
import torch.nn as nn
import torch.optim as optim

from src.redo.recycle import redo_recycle_linear_pair


def test_redo_recycle_linear_pair_changes_incoming_and_zeros_outgoing():
    torch.manual_seed(0)

    lin_in = nn.Linear(5, 4, bias=True)   # out=4 neurons
    lin_out = nn.Linear(4, 3, bias=True)

    # make copies
    w_in_before = lin_in.weight.detach().clone()
    b_in_before = lin_in.bias.detach().clone()
    w_out_before = lin_out.weight.detach().clone()

    # optimizer to create state
    opt = optim.Adam(list(lin_in.parameters()) + list(lin_out.parameters()), lr=1e-3)
    # run dummy backward to populate moments
    x = torch.randn(8, 5)
    y = lin_out(torch.relu(lin_in(x))).sum()
    opt.zero_grad()
    y.backward()
    opt.step()

    dormant = torch.tensor([True, False, True, False])  # recycle neurons 0 and 2
    k = redo_recycle_linear_pair(
        lin_in=lin_in,
        lin_out=lin_out,
        dormant_mask=dormant,
        optimizer=opt,
        init_mode="xavier_uniform",
        reset_bias=True,
        outgoing="zero",
        max_frac=1.0,
    )
    assert k == 2

    # incoming rows changed for recycled indices
    assert not torch.allclose(lin_in.weight[0], w_in_before[0])
    assert not torch.allclose(lin_in.weight[2], w_in_before[2])
    # incoming rows unchanged for non-recycled
    assert torch.allclose(lin_in.weight[1], w_in_before[1])
    assert torch.allclose(lin_in.weight[3], w_in_before[3])

    # bias reset for recycled
    assert lin_in.bias[0].item() == 0.0
    assert lin_in.bias[2].item() == 0.0

    # outgoing cols zeroed for recycled
    assert torch.all(lin_out.weight[:, 0] == 0.0)
    assert torch.all(lin_out.weight[:, 2] == 0.0)
    # non-recycled cols unchanged (approximately)
    assert torch.allclose(lin_out.weight[:, 1], w_out_before[:, 1])
    assert torch.allclose(lin_out.weight[:, 3], w_out_before[:, 3])
