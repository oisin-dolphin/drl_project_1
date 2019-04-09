import torch.nn as nn
import torch
from src.agent import Agent


def test_soft_update():

    net_1 = nn.Linear(1, 1, bias=False)
    next(net_1.parameters()).data.copy_(torch.ones(1))
    net_2 = nn.Linear(1, 1, bias=False)
    next(net_2.parameters()).data.copy_(torch.zeros(1))

    Agent.soft_update(net_1, net_2, 0.5)
    assert net_1(torch.ones(1)) == torch.ones(1)
    assert net_2(torch.ones(1)) == torch.ones(1) * 0.5
