from src.network import QNetwork
import torch


def test_input_output_size():
    net = QNetwork(state_size=10, action_size=5)
    output = net(torch.zeros(10))

    assert output.size(0) == 5
