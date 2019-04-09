from src.memory import ReplayBuffer
import numpy as np

fake_data = {
    "state": np.zeros(5),
    "action": np.zeros(2),
    "reward": np.ones(1),
    "next_state": np.zeros(5),
    "done": np.zeros(0),
}


def test_memory_size():
    rb = ReplayBuffer(action_size=2, buffer_size=1, batch_size=1)
    assert len(rb.memory) == 0
    rb.add(**fake_data)
    assert len(rb.memory) == 1

    rb.add(**fake_data)
    assert len(rb.memory) == 1
