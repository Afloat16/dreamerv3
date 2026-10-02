import numpy as np
import pytest

from native_replay import Replay, selectors


@pytest.mark.parametrize('selector', [selectors.Uniform(), selectors.Fifo(),
                                     selectors.Prioritized(), selectors.Recency(np.ones(8))])
def test_empty_explicit_selector_is_preserved(selector):
  assert len(selector) == 0
  replay = Replay(length=1, selector=selector)
  assert replay.sampler is selector
  replay.add({'value': np.array(1)})
  assert len(selector) == 1


def test_priority_feedback_reaches_the_requested_sampler():
  selector = selectors.Prioritized(initial=1.0)
  replay = Replay(length=1, selector=selector)
  replay.add({'value': np.array(1)})
  replay.add({'value': np.array(2)})
  data = replay.sample(batch=2)
  data = {'stepid': data['stepid'], 'priority': np.array([[7.], [7.]])}
  updated = set(step.tobytes() for step in data['stepid'].reshape(-1, 20))
  replay.update(data)
  assert all(selector.prios[stepid] == 7. for stepid in updated)


def test_omitted_selector_still_uses_seeded_uniform_sampling():
  first = Replay(length=1, seed=5)
  second = Replay(length=1, seed=5, selector=None)
  for replay in [first, second]:
    assert isinstance(replay.sampler, selectors.Uniform)
    for value in range(4):
      replay.add({'value': np.array(value)})
  np.testing.assert_array_equal(first.sample(20)['value'], second.sample(20)['value'])
