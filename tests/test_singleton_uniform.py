import numpy as np
import pytest

from native_replay import Replay, selectors


def test_uniform_can_be_emptied_and_reused():
  selector = selectors.Uniform()
  for key in range(10):
    selector[key] = []
    assert selector() == key
    del selector[key]
    assert len(selector) == 0
    assert selector.indices == {}
    assert selector.keys == []


def test_unknown_key_preserves_the_live_singleton():
  selector = selectors.Uniform()
  selector['live'] = []
  with pytest.raises(KeyError):
    del selector['missing']
  assert selector() == 'live'


@pytest.mark.parametrize('length', [1, 3])
def test_capacity_one_replay_keeps_the_newest_sequence(length):
  replay = Replay(length=length, capacity=1, chunksize=4)
  for value in range(20):
    replay.add({'value': np.array(value)})
    if value + 1 >= length:
      assert len(replay) == 1
      actual = replay.sample(batch=2)['value']
      expected = np.tile(np.arange(value + 1 - length, value + 1), (2, 1))
      np.testing.assert_array_equal(actual, expected)
