import pickle
import unittest

import numpy as np

from embodied.core import streams


class Batches:

  def __init__(self):
    self.index = 0

  def __iter__(self):
    return self

  def __next__(self):
    start = self.index * 100
    self.index += 1
    values = np.arange(start, start + 7)[None]
    return {'value': values, 'is_first': np.zeros(values.shape, bool)}

  def save(self):
    return self.index

  def load(self, state):
    self.index = state


def make_stream(contiguous=False):
  return streams.Consec(Batches(), length=2, consec=3, prefix=1, contiguous=contiguous)


class TestConsecCheckpoint(unittest.TestCase):

  def test_resume_at_each_chunk_boundary(self):
    for count in (1, 2, 3, 4, 5, 6):
      for contiguous in (False, True):
        with self.subTest(count=count, contiguous=contiguous):
          original = iter(make_stream(contiguous))
          for _ in range(count):
            next(original)
          saved = pickle.loads(pickle.dumps(original.save()))
          expected = [next(original)['value'].copy() for _ in range(9)]
          restored = make_stream(contiguous)
          restored.load(saved)
          restored = iter(restored)
          for value in expected:
            np.testing.assert_array_equal(next(restored)['value'], value)

  def test_current_batch_prefix_is_preserved(self):
    original = iter(make_stream())
    np.testing.assert_array_equal(next(original)['value'], [[0, 1, 2]])
    saved = pickle.loads(pickle.dumps(original.save()))
    restored = make_stream()
    restored.load(saved)
    restored = iter(restored)
    np.testing.assert_array_equal(next(restored)['value'], [[2, 3, 4]])
    np.testing.assert_array_equal(next(restored)['value'], [[4, 5, 6]])
    np.testing.assert_array_equal(next(restored)['value'], [[100, 101, 102]])

  def test_legacy_checkpoint_starts_at_next_available_batch(self):
    restored = make_stream()
    restored.load({'source': 4, 'index': 2})
    restored = iter(restored)
    np.testing.assert_array_equal(next(restored)['value'], [[400, 401, 402]])


if __name__ == '__main__':
  unittest.main()
