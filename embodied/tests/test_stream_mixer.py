import pickle
import unittest

import numpy as np

from embodied.core import streams


class Counter:

  def __init__(self, tag):
    self.tag = tag
    self.index = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = (self.tag, self.index)
    self.index += 1
    return value

  def save(self):
    return self.index

  def load(self, state):
    self.index = state


def make_mixer(seed=5):
  return streams.Mixer(
      {'alpha': Counter('alpha'), 'beta': Counter('beta')},
      {'alpha': 1, 'beta': 3}, seed=seed)


class TestMixer(unittest.TestCase):

  def test_iterates_weighted_sources(self):
    mixer = iter(make_mixer())
    samples = [next(mixer) for _ in range(4000)]
    fraction = np.mean([key == 'beta' for key, _ in samples])
    self.assertAlmostEqual(fraction, 0.75, delta=0.025)
    for key in ('alpha', 'beta'):
      self.assertEqual(
          [index for tag, index in samples if tag == key],
          list(range(sum(tag == key for tag, _ in samples))))

  def test_checkpoint_restores_choices_and_source_positions(self):
    original = iter(make_mixer(seed=73))
    for _ in range(57):
      next(original)
    state = pickle.loads(pickle.dumps(original.save()))
    expected = [next(original) for _ in range(300)]
    restored = iter(make_mixer(seed=999))
    restored.load(state)
    self.assertEqual([next(restored) for _ in range(300)], expected)

  def test_seed_controls_sequence(self):
    a, b = iter(make_mixer(seed=12)), iter(make_mixer(seed=12))
    self.assertEqual([next(a) for _ in range(100)], [next(b) for _ in range(100)])


if __name__ == '__main__':
  unittest.main()
