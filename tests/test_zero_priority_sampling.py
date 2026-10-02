import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'selectors_under_test', Path(__file__).parents[1] / 'embodied/core/selectors.py')
selectors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selectors)


class PathRng:
  def __init__(self, path):
    self.path = iter(path)
    self.probability = 1.0

  def choice(self, choices, p):
    index = next(self.path)
    self.probability *= p[index]
    return np.int64(index)


def leaf_probabilities(tree):
  probabilities = {}
  pending = [(tree.root, [])]
  while pending:
    node, path = pending.pop()
    if hasattr(node, 'children'):
      pending.extend((child, path + [index])
                     for index, child in enumerate(node.children))
    else:
      tree.rng = PathRng(path)
      assert tree.sample() == node.key
      probabilities[node.key] = tree.rng.probability
  return probabilities


@pytest.mark.parametrize('branching', [2, 3, 5, 16])
@pytest.mark.parametrize('size', [3, 17, 50])
def test_zero_priorities_are_uniform_over_items(branching, size):
  tree = selectors.SampleTree(branching)
  for key in range(size):
    tree.insert(key, 0.0)
  assert leaf_probabilities(tree) == pytest.approx(
      dict.fromkeys(range(size), 1 / size))


@pytest.mark.parametrize('branching', [2, 3, 5, 16])
def test_zero_fallback_survives_removal_and_reinsertion(branching):
  tree = selectors.SampleTree(branching)
  for key in range(51):
    tree.insert(key, 1.0)
  for key in range(0, 51, 3):
    tree.remove(key)
  for key in range(51, 58):
    tree.insert(key, 1.0)
  for key in tree.entries:
    tree.update(key, 0.0)
  expected = dict.fromkeys(tree.entries, 1 / len(tree))
  assert leaf_probabilities(tree) == pytest.approx(expected)
  for key in list(tree.entries):
    tree.remove(key)
  tree.insert('new', 0.0)
  assert leaf_probabilities(tree) == {'new': 1.0}


def test_zero_on_sample_can_reach_the_uniform_fallback():
  selector = selectors.Prioritized(
      initial=1.0, zero_on_sample=True, branching=2)
  for key in range(3):
    selector[key] = [bytes([key])]
  assert {selector() for _ in range(3)} == {0, 1, 2}
  assert selector.tree.root.uprob == 0.0
  assert leaf_probabilities(selector.tree) == pytest.approx(
      dict.fromkeys(range(3), 1 / 3))


def test_positive_priorities_preserve_the_normalized_measure():
  tree = selectors.SampleTree(branching=3)
  weights = {key: float(key % 4) for key in range(17)}
  for key, weight in weights.items():
    tree.insert(key, weight)
  assert leaf_probabilities(tree) == pytest.approx(
      {key: weight / sum(weights.values()) for key, weight in weights.items()})
