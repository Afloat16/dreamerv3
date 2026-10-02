import importlib.util
from pathlib import Path
import sys
import types

root = Path(__file__).parents[1] / 'embodied/core'
package_name = '_native_replay_core'
package = types.ModuleType(package_name)
package.__path__ = [str(root)]
sys.modules[package_name] = package
spec = importlib.util.spec_from_file_location(package_name + '.replay', root / 'replay.py')
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)
Replay = replay.Replay
selectors = replay.selectors
