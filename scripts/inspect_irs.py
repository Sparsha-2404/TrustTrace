import importlib
import sys
from pathlib import Path
# Ensure workspace root is on sys.path so local packages import correctly
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

m = importlib.import_module('irs.injection_risk')
print('module_file=', getattr(m, '__file__', None))
print('has_compute_rule_score=', hasattr(m, 'compute_rule_score'))
print('dir_sample=', [k for k in dir(m) if 'compute' in k or 'rule' in k][:20])
