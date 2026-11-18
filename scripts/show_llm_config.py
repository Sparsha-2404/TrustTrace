import yaml
from pathlib import Path
p = Path(__file__).resolve().parents[1] / 'config.yaml'
cfg = yaml.safe_load(open(p, 'r', encoding='utf-8'))
print('LLM config:')
for k, v in (cfg.get('llm') or {}).items():
    print(f"  {k}: {v}")
