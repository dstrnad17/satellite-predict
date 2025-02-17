import numpy as np
from .arv import arv

def stats(boots, outputs):
  arvs = {}
  for boot in boots: # Loop over each bootstrap repetition
    # result = {actual: df, nn1: df, nn3: df, lr: df}
    models = list(boot.keys())
    models.remove('actual')
    for model in models:
      if model not in arvs:
        arvs[model] = {}
      for output in outputs:
        if output not in arvs[model]:
          arvs[model][output] = []
        arvs[model][output].append(arv(boot['actual'][output], boot[model]['predicted'][output]))

  stats = {}
  for output in arvs[model]:
    stats[output] = {}
    for model in arvs:
      stats[output][model] = {}
      stats[output][model]['mean'] = np.mean(arvs[model][output])
      stats[output][model]['std'] = None
      if len(arvs[model][output]) > 19:
        stats[output][model]['std'] = np.std(arvs[model][output], ddof=1)

  return stats