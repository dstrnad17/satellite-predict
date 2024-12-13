import os
import numpy as np
import pandas as pd

from .arv import arv

def table(**kwargs):

  tag = kwargs['tag']

  directory = kwargs.get('results_directory', './results')
  directory = os.path.join(directory, tag)

  summary = {}

  # Iterate through subdirectories under each pattern
  for removed_input in os.listdir(directory):
    removed_input_dir = os.path.join(directory, removed_input)

    if not os.path.isdir(removed_input_dir):
      continue

    print(f"    Processing removed input dir: {removed_input_dir}")

    loo_files = []
    summary[removed_input] = []
    loo_dir = os.path.join(removed_input_dir, 'loo')
    for file_name in sorted(os.listdir(loo_dir)):
      if not file_name.endswith('.pkl'):
        continue
      file_path = os.path.join(loo_dir, file_name)

      print(f"      Reading: {file_path}")
      boots = pd.read_pickle(file_path) # Array of bootstrap results

      # Each element of summary[removed_input] is a loo repetition
      summary[removed_input].append(boot_stats(boots, kwargs['outputs']))

      loo_files.append(file_path)


  columns = ['Removed Input', 'Model']
  for output in kwargs['outputs']:
    columns.append(f"{output} ARV")
    columns.append(f"{output} ARV SE")

  #import pprint
  #pprint.pp(summary, indent=2)

  for loo_idx, loo_file in enumerate(loo_files):
    table = []
    for removed_input in summary:
      for model in summary[removed_input][loo_idx][output]:
        row = [removed_input, model]
        for output in summary[removed_input][loo_idx].keys():
          row.append(summary[removed_input][loo_idx][output][model]['mean'])
          row.append(summary[removed_input][loo_idx][output][model]['std'])
        table.append(row)

    # Combine all the summary data into one table
    table = pd.DataFrame(table, columns=columns)

    summary_file = loo_file.replace('.pkl', '.md')
    table.to_markdown(summary_file, index=False, floatfmt=".3f")
    print(f"    Wrote table to {summary_file}")


def boot_stats(boots, outputs):
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
        arvs[model][output].append(arv(boot['actual'][output], boot[model][output]))

  stats = {}
  for output in arvs[model]:
    stats[output] = {}
    for model in arvs:
      stats[output][model] = {}
      stats[output][model]['mean'] = np.mean(arvs[model][output])
      stats[output][model]['std'] = None
      if len(arvs[model][output]) > 1:
        stats[output][model]['std'] = np.std(arvs[model][output], ddof=1)

  return stats