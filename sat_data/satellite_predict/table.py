import os
import pandas as pd

def table(stats, directory, **kwargs):

  columns = ['Removed Input', 'Model']
  for output in kwargs['outputs']:
    columns.append(f"{output} ARV")
    columns.append(f"{output} ARV SE")

  # Check that the number of loo repetitions is consistent across removed inputs
  n_loo = None
  for removed_input in stats.keys():
    if n_loo is not None and len(stats[removed_input]) != n_loo:
      raise ValueError("Number of loo repetitions is not consistent across removed inputs.")
    n_loo = len(stats[removed_input])

  for loo_idx in range(n_loo):
    table = []
    for removed_input in stats.keys():
      for model in stats[removed_input][loo_idx][output]:
        row = [removed_input, model]
        for output in stats[removed_input][loo_idx].keys():
          row.append(stats[removed_input][loo_idx][output][model]['mean'])
          row.append(stats[removed_input][loo_idx][output][model]['std'])
        table.append(row)

    # Combine all the summary data into one table
    table = pd.DataFrame(table, columns=columns)

    table_file = os.path.join(directory, f'loo_{loo_idx}.md')
    table.to_markdown(table_file, index=False, floatfmt=".3f")
    print(f"    Wrote {table_file}")
