import os
import matplotlib.pyplot as plt

def savefig(plt, save_path):
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

  plt.savefig(f"{save_path}")
  print(f"      Wrote:   {save_path}")
  plt.close()

def plot(boots, save_path, stats):

  #print(stats)
  boot_num = 0
  actual = boots[boot_num]['actual']
  outputs = actual.columns[1:] # Exclude timestamp

  models = list(boots[0].keys())
  models.remove('actual')
  for model in models:
    predicted = boots[boot_num][model]['predicted']
    plt.figure(figsize=(8.5, 12), facecolor='white')
    gs = plt.gcf().add_gridspec(len(outputs))
    axs = gs.subplots(sharex=True)
    for i, output in enumerate(outputs):
      axs[i].plot(actual['timestamp'], actual[output], linestyle='-', color='black', label='Actual')
      axs[i].plot(predicted['timestamp'], predicted[output], linestyle='-', color='red', label=model)
      if i == 0:
        axs[i].legend()
      axs[i].set_ylabel(output)
      axs[i].grid(True)
      #datetick()

    plt.tight_layout()
    savefig(plt, f"{save_path}_boot_{boot_num}_{model}.png")

    if 'epochs' in boots[boot_num][model]:
      plt.figure(figsize=(8.5, 8.5), facecolor='white')
      plt.semilogy(boots[boot_num][model]['epochs'])
      plt.grid(True)
      plt.ylabel('ARV')
      plt.xlabel('Epoch')
      plt.tight_layout()
      savefig(plt, f"{save_path}_boot_{boot_num}_{model}_epochs.png")
