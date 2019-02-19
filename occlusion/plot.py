import torch
import numpy as np
import matplotlib.pyplot as plt

results = torch.load('occlusion_results.pt')

results = results * 100
std = results.std(0) * 1.0
results = results.mean(0)

results[:, 0] = 50
plt.plot(100 - results[:, 0].numpy(), color='#929591', linestyle='--')
plt.plot(100 - results[:, 3].numpy(), color='#87ae73')
plt.plot(100 - results[:, 2].numpy(), color='#5b7c99')
plt.plot(100 - results[:, 1].numpy(), color='#ff796c')

plt.fill_between(np.arange(6), 100 - results[:, 3].numpy() - std[:, 3].numpy(), 100 - results[:, 3].numpy() + std[:, 3].numpy(), color='#87ae73', alpha=0.3, linewidth=0)
plt.fill_between(np.arange(6), 100 - results[:, 2].numpy() - std[:, 2].numpy(), 100 - results[:, 2].numpy() + std[:, 2].numpy(), color='#5b7c99', alpha=0.3, linewidth=0)
plt.fill_between(np.arange(6), 100 - results[:, 1].numpy() - std[:, 1].numpy(), 100 - results[:, 1].numpy() + std[:, 1].numpy(), color='#ff796c', alpha=0.3, linewidth=0)

plt.text(torch.argmax(results[:, 3] > 90).item(), -15, '2.5 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'})
plt.text(torch.argmax(results[:, 2] > 90).item(), -15, '3.0 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'})
plt.text(torch.argmax(results[:, 1] > 90).item(), -15, '3.5 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'})


plt.ylim(-5, 80.5)
plt.xticks(np.arange(6), [256, 512, 1024, 2048, 4096, 8192])
plt.xlim(0, 5)
plt.legend(['baseline', 'everything removed','bottom removed', 'top removed'])
plt.ylabel('Error (%)')
#plt.tight_layout()
plt.savefig('occlusion.pdf')
