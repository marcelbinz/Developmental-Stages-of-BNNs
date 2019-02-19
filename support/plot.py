import torch
import numpy as np
import matplotlib.pyplot as plt

results = torch.load('support_results.pt')
results = torch.cat((torch.ones(results.size(0), results.size(1), 1), results), dim=-1)
results = results * 100
std = results.std(0)
results = results.mean(0)

results[:, 0] = 50
plt.plot(100 - results[:, 0].numpy(), color='#929591', linestyle='--')
plt.plot(100 - results[:, 4].numpy(), color='grey')
plt.plot(100 - results[:, 3].numpy(), color='#87ae73')
plt.plot(100 - results[:, 2].numpy(), color='#5b7c99')
plt.plot(100 - results[:, 1].numpy(), color='#ff796c')

plt.fill_between(np.arange(6), 100 - results[:, 4].numpy() - std[:, 4].numpy(), 100 - results[:, 4].numpy() + std[:, 4].numpy(), color='grey', alpha=0.3, linewidth=0)
plt.fill_between(np.arange(6), 100 - results[:, 3].numpy() - std[:, 3].numpy(), 100 - results[:, 3].numpy() + std[:, 3].numpy(), color='#87ae73', alpha=0.3, linewidth=0)
plt.fill_between(np.arange(6), 100 - results[:, 2].numpy() - std[:, 2].numpy(), 100 - results[:, 2].numpy() + std[:, 2].numpy(), color='#5b7c99', alpha=0.3, linewidth=0)
plt.fill_between(np.arange(6), 100 - results[:, 1].numpy() - std[:, 1].numpy(), 100 - results[:, 1].numpy() + std[:, 1].numpy(), color='#ff796c', alpha=0.3, linewidth=0)

plt.text(torch.argmax(results[:, 4] > 91.4).item(), -15, '12.5 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'}, size=8)
plt.text(torch.argmax(results[:, 3] > 91.4).item(), -15, '6.5 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'}, size=8)
plt.text(torch.argmax(results[:, 2] > 91.4).item(), -15, '5.0 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'}, size=8)
plt.text(torch.argmax(results[:, 1] > 91.4).item(), -15, '3.0 month', {'color': '#9e003a', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'}, size=8)

plt.ylim(-5.0, 80.5)
plt.xticks(np.arange(6), [256, 512, 1024, 2048, 4096, 8192])
plt.xlim(0, 5)
plt.legend(['baseline', 'proportional', 'amount','side', 'top'])
plt.ylabel('Error (%)')
plt.savefig('support.pdf')
