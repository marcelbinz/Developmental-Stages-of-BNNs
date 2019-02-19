import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import seaborn as sns
from scipy.optimize import curve_fit

results = torch.load('ans_results.pt')

results = results * 100
results = np.flip(results.numpy(), 2)

x = np.array([(i+1)/(i) for i in range(9, 0, -1)])

wf = np.zeros((results.shape[0], results.shape[1]))
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        y = results[i, j]
        slope, intercept, _, _, _ = stats.linregress(x, y)
        wf[i, j] = max(0, ((75 - intercept) / slope) - 1)

fig, ax1 = plt.subplots()
color = sns.xkcd_rgb["windows blue"]

x = np.array([2048 * i for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
y = wf.mean(0)

def func(x, a, b, c, d):
    return a * np.exp(-b * (x - c)) + d

init_vals = [1,0,5000,0]
popt, pcov = curve_fit(func, x, y, p0=init_vals)
fit_x = [i for i in range(8000, 33000)]
fit_y = [func(i, *popt) for i in fit_x]

ax1.plot(fit_x, fit_y, color=color, label='_nolegend_')
ax1.scatter([2048 * i for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], wf.mean(0), color=color, marker="+", label='Model')
ax1.tick_params(axis='x', labelcolor=color)
ax1.set_xlim(8000, 33000)
ax1.set_xticks([10000, 15000, 20000, 25000, 30000])
ax1.set_yticks([0, 0.5, 1, 1.5, 2])
ax1.set_ylim(0, 2)
ax1.set_ylabel('Weber fraction')
ax1.set_xlabel('Data-set size', color=color)

x = np.array([3, 4, 5, 6, 21])
y = np.array([0.525, 0.383, 0.229, 0.179, 0.108])

def func(x, a, b, c, d):
    return a * np.exp(-b * (x - c)) + d

init_vals = [1,1,1,-1]
popt, pcov = curve_fit(func, x, y, p0=init_vals)
fit_x = [x * 0.1 for x in range(0, 221)]
fit_y = [func(i, *popt) for i in fit_x]

ax2 = ax1.twiny()
color = sns.xkcd_rgb["pale red"]
ax2.plot(fit_x, fit_y, color=color, label='_nolegend_')
ax2.set_xlim(-1, 22)
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.scatter(x, y, color=color, marker="+", label='Human')
ax2.tick_params(axis='x', labelcolor=color)
ax2.set_xlabel('Age', color=color)
plt.savefig('ans.pdf')
plt.show()
