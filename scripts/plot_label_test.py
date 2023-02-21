
# %%

import matplotlib.pyplot as plt
from gf3d.plot.util import plot_label

plt.ion()

# %%
plt.close('all')

plt.rcParams["font.family"] = "monospace"
plt.rcParams["axes.edgecolor"] = 'k'

fig = plt.figure(figsize=(6, 5))
ax = plt.axes()
ax.tick_params(which='both',
               right=True, left=True, top=True, bottom=True,
               labelbottom=False, labeltop=False,
               labelleft=False, labelright=False)
for i in range(0, 25):
    dist = 0.025 if i % 2 == 0 else 0.01
    box = True if i % 2 == 0 else False
    plot_label(ax, f'{i}', location=i, box=box, dist=dist)

fig.suptitle('Even: box and dist=0.025 -- odd dist=0.01')
