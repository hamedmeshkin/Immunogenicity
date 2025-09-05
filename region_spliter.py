import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

rng = np.random.default_rng(7)  # reproducible
# Rectangles: (x, y, width, height)
rects = [
    (0.0, 0.0, 2.0, 5.0),
    (2.6, 0.0, 2.0, 5.0),
    (5.3, 0.0, 2.0, 5.0),
    (7.9, 0.0, 2.0, 5.0),
]

# For manual control, you could instead do e.g.:
totals = [36, 21, 77, 29]
red_fracs = [0.58, 0.86, 0.82, 0.90]

fig, ax = plt.subplots(figsize=(10, 5))

# Collect points for legend once
plotted_blue = plotted_red = False

for idx, ((x, y, w, h), n_total, p_red) in enumerate(zip(rects, totals, red_fracs)):
    # draw rectangle
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1.8, edgecolor='#4aa3ff'))

    # decide counts
    n_red = int(round(n_total * p_red))
    n_blue = n_total - n_red

    eps=0.1
    # sample uniformly inside this rectangle
    xs_red  = rng.uniform(x+eps, x + w-eps, size=n_red)
    ys_red  = rng.uniform(y+eps, y + h-eps, size=n_red)
    xs_blue = rng.uniform(x+eps, x + w-eps, size=n_blue)
    ys_blue = rng.uniform(y+eps, y + h-eps, size=n_blue)

    # plot points
    if n_blue > 0:
        ax.scatter(xs_blue, ys_blue, s=16, c='tab:blue', alpha=0.9,
                   label='Blue' if not plotted_blue else None)
        plotted_blue = True
    if n_red > 0:
        ax.scatter(xs_red, ys_red, s=16, c='tab:red', alpha=0.9,
                   label='Red' if not plotted_red else None)
        plotted_red = True

    # annotate counts on top of the box
    ax.text(x + w/2, y + h + 0.030, f'Cluster {idx+1}',
            ha='center', va='bottom', fontsize=9)

ax.set_aspect('equal', adjustable='box')
# expand axes a bit around the outer boxes
x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
x1 = max(r[0] + r[2] for r in rects); y1 = max(r[1] + r[3] for r in rects)
padx = 0.05 * (x1 - x0); pady = 0.05 * (y1 - y0)
ax.set_xlim(x0 - padx, x1 + padx); ax.set_ylim(y0 - pady, y1 + pady)

ax.set_axis_off()
ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1), borderaxespad=0.)
ax.grid(alpha=0.25)
plt.tight_layout()

plt.savefig('plot/Split_' + File_name + '.png')
