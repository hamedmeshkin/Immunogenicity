import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from matplotlib.colors import ListedColormap, to_rgb
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap, ScalarMappable
from scipy.cluster.hierarchy import inconsistent, fcluster,linkage
from matplotlib import cm
import matplotlib.patches as patches
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap, ScalarMappable
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from permetrics import ClusteringMetric

def color_by_second_split(Z, y=None, ax=None, leaf_font_size=7,
                          cluster_colors=('tab:blue','tab:orange','tab:green','tab:red'),
                          show_cluster_strip=True, keep_leaf_class_colors=True,
                          class_colors=('blue','red')):

    n = Z.shape[0] + 1
    ax = plt.gca() if ax is None else ax

    # child map for internal nodes (ids: n .. n+n-2)
    children = {n+i: (int(Z[i,0]), int(Z[i,1])) for i in range(n-1)}

    def get_children(node):
        return children[node] if node >= n else ()

    def leaves_under(node):
        out, stack = [], [node]
        while stack:
            k = stack.pop()
            if k < n:
                out.append(k)
            else:
                stack.extend(get_children(k))
        return sorted(out)

    # Root and its two children
    left_root, right_root = int(Z[-1,0]), int(Z[-1,1])

    # Second split: children of each root child (handle leaves gracefully)
    candidates = []
    for rc in (left_root, right_root):
        kids = get_children(rc)
        if kids:
            candidates.extend(kids)          # two children
        else:
            candidates.append(rc)            # if rc is a leaf (rare at top)
    # keep at most four
    candidates = candidates[:4]

    # Build leaf sets for each of the up-to-4 subtrees
    groups = [leaves_under(c) for c in candidates]
    group_sets = [set(g) for g in groups]

    # Precompute membership of every node in exactly one group
    # (root spans multiple groups → gray)
    in_group = np.zeros((len(groups), n + (n-1)), dtype=bool)
    for gi, gset in enumerate(group_sets):
        in_group[gi, list(gset)] = True
    for i in range(n-1):
        node = n + i
        a, b = children[node]
        in_group[:, node] = in_group[:, a] | in_group[:, b]

    def link_color_func(node_id):
        # If a node belongs entirely to one group, use that group's color.
        belongs = [in_group[gi, node_id] for gi in range(len(groups))]
        if sum(belongs) == 1:
            gi = belongs.index(True)
            return cluster_colors[min(gi, len(cluster_colors)-1)]
        return '0.5'  # spans multiple groups (e.g., the root); draw gray

    # Draw dendrogram with our link-color function
    dendro = dendrogram(
        Z, ax=ax, link_color_func=link_color_func,
        color_threshold=0, leaf_font_size=leaf_font_size)

    # Color leaf tick labels by class or by cluster
    leaf_order = dendro['leaves']
    ticks = ax.get_xmajorticklabels()
    if keep_leaf_class_colors and (y is not None):
        y = np.asarray(y).ravel()
        for txt, idx in zip(ticks, leaf_order):
            txt.set_color(class_colors[int(y[idx])])
    else:
        # color by group instead
        leaf_to_group = {}
        for gi, g in enumerate(groups):
            for idx in g:
                leaf_to_group[idx] = gi
        for txt, idx in zip(ticks, leaf_order):
            gi = leaf_to_group.get(idx, 0)
            txt.set_color(cluster_colors[min(gi, len(cluster_colors)-1)])

    if show_cluster_strip:
        ax2 = ax.inset_axes([0, -0.15, 1, 0.04])  # x,y,w,h (axes fraction)
        strip = (y[leaf_order] == 1).astype(int)[None, :]
        ax2.imshow(strip, aspect='auto', cmap=ListedColormap(['blue', 'red']))
        ax2.set_xticks([])
        ax2.set_yticks([])

    return dendro, groups




def plot_boxes_with_ada_and_shapes(groups, ada_values, labels01, rects,
                                   seed=7, point_size=28,
                                   cmap_name='inferno', show_counts=True):
    """
    groups   : (n,) ints in {0,1,2,3} assigning each sample to a box
    ada_values : (n,) float ADA per sample (controls color)
    labels01 : (n,) ints {0,1}  (0=blue class -> circle 'o', 1=red class -> square 's')
    rects    : list of 4 (x,y,w,h) rectangles to scatter points into
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups).astype(int)
    ada    = np.asarray(ada_values, dtype=float)
    y      = np.asarray(labels01, dtype=int)

    # global color normalization so colors are comparable across boxes
    norm = mcolors.Normalize(vmin=np.nanmin(ada), vmax=np.nanmax(ada))
    cmap = get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(10, 5))

    # For legend: proxy artists (shape only; color varies by ADA)
    circle_proxy = plt.scatter([], [], marker='o', s=point_size, c='k', label='Blue (circle)')
    square_proxy = plt.scatter([], [], marker='s', s=point_size, c='k', label='Red  (square)')

    for gi, (x, y0, w, h) in enumerate(rects):
        # frame
        ax.add_patch(patches.Rectangle((x, y0), w, h, fill=False, lw=2, ec='#4aa3ff'))

        idx = np.where(groups == gi)[0]
        if idx.size == 0:
            continue

        # random positions in this box
        xs = rng.uniform(x+0.1, x + w-0.1, size=idx.size)
        ys = rng.uniform(y0+0.1, y0 + h-0.1, size=idx.size)

        # split by class → shape
        idx_blue = idx[labels01[idx] == 0]
        idx_red  = idx[labels01[idx] == 1]

        if idx_blue.size:
            ax.scatter(xs[:idx_blue.size], ys[:idx_blue.size],
                       s=point_size, marker='o',
                       c=ada[idx_blue], cmap=cmap, norm=norm, edgecolors='none')
        if idx_red.size:
            # use the remaining coords for red (or generate fresh ones; either is fine)
            rb = idx_blue.size
            ax.scatter(xs[rb:rb+idx_red.size], ys[rb:rb+idx_red.size],
                       s=point_size, marker='s',
                       c=ada[idx_red], cmap=cmap, norm=norm, edgecolors='none')

        if show_counts:
            n_total = idx.size
            n_blue = (labels01[idx] == 0).sum()
            n_red  = n_total - n_blue
#            ax.text(x + w/2, y0 + h + 0.06*h,
#                    f"N={n} | blue={n_blue} red={n_red}  ",
#                    ha='center', va='bottom', fontsize=9)
            ax.text(x + w/2, y0 + h + 0.25,
            f"{n_blue} out of {n_total} blue\n{n_red} out of {n_total} red",
            ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

            ax.text(x + w/2, y0  - 0.25, f'Cluster {gi+1}',
            ha='center', va='bottom', fontsize=9)

            if gi==0:
                if len(File_name.split('_'))>2:
                    ax.text(
                        x - 0.2, y0 + h / 2+0.53, "1024 Feature", rotation=90,
                        va='center', ha='right', rotation_mode='anchor', fontsize=11, clip_on=False
                    )
                else:
                    ax.text(
                        x - 0.2, y0 + h / 2+0.53, "2 PCA Feature", rotation=90,
                        va='center', ha='right', rotation_mode='anchor', fontsize=11, clip_on=False
                    )

    # bounds, cosmetics
    x0 = min(r[0] for r in rects); x1 = max(r[0]+r[2] for r in rects)
    y0 = min(r[1] for r in rects); y1 = max(r[1]+r[3] for r in rects)
    pad_x, pad_y = 0.05*(x1-x0), 0.05*(y1-y0)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
#    ax.set_title('Boxes with ADA heat (color) and class as shape')

    # colorbar for ADA
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label('ADA number')

    # legend for shapes
    handles=[circle_proxy, square_proxy]
#    ax.legend(handles=[circle_proxy, square_proxy], loc='lower left', frameon=True,bbox_to_anchor=(0.40, -0.12), borderaxespad=0.)
    ax.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.5, -0.0),
          ncol=len(handles), frameon=True)
#    ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1), borderaxespad=0.)

    plt.tight_layout()
    return fig, ax


def random_box_presentation(totals,red_fracs,File_name):
    rng = np.random.default_rng(7)  # reproducible
    # Rectangles: (x, y, width, height)
    rects = [
        (0.0, 0.0, 2.0, 5.0),
        (2.6, 0.0, 2.0, 5.0),
        (5.3, 0.0, 2.0, 5.0),
        (7.9, 0.0, 2.0, 5.0),
    ]

    # For manual control, you could instead do e.g.:
#    totals = [36, 21, 77, 29]
#    red_fracs = [0.58, 0.86, 0.82, 0.90]

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
        ax.text(x + w/2, y  - 0.25, f'Cluster {idx+1}',
                ha='center', va='bottom', fontsize=9)

        ax.text(x + w/2, y + h + 0.25,
            f"{n_blue} out of {n_total} blue\n{n_red} out of {n_total} red",
            ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))





        if idx==0:
            if len(File_name.split('_'))>2:
                ax.text(
                    x - 0.2, y + h / 2+0.53, "1024 Feature", rotation=90,
                    va='center', ha='right', rotation_mode='anchor', fontsize=11, clip_on=False
                )
            else:
                ax.text(
                    x - 0.2, y + h / 2+0.53, "2 PCA Feature", rotation=90,
                    va='center', ha='right', rotation_mode='anchor', fontsize=11, clip_on=False
                )

    ax.set_aspect('equal', adjustable='box')
    # expand axes a bit around the outer boxes
    x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects); y1 = max(r[1] + r[3] for r in rects)
    padx = 0.05 * (x1 - x0); pady = 0.05 * (y1 - y0)
    ax.set_xlim(x0 - padx, x1 + padx); ax.set_ylim(y0 - pady, y1 + pady)

    ax.set_axis_off()
#    ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1), borderaxespad=0.)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig('plot/Split_' + File_name + '.png')


def save_cluster_2_csv(All_ADA,All_labels,dendro):
    leaf_order = dendro['leaves']
    N_predict_class_1 = np.array([tmp=='tab:blue' for tmp in dendro['leaves_color_list']]).sum()
    N_predict_class_2 = np.array([tmp=='tab:orange' for tmp in dendro['leaves_color_list']]).sum()
    N_predict_class_3 = np.array([tmp=='tab:green' for tmp in dendro['leaves_color_list']]).sum()
    N_predict_class_4 = np.array([tmp=='tab:red' for tmp in dendro['leaves_color_list']]).sum()
    cluster_order_class = All_labels[leaf_order]

    clustr1_drugs = All_ADA['Therapeutic_antibody'][leaf_order][0:N_predict_class_1]
    clustr2_drugs = All_ADA['Therapeutic_antibody'][leaf_order][N_predict_class_1:(N_predict_class_2+N_predict_class_1)]
    clustr3_drugs = All_ADA['Therapeutic_antibody'][leaf_order][(N_predict_class_2+N_predict_class_1):(N_predict_class_2+N_predict_class_1+N_predict_class_3)]
    clustr4_drugs = All_ADA['Therapeutic_antibody'][leaf_order][(N_predict_class_2+N_predict_class_1+N_predict_class_3):]

    clustr1_score = All_ADA['Immun_value'][leaf_order][0:N_predict_class_1]
    clustr2_score = All_ADA['Immun_value'][leaf_order][N_predict_class_1:(N_predict_class_2+N_predict_class_1)]
    clustr3_score = All_ADA['Immun_value'][leaf_order][(N_predict_class_2+N_predict_class_1):(N_predict_class_2+N_predict_class_1+N_predict_class_3)]
    clustr4_score = All_ADA['Immun_value'][leaf_order][(N_predict_class_2+N_predict_class_1+N_predict_class_3):]

    clustr1_label = cluster_order_class[0:N_predict_class_1]
    clustr2_label = cluster_order_class[N_predict_class_1:(N_predict_class_2+N_predict_class_1)]
    clustr3_label = cluster_order_class[(N_predict_class_2+N_predict_class_1):(N_predict_class_2+N_predict_class_1+N_predict_class_3)]
    clustr4_label = cluster_order_class[(N_predict_class_2+N_predict_class_1+N_predict_class_3):]


    True_label_clustr1 = ['red' if tmp==1 else 'blue' for tmp in clustr1_label]
    True_label_clustr2 = ['red' if tmp==1 else 'blue' for tmp in clustr2_label]
    True_label_clustr3 = ['red' if tmp==1 else 'blue' for tmp in clustr3_label]
    True_label_clustr4 = ['red' if tmp==1 else 'blue' for tmp in clustr4_label]

    clustr1 = pd.DataFrame({'Therapeutic_antibody':clustr1_drugs, 'Immun_value':clustr1_score , 'True_label':True_label_clustr1, 'Cluster_label':len(clustr1_label)*['blue']})
    clustr2 = pd.DataFrame({'Therapeutic_antibody':clustr2_drugs, 'Immun_value':clustr2_score , 'True_label':True_label_clustr2, 'Cluster_label':len(clustr2_label)*['Orange']})
    clustr3 = pd.DataFrame({'Therapeutic_antibody':clustr3_drugs, 'Immun_value':clustr3_score , 'True_label':True_label_clustr3, 'Cluster_label':len(clustr3_label)*['Green']})
    clustr4 = pd.DataFrame({'Therapeutic_antibody':clustr4_drugs, 'Immun_value':clustr4_score , 'True_label':True_label_clustr4, 'Cluster_label':len(clustr4_label)*['red']})


    All_drug =  pd.concat([clustr1,clustr2,clustr3,clustr4],axis=0)
    All_drug.to_csv('output/4-Cluster.csv')

# Z_re is your (optionally) reordered linkage that pushes class-1 right
Z_re = HierC.Z
y = All_labels

fig, ax = plt.subplots(figsize=(20, 6))
dendro, groups_list = color_by_second_split(Z_re, y=All_labels,  ax=ax, leaf_font_size=7,
                                       cluster_colors=('tab:blue','tab:orange','tab:green','tab:red'),
                                       show_cluster_strip=True,
                                       keep_leaf_class_colors=True)


save_cluster_2_csv(All_ADA,All_labels,dendro)

leaf_order = dendro['leaves']

N_predict_class_1 = np.array([tmp=='tab:blue' for tmp in dendro['leaves_color_list']]).sum()
N_predict_class_2 = np.array([tmp=='tab:orange' for tmp in dendro['leaves_color_list']]).sum()
N_predict_class_3 = np.array([tmp=='tab:green' for tmp in dendro['leaves_color_list']]).sum()
N_predict_class_4 = np.array([tmp=='tab:red' for tmp in dendro['leaves_color_list']]).sum()

cluster_order_class = y[leaf_order]

cluste_1  = cluster_order_class[0:N_predict_class_1]
cluste_2  = cluster_order_class[N_predict_class_1:(N_predict_class_2+N_predict_class_1)]
cluste_3  = cluster_order_class[(N_predict_class_2+N_predict_class_1):(N_predict_class_2+N_predict_class_1+N_predict_class_3)]
cluste_4  = cluster_order_class[(N_predict_class_2+N_predict_class_1+N_predict_class_3):]


FP1 = (cluste_1==1).sum().item()
TP1 = (cluste_1==0).sum().item()

FP2 = (cluste_2==1).sum().item()
TP2 = (cluste_2==0).sum().item()

FP3 = (cluste_3==1).sum().item()
TP3 = (cluste_3==0).sum().item()

FP4 = (cluste_4==1).sum().item()
TP4 = (cluste_4==0).sum().item()

true_percentage_class_1   = np.round(100 * TP1 / (TP1 + FP1)).item()
false_percentage_class_1  = np.round(100-100 * TP1 / (TP1 + FP1)).item()

true_percentage_class_2   = np.round(100 * TP2 / (TP2 + FP2)).item()
false_percentage_class_2  = np.round(100-100 * TP2 / (TP2 + FP2)).item()

true_percentage_class_3   = np.round(100 * TP3 / (TP3 + FP3)).item()
false_percentage_class_3  = np.round(100-100 * TP3 / (TP3 + FP3)).item()

true_percentage_class_4   = np.round(100 * TP4 / (TP4 + FP4)).item()
false_percentage_class_4  = np.round(100-100 * TP4 / (TP4 + FP4)).item()

# Build multi-line text
stats_text = (
    f"Percentage cluster blue, <{File_name.split('_')[1]} = {true_percentage_class_1}%  -> {TP1}  out of {TP1 + FP1}  \n"
    f"Percentage cluster blue, >{File_name.split('_')[1]} = {false_percentage_class_1}%  -> {FP1}  out of {TP1 + FP1} \n\n"

    f"Percentage cluster orange, <{File_name.split('_')[1]} = {true_percentage_class_2}%  -> {TP2}  out of {TP2 + FP2}  \n"
    f"Percentage cluster orange, >{File_name.split('_')[1]} = {false_percentage_class_2}%  -> {FP2}  out of {TP2 + FP2} \n\n"

    f"Percentage cluster green, <{File_name.split('_')[1]} = {true_percentage_class_3}%  -> {TP3}  out of {TP3 + FP3}  \n"
    f"Percentage cluster green, >{File_name.split('_')[1]} = {false_percentage_class_3}%  -> {FP3}  out of {TP3 + FP3} \n\n"

    f"Percentage cluster red, <{File_name.split('_')[1]} = {true_percentage_class_4}%  -> {TP4}  out of {TP4 + FP4}  \n"
    f"Percentage cluster red, >{File_name.split('_')[1]} = {false_percentage_class_4}%  -> {FP4}  out of {TP4 + FP4}"
)

# Place the text inside the plot
ax.text(
    0.15, 0.90, stats_text,
    transform=ax.transAxes,  # relative coordinates (0–1)
    fontsize=9,
    va='top', ha='left',
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')  # optional background box
)


ax.set_title('Hierarchical clustering – colored by 2nd split (4 branches)')
ax.set_xlabel('Antibody Index'); ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('plot/Hierarchical_4cluster_'+File_name+'.png')
#############################################################################################################################
print("\nGenerating the random box presentation\n")
totals = [TP1 + FP1, TP2 + FP2, TP3 + FP3, TP4 + FP4]
red_fracs = [false_percentage_class_1/100, false_percentage_class_2/100, false_percentage_class_3/100, false_percentage_class_4/100]
random_box_presentation(totals,red_fracs,File_name)

#############################################################################################################################
print("\nGenerating the random box presentation with heatmap")
rects = [
    (0.0, 0.0, 2.0, 5.0),
    (2.6, 0.0, 2.0, 5.0),
    (5.3, 0.0, 2.0, 5.0),
    (7.9, 0.0, 2.0, 5.0),
]

ada_values = np.hstack([Model.train_dataset['Immun_value'], Model.test_dataset['Immun_value']])
group_id = np.full(y.shape[0], -1, dtype=int)
for gi, leaf_idxs in enumerate(groups_list):
    group_id[leaf_idxs] = gi
groups = group_id

 #cmap_name are:
#'viridis' – balanced, not too dark, great default
#'plasma' – brighter, higher contrast than viridis
#'turbo' – vivid/rainbow-ish but still perceptually smooth
#'YlOrRd', 'YlGnBu', 'Oranges', 'Reds' – light-to-dark sequential maps starting from very light
fig, ax = plot_boxes_with_ada_and_shapes(groups, ada_values, y , rects, cmap_name='plasma')
plotname = 'plot/Heatmap_' + File_name + '.png'
plt.savefig(plotname)
print(f"heatmap plot generated at this location {plotname}\n")

#############################################################################################################################
Z_re = HierC.Z
Y_True = All_labels
Z = linkage(All_data_redu, method='ward')
y_Pred = fcluster(Z_re, t=2, criterion='maxclust')-1

#Y_True = All_labels[HierC.dendro['leaves']]
#y_Pred = np.array([tmp=='tab:red' for tmp in HierC.dendro['leaves_color_list']]).astype(int)

from permetrics import ClusteringMetric
cm = ClusteringMetric(X=All_data_redu, y_pred=y_Pred, y_true=Y_True)

print(f"dunn_index is {cm.dunn_index()}")
print(f"hartigan_index is {cm.hartigan_index()}")
print(f"entropy_score is {cm.entropy_score()}")
print(f"ball_hall_index is {cm.ball_hall_index()}")
print(f"purity_score is {cm.purity_score()}")
print(f"tau_score is {cm.tau_score()}")
print(f"calinski_harabasz_index is {cm.calinski_harabasz_index()}")
print(f"density_based_clustering_validation_index is {cm.density_based_clustering_validation_index()}")
print(f"r_squared_index is {cm.r_squared_index()}")
print(f"beale_index is {cm.beale_index()}")
print(f"sum_squared_error_index is {cm.sum_squared_error_index()}")
print(f"duda_hart_index is {cm.duda_hart_index()}")

################################################################################################################################

print("\033[91m\nClustering using fcluster:\033[0m")
Z_re = HierC.Z
#R = inconsistent(Z_re, depth=3)
#monocrit = inconsistent(Z_re, depth=3)[:, 3]
Y_True = All_labels

Z = linkage(All_data_redu, method='ward')
y_Pred = fcluster(Z_re, t=4, criterion='maxclust')-1
sil = silhouette_score(All_data_redu, y_Pred, metric='euclidean')
print(f"silhouette_score is {sil}")
#y_Pred = fcluster(Z_re, t=4, criterion='distance')
#y_Pred = fcluster(Z_re, t=4, criterion='maxclust_monocrit', monocrit=monocrit)
#y_Pred = fcluster(Z_re, t=1.2, criterion='inconsistent', R=R)
X_std = StandardScaler().fit_transform(All_data_redu)
y_Pred = fcluster(Z_re, t=2, criterion='maxclust')-1
sil = silhouette_score(X_std, All_labels, metric='euclidean')
print(f"silhouette_score is {sil}")


cm = confusion_matrix(Y_True,y_Pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(Y_True, y_Pred)
precision = precision_score(Y_True, y_Pred)
f1 = f1_score(Y_True, y_Pred)
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
# Optional: print or log individual results
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
#
#def plot_pca_scatter(data, labels, title, split_label=None, highlight_mask=None):
#    plt.figure(figsize=(16, 8))
#    unique_classes = np.unique(labels)
#
#    for cls in unique_classes:
#        idx = labels == cls
#        plt.scatter(
#            data[idx, 0], data[idx, 1],
#            label=f'ADA cluster {cls}',
#            alpha=0.6
#        )
#
#    # Overlay correctly predicted points if provided
#    if highlight_mask is not None:
#        plt.scatter(
#            data[highlight_mask, 0],
#            data[highlight_mask, 1],
#            s=80,  # Bigger size
#            edgecolors='green',  # <-- Changed from black to red
#            facecolors='none',
#            linewidths=1.2,
#            label="Correctly Predicted"
#        )
#
#
#
#    if False:
#        for idx, (x, y) in enumerate(data):
#            plt.text(x, (y+0.35), str(idx), fontsize=8, ha='center', va='center')
#
#    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
#    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
#
#    names = Model.test_dataset['Therapeutic_antibody']
#    text_x = x_max + 0.5
#    text_y_start = y_max
#    line_height = (y_max - y_min) / max(30, len(names))  # space between lines
#
#    if split_label is not None:
#        plt.title(f"{title} ({split_label})", fontsize=14)
#    else:
#        plt.title(title, fontsize=14)
#    plt.xlabel("PC1", fontsize=12)
#    plt.ylabel("PC2", fontsize=12)
#    plt.legend()
#    plt.grid(True)
#    plt.tight_layout()
#    address = 'plot/' + title.replace(" ", "_") + '.png'
##        plt.savefig(address)
#    plt.savefig(address, format="png", bbox_inches="tight")
#    plt.close()
#
#
#
#cluster_label = np.full(len(All_data_redu), -1, dtype=int)
#[cluster_label.__setitem__(group, index) for index, group in enumerate(groups)]
#
## ---- Plot Training, Test, Combined ---- #
#plot_pca_scatter(All_data_redu, cluster_label, title="2D cluster of Training_"+File_name)
#
#
