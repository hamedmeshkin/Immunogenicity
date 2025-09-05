import re
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--purpose', dest='purpose',default = 'ROC_AUC', type=str, help='Mu, Energy, silhouette_score, dunn_index, tau_score, ')

args = parser.parse_args()
purpose = args.purpose

# --- settings ---
OUTPUT_DIR = Path("./logfiles/")                  # change if needed
SAVE_FIG   = "energy_distance_by_file.png"   # set to None to skip saving

# Regex to catch floats (incl. scientific notation) after "Energy Distance"

if purpose == "Energy":
    pat = re.compile(
        r"Energy\s*Distance\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    ylabel="Energy Distance"

if purpose == "Mu":
    pat = re.compile(
        r"Mean\s*Between\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    ylabel="Mean Between"

if purpose == "silhouette_score":
    pat = re.compile(
        r"silhouette_score\s*is\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    ylabel="silhouette score"

if purpose == "tau_score":
    pat = re.compile(
        r"tau_score\s*is\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    ylabel="tau score"

if purpose == "ROC_AUC":
    pat = re.compile(
        r"ARI:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    ylabel="ARI ARI"

files = sorted([p for p in OUTPUT_DIR.glob("*") if p.is_file()])
names, values = [], []

for f in files:
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    m = pat.findall(text)
    if not m:
        continue

    # If multiple matches in one file, take the last one
    val = float(m[-1])
    names.append(f.name.split('.txt')[0])
    values.append(val)

def numeric_key(fname: str):
    """Return a numeric key from filename (prefer stem as int/float; fallback to first number in name)."""
    stem = Path(fname).stem
    try:
        return int(stem)
    except ValueError:
        try:
            return float(stem)
        except ValueError:
            m = re.search(r"[+-]?\d+(?:\.\d+)?", fname)
            return float(m.group()) if m else float('inf')

# ---- after you've built lists 'names' and 'values' ----
order = sorted(range(len(names)), key=lambda i: numeric_key(names[i]))
names  = [names[i]+'%'  for i in order]
values = [values[i] for i in order]

# ---- plot as a bar chart ----
x = range(len(values))
plt.figure(figsize=(max(6, 0.6*len(values)), 4.5))
plt.bar(x, values, width=0.4, edgecolor='black', linewidth=0.8)
plt.xticks(x, names )
plt.ylabel(ylabel)
plt.xlabel("Cut-off")
#plt.title("Energy Distance by file (sorted numerically)")
plt.tight_layout()

Fig_name = ylabel.split(' ')[0]+'_'+ylabel.split(' ')[1]
plt.savefig('plot/Average_'+ Fig_name + '.png')
