import torch
import glob
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import homogeneity_completeness_v_measure

key = "vae_z"
if len(sys.argv) > 1:
    key = sys.argv[1]
print(f"Clustering feature '{key}' from model outputs")

batches = dict()
for d in glob.glob("output/test_*_output.pt"):
    outputs = torch.load(d)
    batch = int(d.split("_")[1])
    try:
        batches[batch] = outputs[key].detach().cpu()
    except KeyError:
        print(f"Feature '{key}' not found!")
        break
else:
    X = torch.cat([batch for _, batch in sorted(batches.items())], dim=0)

    data = np.load(f"data/bin/test.npz")
    assert max(data["ids"]) == len(data["paths"]) - 1
    labels = data['paths'][data['ids']]
    labels = [int(l.split("/")[-1][3:6]) for l in labels]
    labels = [chr(l-11+ord('A')) if l < 37 else chr(l-37+ord('a')) for l in labels]
    gold = np.array([labels.index(l) for l in labels])
    n_gold = len(set(gold))

    vs = []
    for n_clusters in [n_gold - 5*idx for idx in range(-3,3+1)]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        (h, c, v) = homogeneity_completeness_v_measure(gold, clustering.labels_)
        vs.append(v)
    print(np.mean(vs))
