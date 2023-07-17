import torch
import glob
import sys
import numpy as np

key = "vae_z"
if len(sys.argv) > 1:
    key = sys.argv[1]
print(f"Zipping feature '{key}' from model outputs")

batches = dict()
for d in glob.glob("outputs/test_*_output.pt"):
    outputs = torch.load(d)
    batch = int(d.split("_")[1])
    try:
        batches[batch] = outputs[key].detach().cpu()
    except KeyError:
        print(f"Feature '{key}' not found!")
        break
else:
    np.savez_compressed("output", output=batches)
