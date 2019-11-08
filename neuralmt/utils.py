import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def alphaPlot(alpha,
              output,
              source):
    plt.figure(figsize=(14, 6))
    sns.heatmap(alpha, xticklabels=output.split(), yticklabels=source.split(),
                linewidths=.05, cmap="Blues")
    plt.ylabel('Source')
    plt.xlabel('Target')
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    plt.tight_layout()
    buff = io.BytesIO()
    plt.savefig(buff, format='jpg')
    buff.seek(0)
    return np.array(Image.open(buff))
