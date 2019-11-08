import torch
import numpy as np
from utils import alphaPlot

def validate(model, val_iter, writer):
    model.eval()
    total_loss = 0
    if len(val_iter) == 1:
        random_batch = 0
    else:
        random_batch = np.random.randint(0, len(val_iter) - 1)
    for i, batch in enumerate(val_iter):
        outputs, alpha = model(batch.src, maxLen=len(batch.tgt[1:]))
        (seq_len, batch_size, vocab_size) = outputs.size()

        # tensorboard logging
        preds = outputs.topk(1)[1]
        source = model.src2txt(batch.src[:, 0].data)
        target = model.tgt2txt(batch.tgt[1:, 0].data)
        output = model.tgt2txt(preds[:, 0].data)

        alpha_plot = alphaPlot(alpha[0], output, source)

        writer.add_image('Attention', alpha_plot, dataformats='HWC')
        writer.add_text('Source: ', source)
        writer.add_text('Output: ', output)
        writer.add_text('Target: ', target)
