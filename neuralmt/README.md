## Neural MT Homework

### I. Instructions:

`default.py` contains the default solution, which assigns all alpha in
attention to be equal, and the context vector takes simply the average of all
encoder states.

Before you can run the default solution, make sure you either download the
pre-trained models from `https://jetic.org/cloud/s/YdofIN0CvuCAgux`, or access
it on CSIL machine.

Once you've done that, create a symbolic link to all files in that zip file
from `data/` using something like this (you don't need to unzip on CSIL
machines):

    > unzip ~/Downloads/Archiv.zip -d ~/Downloads/Archiv
    > ln -s ~/Downloads/Archiv/*.pt data/

There should be 5 `.pt` files:

    seq2seq_E045.pt
    seq2seq_E046.pt
    seq2seq_E047.pt
    seq2seq_E048.pt
    seq2seq_E049.pt

By default, the model being used is `seq2seq_E049.pt`, but you can change that
in `default.py`.
When you are implementing ensemble, you can choose which ones to use as well.

#### 1. Baseline: Fixing attention

Attention is defined as follows:

$$\mathrm{score}_i = W_{enc}( h^{enc}_i ) + W_{dec}( h^{dec} )$$

Define $\alpha_i$ for each source side token $i$ as follows:

$$\alpha_i = \mathrm{softmax}(V_{att} \mathrm{tanh} (\mathrm{score}_i))$$

The we define the context vector using the $\alpha$ weights:

$$c = \sum_i \alpha_i \times h^{enc}_i$$

The context vector $c$ is combined with the current decoder hidden
state $h^{dec}$ and this representation is used to compute the
softmax over the target language vocabulary at the current decoder
time step. We then move to the next time step and repeat this process
until we produce an end of sentence marker.

#### 2. Extensions

We fixed the interface in a specific way that allows you to implement at least:

1. UNK replacement: https://www.aclweb.org/anthology/P15-1002/

2. BeamSearch

3. Ensemble

Original training data is also provided (tokenised). You may use it whichever
way you want to augment the provided Seq2Seq model.

### II. Useful Tool

For visualisation, one could easily use the included functions in `utils.py`:

    from utils import alphaPlot

    # Since alpha is batched, alpha[0] refers to the first item in the batch
    alpha_plot = alphaPlot(alpha[0], output, source)

This converts the alpha values into a nice attention graph.
Example code in combination with `tensorboard` is provided in `validator.py`.
This can help you visualise an entire `test_iter`.

In addition, `default.py` has an additional parameter `-n`.
If your inference is taking too long and you'd like to test your implementation
with a subset of dev (say first 100 samples), you can do that.

### Reference

Thanks, @A-Jacobson.
