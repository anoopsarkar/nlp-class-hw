# The Challenge

Imagine that you are an archaeologist from the far future, when the English language has been totally forgotten. You have just uncovered a stash of handwritten English manuscripts which you are trying to decipher. Unfortunately, you don't know the English alphabet, and so you aren't able to tell whether any two symbols represent the same letter. Your task is to recover the English alphabet by clustering the characters from these manuscripts.

## The Baseline

You are given a partial implementation of the character clustering model from [Born et al. 2023](...). However, we have removed the variational autoencoder (VAE), which encodes images as dense vectors and decodes those vectors to recover the original images. You must complete the model by finishing the partial implementations of `VAE`, `kld()`, and `vae_loss()` provided in `default.py`. 

## Extensions to the Baseline

The `VAECluster.forward()` method (in `model/__init__.py`) returns a Python `dict` named `results` which contains sequences of features describing the images which the model has seen. For example, `results["vae_z"]` contains the code $z$ computed for each input image, and `results["gen_vae_self"]` contains the images that have been reconstructed from these codes. 

By default, `check.py` evaluates your output by clustering the `vae_z` features. You can pass any key from the `results` dictionary to `check.py` in order to cluster on that feature instead:
```
python3 check.py gen_vae_self # will cluster the reconstructed images instead of the VAE codes
```

Once you have completed the basic model implementation, you should try to improve your clustering accuracy by adding additional features to the `results` dictionary and clustering on those features. You may use any features you think will be helpful. Possible ideas include:
- internal layers, self-attention scores, or other features from the Transformer in `VAECluster.txr`
- concatenating `vae_z` with another feature such as `txr_z` (the code output by the Transformer LM)
- [2D cross-correlations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html) between images
- features from the `pseudolabels` or `centroids` arrays used to compute the pseudolabel loss (additional details [here](https://arxiv.org/pdf/1807.05520.pdf))

# Check Your Accuracy

To train the model and save the outputs on the test set, run:
```
python3 train.py
```

If you have a CUDA-capable GPU, you can train on GPU using:
```
python3 train.py --cuda
```

To check your accuracy on the test set:
```
python3 check.py
```

Or, to check the accuracy from clustering on a specific feature (e.g. `txr_z`):
```
python3 check.py txr_z
```

The output score is the [V-measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html) of your clustering compared to the ground truth.

## Additional Tools

`inspect.ipynb` contains sample code for inspecting the model inputs and outputs. You may wish to use this to check whether your model is correctly recovering the character images.

# Submit your homework on Coursys

Once you are done with your homework submit all the relevant materials to Coursys for evaluation.

## Create output.npz

Once you have trained a model and saved the outputs on the test set, create the `output.npz` for Coursys by running:
```
python3 zipout.py
```

By default, this will zip the `vae_z` feature vectors. If you have added a new feature to the `results` dictionary, and would like to be evaluated on that feature instead, please specify the feature name when running `zipout.py`:
```
python3 zipout.py your_feature_name
```

