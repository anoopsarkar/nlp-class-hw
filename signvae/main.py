import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transform
from torchvision.transforms import RandomAffine

from sklearn.cluster import KMeans

import model

# Default solution
from default import vae_loss
# Student solution
#from answer.vae import vae_loss

##################################################
# Configuration

parser = argparse.ArgumentParser(description='')
parser.add_argument('--cuda', action='store_true',
                    help='If true, use CUDA instead of CPU.')
parser.add_argument('--seed', default=0xbeef, type=int)

parser.add_argument('--train', default=False, action='store_true',
                    help='If true, run training loop.')
parser.add_argument('--test', default=False, action='store_true',
                    help='If true, save model outputs on test set.')

parser.add_argument('--traindata', default=os.path.join('data', 'bin', 'train.npz'),
                    help='default: data/bin/train.npz')
parser.add_argument('--testdata', default=os.path.join('data', 'bin', 'test.npz'),
                    help='default: data/bin/test.npz')

parser.add_argument('--modelfile', default=os.path.join('data', 'trained.pt'),
                    help='default: data/trained.pt')
parser.add_argument('--configfile', default=os.path.join('data', 'trained.config'),
                    help='default: data/trained.config')
parser.add_argument('--outputdir', default='output', help='default: output')

parser.add_argument('--augment', default=False, action='store_true',
                    help='If true, apply random augmentations to images during training.')

parser.add_argument('--pseudolabels', default=100, type=int,
                    help='Number of clusters used to assign pseudolabels.')
parser.add_argument('--pseudolabel_trials', default=3, type=int,
                    help='Number of times to repeat pseudolabel clustering.')
parser.add_argument('--pseudolabel_interval', default=600, type=int,
                    help='How many epochs to cache the pseudolabels.')

parser.add_argument('--dim', default=16, type=int,
                    help='Hidden dimension (default 16).')
parser.add_argument('--train_iters', required=True, type=int,
                    help='Number of training iterations.')
parser.add_argument('--lr', required=True, type=float,
                    help='Learning rate.')
parser.add_argument('--kl_scale', default=1, type=float,
                    help='KL divergence loss will be scaled up/down by this factor.')
parser.add_argument('--seq_length', required=True, type=int,
                    help='Length of training sequences.')

parser.add_argument('--log_interval', default=10, type=int,
                    help='How often to print training details.')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: CUDA is available but --cuda flag was not set.")
device = torch.device("cuda" if args.cuda else "cpu")

##################################################
# Data Setup

# Load image sequences:
data = np.load(args.traindata)
data = {k:v for k,v in data.items()}
ntokens = data['ids'].shape[0]

# Get the mean and stddev of the training
# data: we need to normalize them to have
# zero mean and unit stddev.
norm_mu  = data['images'].mean()
norm_std = data['images'].std()
black = (0 - norm_mu)/norm_std

def batch_generator(data_src, seq_length=args.seq_length):
    """
    Returns a generator which yields one batch at a time
    from a given data source. Each batch is a dict with
    the following keys:
    - input_ids:    sequence of image ids
    - input_images: corresponding sequence of images
    - indices:      where these tokens occur in the dataset
    """
    idx = 0
    while True:
        batch = dict()
        if idx + seq_length >= len(data_src['ids']):
            idx = 0

        # construct sequence of indices
        grid = np.mgrid[0:2, 0:seq_length]
        indices = (grid[0] + grid[1] + idx)
        # convert to sequence of image ids
        ids = data_src['ids'][indices]

        batch['input_ids']    = ids[:-1]
        batch['input_images'] = torch.FloatTensor(
            (data_src['images'][batch['input_ids']]-norm_mu)/norm_std
        ).unsqueeze(2).to(device)
        batch['indices']      = indices

        idx += seq_length
        yield batch

##################################################
# Loss

def L_Phi(output, target_centroids, centroids):
    """
    Measures pairwise distance between the intended pseudolabel
    and the actual nearest centroid for each sample; returns
    cross-entropy wrt a distribution with all mass on the target
    pseudolabel.
    """
    pseudolabel_loss = 0
    for trial in range(args.pseudolabel_trials):
        a = output['vae_z'].repeat_interleave(centroids[trial].shape[0], dim=0)
        b = centroids[trial].repeat((output['vae_z'].shape[0],1))
        observed_centroids  =  F.pairwise_distance(a,b).reshape((target_centroids[trial].shape[0],-1))
        pseudolabel_loss   +=  F.cross_entropy(observed_centroids, target_centroids[trial])
    return pseudolabel_loss # TODO /args.pseudolabel_trials ?

def loss_fn(x, output, target_centroids, centroids):
    # Pseudolabel loss to cluster codes in the representation space:
    pseudolabel_loss = L_Phi(output, target_centroids, centroids)

    # List each seqeunce of generated images:
    gen_images = [
        output['gen_vae_self'].nan_to_num().clamp(min=0.01,max=0.99), 
        output['gen_vae_txr'].nan_to_num().clamp(min=0.01,max=0.99),
    ]
    # List each sequence of mean/stdev vectors:
    mu_sigmas = [
        (output['vae_mu'], output['vae_sigma']),
        (output['txr_mu'], output['txr_sigma']),
    ]
    # Compute reconstruction and KL divergence losses:
    BCEs, KLDs = vae_loss(gen_images, x, mu_sigmas)

    return BCEs, KLDs, pseudolabel_loss

##################################################
# Training Loop

def refresh_pseudolabels(n_clusters):
    """
    Encodes the entire training data, then clusters
    the encodings with KMeans. Returns a tuple of
    (cluster labels, cluster centroids).
    """
    print("Refreshing pseudolabels...")
    pseudo_generator = batch_generator(data)
    codes = []
    n_batches = np.ceil(data['ids'].shape[0]/args.seq_length)
    while len(codes) < n_batches:
        print('Seen ', len(codes)*args.seq_length, end='\r')
        batch = next(pseudo_generator)
        output = model(batch['input_images'][0])
        codes.append(output['vae_z'].detach().cpu())
    codes = torch.concatenate(codes, axis=0)[:data['ids'].shape[0]]
    print("Clustering codes...")
    clustering = KMeans(n_clusters=n_clusters, n_init=10).fit(codes)
    return clustering.labels_, clustering.cluster_centers_

batches = batch_generator(data)
pseudolabels = torch.zeros(
    (args.pseudolabel_trials, data['ids'].shape[0]), 
    requires_grad=False
).to(device).long()
centroids = torch.zeros(
    (args.pseudolabel_trials, args.pseudolabels, args.dim), 
    requires_grad=False
).to(device).double()
def train(model, iters=1000):
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = None
    for i in range(iters):
        # Refresh pseudolabels:
        if i%args.pseudolabel_interval == 0:
            for trial in range(args.pseudolabel_trials):
                trial_labels, trial_centroids = refresh_pseudolabels(args.pseudolabels) 
                pseudolabels[trial] = torch.tensor(trial_labels)
                centroids[trial]    = torch.tensor(trial_centroids)

        optim.zero_grad()

        batch = next(batches)
        im_seq = batch['input_images'][0]
        # Optionally apply a random affine
        # transformation which the model
        # must learn to denoise:
        if args.augment:
            transformation = RandomAffine(
                degrees=45, 
                scale=(0.4, 1),
                shear=25,
                fill=black,
            )
            aug_seq = transformation(im_seq)
        else:
            aug_seq = im_seq
        # Run model on augmented image sequence
        output = model(aug_seq)

        # Targets need to be rescaled to the range 0-1:
        targets = (im_seq * norm_std) + norm_mu
        loss_bces, loss_klds, loss_pseudo = loss_fn(
            targets, 
            output, 
            pseudolabels[:,batch['indices'][0]], 
            centroids
        )
        loss = sum(loss_bces) + args.kl_scale*sum(loss_klds) + loss_pseudo
        loss.backward()
        # Model WILL explode if gradients are not clipped:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()

        if i%args.log_interval == 0:
            print(f"""iter. {i:3} | Loss: {loss.item() / args.seq_length}
BCE: {' '.join([f'{l.item():.02f}' for l in loss_bces])} | KLD: {' '.join([f'{l.item():.2e}' for l in loss_klds])}""")

if __name__ == "__main__":
    if args.train:
        print(f"Training on {ntokens} tokens")
        model = model.VAECluster(args.dim).to(device)
        train(model, args.train_iters)

        ##################################################
        # Save Model

        print(f"Saving model to {args.modelfile}")
        with open(args.modelfile, 'wb') as fp:
            torch.save(model, fp)
        with open(args.configfile, "w") as fp:
            fp.write(str(args))
    else:
        print(f"Loading trained model from {args.modelfile}")
        model = torch.load(args.modelfile)

    if args.test:
        ##################################################
        # Evaluation

        data = np.load(args.testdata)
        data = {k:v for k,v in data.items()}
        ntokens = data['ids'].shape[0]
        print(f"Test on {ntokens} tokens")

        test_batches = batch_generator(data)
        batch_no = 0
        while args.seq_length * batch_no < ntokens:
            batch_no += 1
            test_batch = next(test_batches)
            im_seq = test_batch['input_images'][0]
            # Disable masking at test time:
            output = model(im_seq, mask_prob=0)

            with open(os.path.join(args.outputdir, f"test_{batch_no:02d}_ids.pt"), "wb") as fp:
                torch.save(test_batch['input_ids'], fp)
            with open(os.path.join(args.outputdir, f"test_{batch_no:02d}_inputs.pt"), "wb") as fp:
                torch.save(im_seq.detach().to('cpu'), fp)
            with open(os.path.join(args.outputdir, f"test_{batch_no:02d}_output.pt"), "wb") as fp:
                torch.save(output, fp)
