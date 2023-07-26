import torch
import math
from torch import nn
import torch.nn.functional as F

# Default solution
from default import VAE
# Student solution
#from vae import VAE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0), :].squeeze()
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(self, z_dim):
        super(TransformerLM, self).__init__()
        self.z_dim = z_dim

        self.ePos = PositionalEncoding(z_dim,0.5)
        self.eEmb = nn.Linear(z_dim,z_dim)
        self.eLM = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    z_dim, # input dimension
                    8,
                    z_dim, # output dimension
                    0.5,
                    batch_first = True,
                ),
                6, # num. layers
        )

    def forward(self, x, mask_prob=0.05):
        masked = x.clone()
        # Sample random positions to mask, and replace
        # those token images with random normal noise
        mask   = torch.bernoulli(mask_prob*torch.ones_like(x[:,0])).bool()
        masked[mask].uniform_()
        emb = self.eEmb(masked)
        pos = self.ePos(emb)
        device = 'cpu' if x.get_device() < 0 else x.get_device()
        y = self.eLM(
            pos,
            mask=torch.eye(pos.shape[0]).bool().to(device)
        )
        return y

class VAECluster(nn.Module):
    def __init__(self, H_DIM):
        super(VAECluster, self).__init__()
        self.H_DIM = H_DIM

        # VAE to encode/decode images <-> codes.
        # In the contextual models, the entire
        # sequence shares a single encoder/decoder.
        self.vae  = VAE(self.H_DIM)

        # The VAE+Transformer model needs a Transformer encoder:
        self.txr  = TransformerLM(self.H_DIM)
        # ...and a way to decode the Transformer outputs back
        # into the same space as the VAE codes:
        self.dLM = nn.Linear(self.H_DIM, self.H_DIM)

        # Projections for reparametrizing LM outputs:
        self.lmMu = nn.Linear(self.H_DIM,self.H_DIM)
        self.lmSigma = nn.Linear(self.H_DIM,self.H_DIM)

    def forward(self,x, mask_prob=0.05):
        results = dict()

        # 1. Encode the input sequence:
        # - We technically don't need to return mu and sigma,
        #   but we will want to look at them during the analysis:
        results["vae_mu"], results["vae_sigma"] = self.vae.project(
            self.vae.encode(x)
        )
        # - The actual encodings are in vae_z:
        results["vae_z"] = self.vae.reparametrize(
            results["vae_mu"],
            results["vae_sigma"],
        )

        # 2. Decode vae_z to get a sequence of output images:
        results["gen_vae_self"] = self.vae.decode(results["vae_z"])


        # 3. Try to recover noised images using a Transformer as a masked LM:
        # Run the Transformer over the sequence of codes (vae_z),
        # masking some with a constant probability:
        results["txr_y"] = self.txr(results["vae_z"], mask_prob=mask_prob)
        # Decode the top layer of the Transformer into a sequence of codes:
        # TODO Do this before or after reparametrizing?
        results["txr_z"] = self.dLM(results["txr_y"])
            
        # Optionally(?) reparameterize the Transformer output?
        # Not clear if this is always needed. 
        #
        # TODO Do we need lmMu and lmSigma, or can we reuse eMu
        # and eSigma? Initial expts suggest it's better to have 
        # separate projections here.
        #
        # TODO Expts to test whether/in what cases this step is needed.
        results["txr_mu"]    = self.lmMu(results["txr_z"])
        results["txr_sigma"] = self.lmSigma(results["txr_z"])
        results["txr_z"]     = self.vae.reparametrize(
            results["txr_mu"], 
            results["txr_sigma"]
        )
        results["gen_vae_txr"] = self.vae.decode(results["txr_z"])

        return results
