import torch
import torch.nn as nn
import torch.nn.functional as F


class SNPPOEMSVAE(nn.Module):
    """
    Configurable supervised SNP-only VAE.

    Shared components
    -----------------
    x -> encoder -> q_phi(z|x)
                  -> phenotype head

    Decoder options
    ---------------
    decoder_type="poems"
        Current POEMS SNP-specific latent gating.
        Supported genotype_likelihood: "mse".

    decoder_type="dense_mlp"
        Andreas-style dense decoder:
            z -> dec_hidden_dim -> dec_hidden_dim2 -> SNP output parameters
        Supported genotype_likelihood:
            "mse"         : one real-valued output / SNP
            "bernoulli"   : one logit / SNP, for x_j in {0,1}
            "categorical" : C logits / SNP, for discrete classes e.g. {0,1,2}

    Notes for interpretation
    ------------------------
    * For Bernoulli decoder geometry, decode_logits(z) gives the underlying
      latent-to-SNP logit map. decode(z) returns probabilities.
    * For categorical reconstruction, decode(z) returns E[X|z] for convenient
      common-scale reconstruction summaries, while decode_logits(z) and
      decode_probs(z) expose the full class distribution.
    """

    def __init__(
        self,
        input_dim,
        latent_dim=16,
        enc_hidden_dim=128,
        dec_hidden_dim=64,
        dec_hidden_dim2=512,
        n_genotype_classes=3,
        n_traits=1,
        dropout=0.2,
        regressor_type="mlp",
        trait_likelihood="gaussian",
        decoder_type="poems",
        genotype_likelihood="mse",
    ):
        super().__init__()

        if regressor_type not in {"mlp", "linear"}:
            raise ValueError("regressor_type must be 'mlp' or 'linear'.")
        if trait_likelihood not in {"gaussian", "mse"}:
            raise ValueError("trait_likelihood must be 'gaussian' or 'mse'.")
        if decoder_type not in {"poems", "dense_mlp"}:
            raise ValueError("decoder_type must be 'poems' or 'dense_mlp'.")
        if genotype_likelihood not in {"mse", "bernoulli", "categorical"}:
            raise ValueError(
                "genotype_likelihood must be 'mse', 'bernoulli', or 'categorical'."
            )
        if decoder_type == "poems" and genotype_likelihood != "mse":
            raise ValueError(
                "The controlled ablation currently supports POEMS only with MSE. "
                "Use decoder_type='dense_mlp' for Bernoulli/categorical likelihoods."
            )
        if int(n_genotype_classes) < 2:
            raise ValueError("n_genotype_classes must be >= 2.")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.enc_hidden_dim = int(enc_hidden_dim)
        self.dec_hidden_dim = int(dec_hidden_dim)
        self.dec_hidden_dim2 = int(dec_hidden_dim2)
        self.n_genotype_classes = int(n_genotype_classes)
        self.n_traits = int(n_traits)
        self.regressor_type = regressor_type
        self.trait_likelihood = trait_likelihood
        self.decoder_type = decoder_type
        self.genotype_likelihood = genotype_likelihood

        # -------------------------
        # Encoder: unchanged
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.enc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.enc_hidden_dim, self.enc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(self.enc_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.enc_hidden_dim, self.latent_dim)

        # -------------------------
        # Decoder
        # -------------------------
        if self.decoder_type == "poems":
            self.W = nn.Parameter(
                torch.randn(self.input_dim, self.latent_dim) * 0.02
            )
            self.decoder_hidden = nn.Sequential(
                nn.Linear(self.latent_dim, self.dec_hidden_dim, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.column_means_weight = nn.Parameter(
                torch.randn(self.input_dim, self.dec_hidden_dim, 1) * 0.02
            )
            self.column_means_bias = nn.Parameter(
                torch.zeros(self.input_dim, 1)
            )
        else:
            self.decoder_body = nn.Sequential(
                nn.Linear(self.latent_dim, self.dec_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.dec_hidden_dim, self.dec_hidden_dim2),
                nn.ReLU(),
            )
            if self.genotype_likelihood == "categorical":
                out_dim = self.input_dim * self.n_genotype_classes
            else:
                out_dim = self.input_dim
            self.decoder_output = nn.Linear(self.dec_hidden_dim2, out_dim)

        # -------------------------
        # Phenotype head: unchanged
        # -------------------------
        trait_out_dim = (
            self.n_traits * 2
            if self.trait_likelihood == "gaussian"
            else self.n_traits
        )
        if self.regressor_type == "linear":
            self.regressor = nn.Linear(self.latent_dim, trait_out_dim)
        else:
            self.regressor = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.latent_dim, trait_out_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=10.0)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def get_decoder_mask(self):
        if self.decoder_type != "poems":
            raise AttributeError("Dense decoder does not have a POEMS gate W.")
        return self.W

    def _decode_poems_mse(self, z):
        batch_size = z.shape[0]
        W = self.W.unsqueeze(1)          # [P, 1, L]
        z_expand = z.unsqueeze(0)        # [1, B, L]
        masked_inputs = W * z_expand     # [P, B, L]

        decoder_in = masked_inputs.reshape(
            self.input_dim * batch_size,
            self.latent_dim,
        )
        hidden = self.decoder_hidden(decoder_in)
        hidden = hidden.view(
            self.input_dim,
            batch_size,
            self.dec_hidden_dim,
        )
        x_hat = (
            torch.matmul(hidden, self.column_means_weight).squeeze(-1)
            + self.column_means_bias
        )
        x_hat = x_hat.transpose(0, 1)    # [B, P]
        return torch.clamp(x_hat, min=-10.0, max=10.0)

    def decode_logits(self, z):
        """
        Return natural decoder output parameters.

        Bernoulli:
            [B,P] logits.
        Categorical:
            [B,P,C] logits.

        For MSE there is no probability logit parameterization; the raw decoder
        output itself is returned.
        """
        if self.decoder_type == "poems":
            return self._decode_poems_mse(z)

        h = self.decoder_body(z)
        raw = self.decoder_output(h)

        if self.genotype_likelihood == "categorical":
            return raw.view(
                -1,
                self.input_dim,
                self.n_genotype_classes,
            )
        return raw

    def decode_probs(self, z):
        if self.genotype_likelihood == "bernoulli":
            return torch.sigmoid(self.decode_logits(z))
        if self.genotype_likelihood == "categorical":
            return F.softmax(self.decode_logits(z), dim=-1)
        raise ValueError("decode_probs() is defined only for probabilistic SNP likelihoods.")

    def decode_expected_genotype(self, z):
        if self.genotype_likelihood == "mse":
            return self.decode_logits(z)
        if self.genotype_likelihood == "bernoulli":
            return torch.sigmoid(self.decode_logits(z))

        probs = self.decode_probs(z)
        classes = torch.arange(
            self.n_genotype_classes,
            dtype=probs.dtype,
            device=probs.device,
        )
        return (
            probs * classes.view(1, 1, -1)
        ).sum(dim=-1)

    def decode(self, z):
        """
        Common-scale [B,P] reconstruction:
          MSE         -> raw reconstruction
          Bernoulli   -> ALT probability E[X|z]
          Categorical -> expected class / dosage E[X|z]
        """
        return self.decode_expected_genotype(z)

    def decoder_regularization(self):
        """
        Unweighted decoder regularization quantity.

        The existing POEMS experiment penalizes mean |W|.
        Dense decoder experiments return zero by default so the architecture
        ablation does not silently introduce a new regularizer.
        """
        if self.decoder_type == "poems":
            return torch.abs(self.W).mean()
        return torch.zeros(
            (),
            device=next(self.parameters()).device,
        )

    # ------------------------------------------------------------------
    # Trait head
    # ------------------------------------------------------------------

    def predict_traits(self, z):
        out = self.regressor(z)
        if self.trait_likelihood == "gaussian":
            y_mu, y_logvar = torch.chunk(out, 2, dim=1)
            y_logvar = torch.clamp(
                y_logvar,
                min=-10.0,
                max=5.0,
            )
            return y_mu, y_logvar
        return out, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)

        x_logits = None
        x_probs = None

        if self.genotype_likelihood == "mse":
            x_hat = self.decode(z)
        elif self.genotype_likelihood == "bernoulli":
            x_logits = self.decode_logits(z)
            x_probs = torch.sigmoid(x_logits)
            x_hat = x_probs
        else:
            x_logits = self.decode_logits(z)
            x_probs = F.softmax(x_logits, dim=-1)
            classes = torch.arange(
                self.n_genotype_classes,
                dtype=x_probs.dtype,
                device=x_probs.device,
            )
            x_hat = (
                x_probs * classes.view(1, 1, -1)
            ).sum(dim=-1)

        y_mu, y_logvar = self.predict_traits(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "x_hat": x_hat,
            "x_logits": x_logits,
            "x_probs": x_probs,
            "y_hat": y_mu,
            "y_mu": y_mu,
            "y_logvar": y_logvar,
            "y_std": (
                torch.exp(0.5 * y_logvar)
                if y_logvar is not None
                else None
            ),
        }
