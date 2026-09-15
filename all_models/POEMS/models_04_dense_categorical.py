import torch
import torch.nn as nn
import torch.nn.functional as F


class SNPPOEMSVAE(nn.Module):
    """
    Experiment D / future diploid option:
    dense MLP decoder + categorical genotype likelihood.

    Intended for discrete classes, normally x_j in {0,1,2}.
    """

    decoder_type = "dense_mlp"
    genotype_likelihood = "categorical"

    def __init__(
        self,
        input_dim,
        latent_dim=16,
        enc_hidden_dim=128,
        dec_hidden_dim=256,
        dec_hidden_dim2=512,
        n_genotype_classes=3,
        n_traits=1,
        dropout=0.2,
        regressor_type="mlp",
        trait_likelihood="gaussian",
    ):
        super().__init__()
        if regressor_type not in {"mlp", "linear"}:
            raise ValueError("regressor_type must be 'mlp' or 'linear'.")
        if trait_likelihood not in {"gaussian", "mse"}:
            raise ValueError("trait_likelihood must be 'gaussian' or 'mse'.")
        if n_genotype_classes < 2:
            raise ValueError("n_genotype_classes must be >= 2.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_genotype_classes = int(n_genotype_classes)
        self.n_traits = n_traits
        self.regressor_type = regressor_type
        self.trait_likelihood = trait_likelihood

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden_dim, enc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(enc_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_hidden_dim, latent_dim)

        self.decoder_body = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden_dim),
            nn.ReLU(),
            nn.Linear(dec_hidden_dim, dec_hidden_dim2),
            nn.ReLU(),
        )
        self.decoder_output = nn.Linear(
            dec_hidden_dim2, input_dim * self.n_genotype_classes
        )

        trait_out_dim = n_traits * 2 if trait_likelihood == "gaussian" else n_traits
        if regressor_type == "linear":
            self.regressor = nn.Linear(latent_dim, trait_out_dim)
        else:
            self.regressor = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, trait_out_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=10.0)
        return mu + torch.randn_like(std) * std

    def decode_logits(self, z):
        h = self.decoder_body(z)
        logits = self.decoder_output(h)
        return logits.view(-1, self.input_dim, self.n_genotype_classes)

    def decode_probs(self, z):
        return F.softmax(self.decode_logits(z), dim=-1)

    def decode_expected_dosage(self, z):
        probs = self.decode_probs(z)
        classes = torch.arange(
            self.n_genotype_classes,
            dtype=probs.dtype,
            device=probs.device,
        )
        return (probs * classes.view(1, 1, -1)).sum(dim=-1)

    def decode(self, z):
        return self.decode_expected_dosage(z)

    def predict_traits(self, z):
        out = self.regressor(z)
        if self.trait_likelihood == "gaussian":
            y_mu, y_logvar = torch.chunk(out, 2, dim=1)
            y_logvar = torch.clamp(y_logvar, min=-10.0, max=5.0)
            return y_mu, y_logvar
        return out, None

    def decoder_regularization(self):
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)

        x_logits = self.decode_logits(z)
        x_probs = F.softmax(x_logits, dim=-1)

        classes = torch.arange(
            self.n_genotype_classes,
            dtype=x_probs.dtype,
            device=x_probs.device,
        )
        x_hat = (x_probs * classes.view(1, 1, -1)).sum(dim=-1)

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
            "y_std": torch.exp(0.5 * y_logvar) if y_logvar is not None else None,
        }
