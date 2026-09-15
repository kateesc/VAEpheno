import torch
import torch.nn as nn


class SNPPOEMSVAE(nn.Module):
    """
    Experiment A: current POEMS-style decoder + MSE reconstruction.
    """

    decoder_type = "poems"
    genotype_likelihood = "mse"

    def __init__(
        self,
        input_dim,
        latent_dim=16,
        enc_hidden_dim=128,
        dec_hidden_dim=64,
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

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
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

        self.W = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.02)

        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.column_means_weight = nn.Parameter(
            torch.randn(input_dim, dec_hidden_dim, 1) * 0.02
        )
        self.column_means_bias = nn.Parameter(torch.zeros(input_dim, 1))

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

    def get_decoder_mask(self):
        return self.W

    def decode(self, z):
        batch_size = z.shape[0]
        W = self.get_decoder_mask().unsqueeze(1)
        z_expand = z.unsqueeze(0)
        masked_inputs = W * z_expand

        decoder_in = masked_inputs.reshape(
            self.input_dim * batch_size, self.latent_dim
        )
        hidden = self.decoder_hidden(decoder_in)
        hidden = hidden.view(self.input_dim, batch_size, self.dec_hidden_dim)

        x_hat = (
            torch.matmul(hidden, self.column_means_weight).squeeze(-1)
            + self.column_means_bias
        )
        return torch.clamp(x_hat.transpose(0, 1), min=-10.0, max=10.0)

    def predict_traits(self, z):
        out = self.regressor(z)
        if self.trait_likelihood == "gaussian":
            y_mu, y_logvar = torch.chunk(out, 2, dim=1)
            y_logvar = torch.clamp(y_logvar, min=-10.0, max=5.0)
            return y_mu, y_logvar
        return out, None

    def decoder_regularization(self):
        return torch.abs(self.W).mean()

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)

        x_hat = self.decode(z)
        y_mu, y_logvar = self.predict_traits(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "x_hat": x_hat,
            "x_logits": None,
            "x_probs": None,
            "y_hat": y_mu,
            "y_mu": y_mu,
            "y_logvar": y_logvar,
            "y_std": torch.exp(0.5 * y_logvar) if y_logvar is not None else None,
        }
