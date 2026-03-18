"""
hypermap/model.py
Neural network architecture for HyperMAP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Generates latent embedding for perturbation or cell."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, VAE: bool = False):
        super().__init__()
        self.VAE = VAE

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        if self.VAE:
            self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.fc_mean(h)
        if self.VAE:
            log_var = self.fc_var(h)
            return mean, log_var
        return mean


class ContextFusion(nn.Module):
    """Combines cell and perturbation latent vectors into a context-aware representation."""

    def __init__(self, query_dim: int, key_dim: int, value_dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj   = nn.Linear(key_dim,   hidden_dim)
        self.value_proj = nn.Linear(value_dim,  hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, query, key, value):
        Q = self.query_proj(query)          # [B, hidden]
        K = self.key_proj(key)              # [B, hidden]
        V = self.value_proj(value)          # [B, hidden]

        Q_r = Q.unsqueeze(1)                # [B, 1, hidden]
        K_r = K.unsqueeze(2)                # [B, hidden, 1]

        scores  = torch.bmm(Q_r, K_r) / self.scale             # [B, 1, 1]
        weights = F.softmax(scores.squeeze(), dim=0).unsqueeze(1)  # [B, 1]

        context_vector = weights * V        # [B, hidden]
        return context_vector


class HyperNetwork(nn.Module):
    """
    Generates dynamic weights for the predictor from the
    concatenated (cell_latent || context_vector).
    """

    def __init__(self, cell_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * latent_dim + hidden_dim),
        )

    def forward(self, cell_context_emb):
        return self.weight_generator(cell_context_emb)


class PerturbationEffectPredictor(nn.Module):
    """
    Applies dynamic weights (from HyperNetwork) to the perturbation latent,
    then projects to gene-expression space.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.projection_output = nn.Linear(hidden_dim, output_dim)
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, weights):
        batch_size = x.shape[0]

        W = weights[:, : self.hidden_dim * self.input_dim].view(
            batch_size, self.hidden_dim, self.input_dim
        )
        b = weights[:, self.hidden_dim * self.input_dim :].view(batch_size, self.hidden_dim)

        x_r = x.unsqueeze(1)                                    # [B, 1, latent]
        hidden = torch.bmm(x_r, W.transpose(1, 2)).squeeze(1) + b  # [B, hidden]
        hidden = F.leaky_relu(hidden, 0.1)

        return self.projection_output(hidden)                    # [B, x_dim]


class Net(nn.Module):
    """
    Full HyperMAP network.
        x_dim : number of genes (expression input dimension)
        p_dim : perturbation embedding dimension
    """

    def __init__(
        self,
        x_dim:      int,
        p_dim:      int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        self.CellEncoder = Encoder(
            input_dim=x_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            VAE=False,
        ).to(self.device)

        self.PerturbationEncoder = Encoder(
            input_dim=p_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            VAE=False,
        ).to(self.device)

        self.HyperNetwork = HyperNetwork(
            cell_dim=latent_dim * 2,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        self.Predictor = PerturbationEffectPredictor(
            input_dim=latent_dim,
            output_dim=x_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.ContextFusion = ContextFusion(
            query_dim=latent_dim,
            key_dim=latent_dim,
            value_dim=latent_dim,
            hidden_dim=latent_dim,
        ).to(self.device)

    def forward(self, control_exp, pert_emb):
        cell_emb = self.CellEncoder(control_exp)
        pert_emb = self.PerturbationEncoder(pert_emb)

        context_vector = self.ContextFusion(
            query=cell_emb,
            key=pert_emb,
            value=pert_emb,
        )

        combined = torch.cat([cell_emb, context_vector], dim=1)
        weights  = self.HyperNetwork(combined)

        del context_vector, combined

        pred_changes = self.Predictor(pert_emb, weights)

        return pred_changes, cell_emb, pert_emb
