from math import e
from pathlib import Path

import numpy as np
import torch

from pep_compass.geodesic_search.load_predictor_models import (
    load_MIC_predictor_model,
    load_predictor_models,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


class Geodesics:
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        device="cpu",
        points_density=90,
        temp=0.3,
        lr=0.001,
        convergence_tol=1e-3,
        patience=100,
        metric=["logsoftmax"],
        weight_ambient_dist=1.0,
        metric_weight=1.0,
        weight_latent_dist=0.0,
        weight_var=1.0,
        models_path="models",
    ):

        self.points_density = points_density
        self.device = device
        self.encoder_decoder = encoder_decoder
        self.temp = temp
        self.lr = lr
        self.convergence_tol = convergence_tol
        self.patience = patience
        self.metric = metric
        self.metric_weight = metric_weight
        self.weight_ambient_dist = weight_ambient_dist
        self.weight_latent_dist = weight_latent_dist
        self.antech, _ = load_predictor_models(
            Trained_predictor_models_path=Path(models_path) / "Trained_predictor_models",
            device=self.device,
        )
        self.antex = load_MIC_predictor_model(
            Trained_predictor_models_path=Path(models_path) / "Trained_predictor_models",
            device=self.device,
        )

        self.weight_var = weight_var
        self.field = None
        self.sanity_check = False

    def calculate_dist(self):
        """
        Calculate the loss based on the distance between decoded points.
        """

        if self.sanity_check:
            decoded_points = self.encoder_decoder.decoder_forward(x=self.points)
            diffs = decoded_points[1:] - decoded_points[:-1]

            latent_diffs = self.points[1:] - self.points[:-1]
            latent_dist = torch.norm(latent_diffs, dim=1).sum()

            mean = torch.mean(diffs)
            var = torch.mean((diffs - mean) ** 2)

            self.loss = (
                torch.norm(diffs, dim=1).sum()
                + self.weight_latent_dist * latent_dist
                + self.weight_var * var
            )

            if self.field is not None:
                self.loss += self.alpha * self.field(decoded_points).sum()

            return

        decoded_logsoftmax = self.encoder_decoder.decoder_forward(x=self.points, log_softmax=True, softmax=False, flatten=True)

        diffs = decoded_logsoftmax[1:] - decoded_logsoftmax[:-1]

        # Flags for presence of models
        has_antex = "antex" in self.metric
        has_antech = "antech" in self.metric

        # Initialize list for model outputs
        preds_list = []

        if has_antex:
            self.antex.eval()
            preds_antex = self.antex(decoded_logsoftmax)[:, 1:4]
            preds_list.append(preds_antex[:-1])

        if has_antech:
            self.antech.eval()
            preds_antech = self.antech(decoded_logsoftmax)
            diffs_preds = preds_antech[1:] - preds_antech[:-1]
            preds_list.append(diffs_preds)

        if preds_list:
            preds_combined = sum(preds_list) if len(preds_list) == 2 else preds_list[0]
            merged_points = torch.cat(
                [diffs, self.metric_weight * preds_combined], dim=1
            )
        else:
            merged_points = diffs  # Only diffs, no model output

        latent_diffs = self.points[1:] - self.points[:-1]
        latent_dist = torch.norm(latent_diffs, dim=1).sum()

        mean = torch.mean(diffs)

        self.ambient_dist = torch.norm(merged_points, dim=1).sum()
        self.var = torch.mean((diffs - mean) ** 2)

        self.loss = (
            self.weight_ambient_dist * self.ambient_dist
            + self.weight_latent_dist * latent_dist
            + self.weight_var * self.var
        )

    def get_latent_dist(self):
        """
        Compute the latent space distance between consecutive points.

        Returns:
            latent_dist (float): The total latent distance.
        """
        with torch.no_grad():
            latent_diffs = self.points[1:] - self.points[:-1]
            latent_dist = torch.norm(latent_diffs, dim=1).sum()

        return latent_dist

    def find(self, begin, end, num_iterations=200, verbose=False):

        self.n_points = int(self.points_density * torch.dist(end, begin, p=2))

        points = torch.stack(
            [
                begin + (i / (self.n_points - 1)) * (end - begin)
                for i in range(self.n_points)
            ],
            dim=0,
        )
        self.points = torch.nn.Parameter(points, requires_grad=True)
        optimizer = torch.optim.Adam([self.points], lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=self.patience
        )

        prev_loss = float("inf")
        no_improvement_iters = 0

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            self.calculate_dist()
            self.loss.backward()

            with torch.no_grad():
                self.points.grad[0, :] = 0
                self.points.grad[-1, :] = 0

                self.points[0, :] = begin
                self.points[-1, :] = end

            optimizer.step()

            scheduler.step(self.loss.item())

            loss_change = abs(prev_loss - self.loss.item())
            if loss_change < self.convergence_tol:
                no_improvement_iters += 1
            else:
                no_improvement_iters = 0

            prev_loss = self.loss.item()

            if verbose:
                latent_dist = self.get_latent_dist().item() * self.weight_latent_dist
                loss = self.loss.item()
                ambient_dist = self.ambient_dist.item() * self.weight_ambient_dist
                lr = scheduler.get_last_lr()
                var = self.var.item() * self.weight_var
                print(
                    f"{iteration=}, {loss=:.2f}, {ambient_dist=:.2f}, {latent_dist=:.2f}, {var=:.2f}, {lr=}"
                )

            if no_improvement_iters >= self.patience:
                if verbose:
                    print(
                        f"Stopping early at iteration {iteration}: No significant improvement for {self.patience} iterations."
                    )
                break
        return self.points, self.ambient_dist.item()
