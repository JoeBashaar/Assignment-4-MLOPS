"""
GAN Training Script – Fashion-MNIST  (MLflow-instrumented)
Runs 5 experiments with varied hyperparameters and logs everything to MLflow.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for Docker
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
import os

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── MLflow Setup ──
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment3_GAN_StudentA")


# ── Model definitions ──
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Training helpers ──
def train_discriminator(G, D, criterion, opt_D, real, real_labels, fake_labels, latent_dim):
    noise = torch.randn(real.size(0), latent_dim, device=device)
    fake = G(noise).detach()
    loss = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)
    opt_D.zero_grad()
    loss.backward()
    opt_D.step()
    return loss.item(), fake


def train_generator(G, D, criterion, opt_G, bs, real_labels, latent_dim):
    noise = torch.randn(bs, latent_dim, device=device)
    loss = criterion(D(G(noise)), real_labels)
    opt_G.zero_grad()
    loss.backward()
    opt_G.step()
    return loss.item()


@torch.no_grad()
def compute_accuracy(D, real, fake):
    real_correct = (D(real) >= 0.5).sum().item()
    fake_correct = (D(fake) < 0.5).sum().item()
    return real_correct, fake_correct


# ── Load dataset ──
df = pd.read_csv("fashion-mnist_train.csv")
X = df.drop("label", axis=1).values.astype("float32")
X = (X / 255.0) * 2 - 1
X_tensor = torch.tensor(X)
print(f"Dataset loaded — {len(X_tensor)} samples")


# ── 5 experiment configurations ──
configs = [
    {"lr": 0.0002, "batch_size": 512, "epochs": 5, "latent_dim": 100},
    {"lr": 0.001,  "batch_size": 512, "epochs": 5, "latent_dim": 100},
    {"lr": 0.0001, "batch_size": 512, "epochs": 5, "latent_dim": 100},
    {"lr": 0.0002, "batch_size": 256, "epochs": 5, "latent_dim": 100},
    {"lr": 0.0002, "batch_size": 128, "epochs": 5, "latent_dim": 100},
]

for run_idx, cfg in enumerate(configs, 1):
    LR         = cfg["lr"]
    BATCH_SIZE = cfg["batch_size"]
    EPOCHS     = cfg["epochs"]
    LATENT_DIM = cfg["latent_dim"]

    G = Generator(latent_dim=LATENT_DIM).to(device)
    D = Discriminator().to(device)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)

    with mlflow.start_run(run_name=f"run_{run_idx}_lr{LR}_bs{BATCH_SIZE}") as run:
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("latent_dim", LATENT_DIM)
        mlflow.log_param("optimizer", "Adam")

        mlflow.set_tag("student_id", "StudentA")
        mlflow.set_tag("model_type", "Vanilla_GAN")
        mlflow.set_tag("dataset", "Fashion-MNIST")

        print(f"\n{'='*60}")
        print(f"RUN {run_idx}/5 — lr={LR}, batch_size={BATCH_SIZE}")
        print(f"{'='*60}")

        for epoch in range(EPOCHS):
            d_real_correct, d_fake_correct, total = 0, 0, 0

            for (imgs,) in loader:
                real = imgs.to(device)
                bs = real.size(0)
                real_labels = torch.ones(bs, 1, device=device)
                fake_labels = torch.zeros(bs, 1, device=device)

                loss_D, fake = train_discriminator(
                    G, D, criterion, opt_D, real, real_labels, fake_labels, LATENT_DIM
                )
                loss_G = train_generator(
                    G, D, criterion, opt_G, bs, real_labels, LATENT_DIM
                )

                rc, fc = compute_accuracy(D, real, fake)
                d_real_correct += rc
                d_fake_correct += fc
                total += bs

            acc_real = d_real_correct / total * 100
            acc_fake = d_fake_correct / total * 100
            acc_overall = (d_real_correct + d_fake_correct) / (2 * total) * 100

            mlflow.log_metric("d_loss", loss_D, step=epoch + 1)
            mlflow.log_metric("g_loss", loss_G, step=epoch + 1)
            mlflow.log_metric("acc_real", acc_real, step=epoch + 1)
            mlflow.log_metric("acc_fake", acc_fake, step=epoch + 1)
            mlflow.log_metric("d_accuracy", acc_overall, step=epoch + 1)

            print(
                f"  Epoch {epoch+1}/{EPOCHS} | D_loss: {loss_D:.4f} | "
                f"G_loss: {loss_G:.4f} | D_Acc: {acc_overall:.1f}%"
            )

        mlflow.log_metric("final_d_loss", loss_D)
        mlflow.log_metric("final_g_loss", loss_G)
        mlflow.log_metric("final_d_accuracy", acc_overall)

        mlflow.pytorch.log_model(G, "generator_model")
        print(f"  ✓ Run {run_idx} logged to MLflow")

        # Export Run ID for the pipeline
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

print("\n✅ All 5 runs completed and logged to MLflow!")
print(f"✅ model_info.txt written with last Run ID")

