import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(
    G,
    D,
    dataloader,
    latent_dim=128,
    num_epochs=100,
    lr=2e-4,
    beta1=0.5,
    device=None,
    save_path="checkpoints"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_path, exist_ok=True)
    samples_dir = os.path.join(save_path, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    G = G.to(device)
    D = D.to(device)

    optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, latent_dim, device=device)
    best_lossG = float("inf")

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()
        epoch_lossG = 0.0
        epoch_lossD = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")

        for real_images, _ in pbar:
            real_images = real_images.to(device)
            b = real_images.size(0)

            real_labels = torch.full((b,), 0.9, device=device)
            fake_labels = torch.zeros((b,), device=device)

            # --- Train D ---
            optD.zero_grad()
            real_logits = D(real_images)
            lossD_real = criterion(real_logits, real_labels)

            z = torch.randn(b, latent_dim, device=device)
            fake_images = G(z)
            fake_logits = D(fake_images.detach())
            lossD_fake = criterion(fake_logits, fake_labels)

            lossD = 0.5 * (lossD_real + lossD_fake)
            lossD.backward()
            optD.step()

            # --- Train G ---
            optG.zero_grad()
            z = torch.randn(b, latent_dim, device=device)
            fake_images = G(z)
            fake_logits_forG = D(fake_images)
            lossG = criterion(fake_logits_forG, real_labels)
            lossG.backward()
            optG.step()

            epoch_lossD += lossD.item()
            epoch_lossG += lossG.item()
            pbar.set_postfix(lossD=f"{lossD.item():.4f}", lossG=f"{lossG.item():.4f}")

        avg_lossG = epoch_lossG / max(1, len(dataloader))
        avg_lossD = epoch_lossD / max(1, len(dataloader))

        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizerG_state_dict": optG.state_dict(),
            "optimizerD_state_dict": optD.state_dict(),
            "lossG": avg_lossG,
            "lossD": avg_lossD,
        }, os.path.join(save_path, "last_checkpoint.pth"))

        if avg_lossG < best_lossG:
            best_lossG = avg_lossG
            torch.save({
                "epoch": epoch,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "optimizerG_state_dict": optG.state_dict(),
                "optimizerD_state_dict": optD.state_dict(),
                "lossG": best_lossG,
                "lossD": avg_lossD,
            }, os.path.join(save_path, "best_checkpoint.pth"))
            print(f"Best checkpoint saved at Epoch {epoch}, LossG={best_lossG:.4f}")

        print(f"[Epoch {epoch}/{num_epochs}] Avg Loss_D: {avg_lossD:.4f}, Avg Loss_G: {avg_lossG:.4f}")

    return G, D
