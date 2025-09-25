import torch
import torch.nn as nn
import torch.optim as optim
import os

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.binary_cross_entropy(
        recon_x, x, reduction="sum"
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(model, dataloader, optimizer, device, epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")

    avg_loss = train_loss / len(dataloader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss

def validate_vae(model, dataloader, device, epoch, optimizer=None, save_path="vae_checkpoints", best_val_loss=None):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    avg_loss = val_loss / len(dataloader.dataset)
    print(f"====> Validation set loss: {avg_loss:.4f}")

    os.makedirs(save_path, exist_ok=True)

    if best_val_loss is None:
        best_val_loss = float("inf")

    saved = False
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_loss,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(ckpt, os.path.join(save_path, "best_checkpoint.pth"))
        print(f"Best checkpoint saved: {os.path.join(save_path, 'best_checkpoint.pth')} (val_loss={avg_loss:.6f})")
        saved = True

    ckpt_last = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": avg_loss,
    }
    if optimizer is not None:
        ckpt_last["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt_last, os.path.join(save_path, "last_checkpoint.pth"))

    return avg_loss, best_val_loss, saved
