import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class IRMTrainer:
    def __init__(self, model, optimizer, irm_lambda=1.0, penalty_anneal_iters=500, device=None):
        self.model = model
        self.optimizer = optimizer
        self.irm_lambda = float(irm_lambda)
        self.penalty_anneal_iters = int(penalty_anneal_iters)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.iteration = 0

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad.pow(2))

    def step(self, env_batches):
        self.model.train()
        nll = 0.0
        penalty = 0.0

        for x, y in env_batches:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(env_batches)
        penalty /= len(env_batches)

        penalty_weight = self.irm_lambda if self.iteration >= self.penalty_anneal_iters else 1.0
        loss = nll + penalty_weight * penalty
        if penalty_weight > 1.0:
            loss = loss / penalty_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1

        return {
            "loss": loss.item(),
            "nll": nll.item(),
            "penalty": penalty.item(),
            "penalty_weight": penalty_weight
        }

def freeze_backbone_for_sequential(model):
    for name, param in model.named_parameters():
        if name.startswith("0.layer4") or name.startswith("2."):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


class RemapLabels(torch.utils.data.Dataset):
    def __init__(self, dataset, class_to_idx):
        self.dataset = dataset
        self.classes = list(class_to_idx.keys())
        self.class_to_idx = class_to_idx
        self.idx_mapping = [-1] * len(self.dataset.classes)
        for old_idx, class_name in enumerate(self.dataset.classes):
            if class_name in self.class_to_idx:
                self.idx_mapping[old_idx] = self.class_to_idx[class_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, old_label_idx = self.dataset[idx]
        new_label_idx = self.idx_mapping[old_label_idx]
        if new_label_idx == -1:
            raise ValueError(f"Class '{self.dataset.classes[old_label_idx]}' not in canonical map.")
        return image, new_label_idx
