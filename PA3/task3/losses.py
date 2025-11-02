# losses.py
import torch 
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Implements the Knowledge Distillation loss, combining hard label loss (Cross-Entropy)
    and soft label loss (KL Divergence).
    """
    def __init__(self, alpha, temperature):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        
        # Standard Cross-Entropy for the hard labels
        self.hard_loss_fn = nn.CrossEntropyLoss()
        
        # KL Divergence for the soft labels
        # 'reduction="batchmean"' averages the loss over the batch
        self.soft_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, teacher_outputs, labels):
        """
        Calculates the total distillation loss.
        :param student_outputs: Raw outputs (can be tuple or tensor) from the student model.
        :param teacher_outputs: Raw outputs (can be tuple or tensor) from the teacher model.
        :param labels: The ground truth labels.
        :return: The combined distillation loss.
        """
        # --- FIX: Unpack logits if the model outputs are tuples ---
        s_logits = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        t_logits = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
        
        # 1. Calculate the hard loss using only the logits
        hard_loss = self.hard_loss_fn(s_logits, labels)

        # 2. Calculate the soft loss using only the logits
        soft_student_probs = F.log_softmax(s_logits / self.temperature, dim=1)
        soft_teacher_probs = F.softmax(t_logits / self.temperature, dim=1)
        
        distillation_loss = self.soft_loss_fn(soft_student_probs, soft_teacher_probs) * (self.temperature ** 2)

        # 3. Combine the two losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * distillation_loss
        
        return total_loss
    
class DKDLoss(nn.Module):
    """
    Implements the original Decoupled Knowledge Distillation loss.
    This is a faithful implementation of the paper's formulation.
    Source: https://arxiv.org/abs/2203.08679
    """
    def __init__(self, beta, gamma, temperature):
        super(DKDLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.eps = 1e-8 # Epsilon for numerical stability

    def forward(self, student_outputs, teacher_outputs, labels):
        
        # --- FIX: Unpack logits if the model outputs are tuples ---
        s_logits = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        t_logits = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs

        # Get softened probabilities from logits
        soft_teacher_probs = F.softmax(t_logits / self.temperature, dim=1)
        soft_student_log_probs = F.log_softmax(s_logits / self.temperature, dim=1)

        # --- TCKD: Target Class Knowledge Distillation ---
        p_t = torch.gather(soft_teacher_probs, 1, labels.unsqueeze(1)).squeeze(1)
        q_t_log = torch.gather(soft_student_log_probs, 1, labels.unsqueeze(1)).squeeze(1)
        q_t = q_t_log.exp()
        
        teacher_tckd_target = torch.stack([p_t, 1 - p_t], dim=1)
        student_tckd_log_probs = torch.stack([q_t_log, torch.log1p(-q_t + self.eps)], dim=1)
        tckd_loss = self.kld_loss(student_tckd_log_probs, teacher_tckd_target) * (self.temperature ** 2)

        # --- NCKD: Non-Target Class Knowledge Distillation ---
        mask = torch.ones_like(soft_teacher_probs).scatter_(1, labels.unsqueeze(1), 0)
        
        teacher_nt_probs = soft_teacher_probs * mask
        student_nt_log_probs = soft_student_log_probs * mask
        
        teacher_nt_probs_normalized = teacher_nt_probs / (teacher_nt_probs.sum(dim=1, keepdim=True) + self.eps)
        student_nt_log_probs_normalized = student_nt_log_probs - torch.logsumexp(student_nt_log_probs, dim=1, keepdim=True)
        nckd_loss = self.kld_loss(student_nt_log_probs_normalized, teacher_nt_probs_normalized) * (self.temperature ** 2)

        # Combine the losses
        total_loss = (self.beta * tckd_loss) + (self.gamma * nckd_loss)
        
        return total_loss
    

class CRDLoss(nn.Module):
    """
    Implements the full CRD loss, combining standard KD with a contrastive loss on embeddings.
    """
    def __init__(self, alpha, kd_temp, crd_temp, lambda_crd):
        super(CRDLoss, self).__init__()
        self.lambda_crd = lambda_crd
        self.kd_temp = kd_temp
        self.crd_temp = crd_temp # New contrastive temperature
        
        # Logit loss uses the high KD temperature
        self.distillation_loss = DistillationLoss(alpha=alpha, temperature=self.kd_temp)
        self.contrastive_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, student_embedding, teacher_embedding, labels):
        # 1. Calculate the standard distillation loss on the logits (uses high temp)
        logit_loss = self.distillation_loss(student_logits, teacher_logits, labels)

        # 2. Calculate the contrastive loss on the embeddings
        batch_size = student_embedding.size(0)
        
        student_embedding = F.normalize(student_embedding, p=2, dim=1)
        teacher_embedding = F.normalize(teacher_embedding, p=2, dim=1)
        
        similarity_matrix = torch.mm(student_embedding, teacher_embedding.T)
        
        # --- THIS IS THE KEY CHANGE ---
        # Scale the similarities by the LOW contrastive temperature
        similarity_matrix = similarity_matrix / self.crd_temp
        
        contrastive_labels = torch.arange(batch_size, dtype=torch.long, device=student_logits.device)
        
        crd_loss = self.contrastive_loss(similarity_matrix, contrastive_labels)
        
        # 3. Combine the two losses
        total_loss = logit_loss + self.lambda_crd * crd_loss
        return total_loss
    