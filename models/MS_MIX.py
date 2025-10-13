import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityPredictor(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        if input_dim > 2000:
            self.dimension_reducer = nn.Linear(input_dim, 1024)
            self.reduced_dim = 1024
        else:
            self.dimension_reducer = None
            self.reduced_dim = input_dim

        self.padding_size = (num_heads - (self.reduced_dim % num_heads)) % num_heads
        self.padded_dim = self.reduced_dim + self.padding_size

        assert self.padded_dim % num_heads == 0, "Padded input_dim must be divisible by num_heads"
        self.head_dim = self.padded_dim // num_heads

        self.qkv_proj = nn.Linear(self.padded_dim, self.padded_dim * 3)
        self.output_proj = nn.Linear(self.padded_dim, 1)

        self.norm1 = nn.LayerNorm(self.padded_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)

        if self.dimension_reducer is not None:
            x = self.dimension_reducer(x)

        if self.padding_size > 0:
            x = F.pad(x, (0, self.padding_size), "constant", 0)

        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [part.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                   for part in qkv]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.padded_dim)

        output = self.norm1(attn_output + x.unsqueeze(1))
        output = output.mean(dim=1)

        output = self.output_proj(output)
        output = self.tanh(output) * 3

        return output.squeeze(-1)


def compute_kl_loss(I_a, I_t, I_v, labels, epsilon=1e-8):

    # [B, 1] -> [B]
    labels = labels.squeeze(-1)

    # Map the predicted values and labels to the [0,1] interval (original range [-3,3])
    def map_to_01(x):
        abs_x = torch.abs(x)
        return abs_x / 3.0

    I_a_01 = map_to_01(I_a)
    I_t_01 = map_to_01(I_t)
    I_v_01 = map_to_01(I_v)
    labels_01 = map_to_01(labels)

    # Use softmax to create a probability distribution
    def to_prob_distribution(values):
        probs = F.softmax(values, dim=0)
        return probs + epsilon

    prob_a = to_prob_distribution(I_a_01)
    prob_t = to_prob_distribution(I_t_01)
    prob_v = to_prob_distribution(I_v_01)
    prob_labels = to_prob_distribution(labels_01)

    kl_a = F.kl_div(
        input=torch.log(prob_a),
        target=prob_labels,
        reduction='batchmean',
        log_target=False
    )

    kl_t = F.kl_div(
        input=torch.log(prob_t),
        target=prob_labels,
        reduction='batchmean',
        log_target=False
    )

    kl_v = F.kl_div(
        input=torch.log(prob_v),
        target=prob_labels,
        reduction='batchmean',
        log_target=False
    )

    kl_loss = kl_a + kl_t + kl_v

    return kl_loss * 1000


def MS_Mix(f_a, f_t, f_v, labels, intensity_predictors, alpha=2.0, threshold=0.4,
            num_mix=None, gamma=1.0):

    B = f_a.size(0)
    device = f_a.device

    num_mix = num_mix if num_mix is not None else B


    I_a = intensity_predictors['audio'](f_a).squeeze(-1)  # [B]
    I_t = intensity_predictors['text'](f_t).squeeze(-1)
    I_v = intensity_predictors['vision'](f_v).squeeze(-1)

    kl_loss = compute_kl_loss(I_a, I_t, I_v, labels)

    w_a = torch.abs(I_a)
    w_t = torch.abs(I_t)
    w_v = torch.abs(I_v)

    w_a = (w_a - w_a.min()) / (w_a.max() - w_a.min() + 1e-8)
    w_t = (w_t - w_t.min()) / (w_t.max() - w_t.min() + 1e-8)
    w_v = (w_v - w_v.min()) / (w_v.max() - w_v.min() + 1e-8)

    f_a_norm = F.normalize(f_a, p=2, dim=1)
    f_t_norm = F.normalize(f_t, p=2, dim=1)
    f_v_norm = F.normalize(f_v, p=2, dim=1)

    # Compute the similarity matrix
    sim_aa = torch.mm(f_a_norm, f_a_norm.t())
    sim_tt = torch.mm(f_t_norm, f_t_norm.t())
    sim_vv = torch.mm(f_v_norm, f_v_norm.t())
    total_sim = (sim_aa + sim_tt + sim_vv) / 3.0
    total_sim = total_sim.fill_diagonal_(0)  # 清除对角线

    mask = torch.ones_like(total_sim, dtype=torch.bool)
    mask.fill_diagonal_(False)

    # Retrieve off-diagonal elements
    non_diag_values = total_sim[mask]

    if len(non_diag_values) > 0:
        min_val = non_diag_values.min()
        max_val = non_diag_values.max()

        if max_val - min_val > 1e-8:
            total_sim_normalized = (total_sim - min_val) / (max_val - min_val)
        else:
            total_sim_normalized = torch.ones_like(total_sim) * 0.5
    else:
        total_sim_normalized = total_sim.clone()

    total_sim = total_sim_normalized.fill_diagonal_(0)

    valid_pairs_mask = (total_sim > threshold)
    valid_pairs = torch.nonzero(valid_pairs_mask, as_tuple=False)

    if len(valid_pairs) < num_mix:
        triu_mask = torch.triu(torch.ones_like(total_sim), diagonal=1).bool()
        valid_sim = torch.where(triu_mask, total_sim, torch.tensor(-1.0).to(device))
        ss = valid_sim.flatten()
        _, topk_indices = torch.topk(ss, int(2 * num_mix))
        valid_pairs = torch.stack([
            topk_indices // B,
            topk_indices % B
        ], dim=1)

    selected_indices = torch.randperm(len(valid_pairs))[:num_mix]
    selected_pairs = valid_pairs[selected_indices]
    idx1 = selected_pairs[:, 0]
    idx2 = selected_pairs[:, 1]

    lam_base = torch.distributions.Beta(alpha, alpha).sample([num_mix]).to(device)

    w_a1, w_a2 = w_a[idx1], w_a[idx2]
    w_t1, w_t2 = w_t[idx1], w_t[idx2]
    w_v1, w_v2 = w_v[idx1], w_v[idx2]

    lam_a_adaptive = (w_a1 + 1e-8) / (w_a1 + w_a2 + 2e-8)
    lam_t_adaptive = (w_t1 + 1e-8) / (w_t1 + w_t2 + 2e-8)
    lam_v_adaptive = (w_v1 + 1e-8) / (w_v1 + w_v2 + 2e-8)

    lam_a = (lam_base + lam_a_adaptive) / 2
    lam_t = (lam_base + lam_t_adaptive) / 2
    lam_v = (lam_base + lam_v_adaptive) / 2

    lam_label = (lam_a + lam_t + lam_v) / 3


    lam_expanded_a = lam_a.view(-1, *([1] * (f_a.dim() - 1)))
    lam_expanded_t = lam_t.view(-1, *([1] * (f_t.dim() - 1)))
    lam_expanded_v = lam_v.view(-1, *([1] * (f_v.dim() - 1)))
    lam_expanded_labels = lam_label.view(-1, *([1] * (labels.dim() - 1))) if labels.dim() > 1 else lam_label

    # mix features
    mixed_f_a = lam_expanded_a * f_a[idx1] + (1 - lam_expanded_a) * f_a[idx2]
    mixed_f_t = lam_expanded_t * f_t[idx1] + (1 - lam_expanded_t) * f_t[idx2]
    mixed_f_v = lam_expanded_v * f_v[idx1] + (1 - lam_expanded_v) * f_v[idx2]
    mixed_labels = lam_expanded_labels * labels[idx1] + (1 - lam_expanded_labels) * labels[idx2]

    # combine original features and mixed features
    combined_f_a = torch.cat([f_a, mixed_f_a], dim=0)
    combined_f_t = torch.cat([f_t, mixed_f_t], dim=0)
    combined_f_v = torch.cat([f_v, mixed_f_v], dim=0)
    combined_labels = torch.cat([labels, mixed_labels], dim=0)

    return combined_f_a, combined_f_t, combined_f_v, labels[idx1], labels[idx2], lam_expanded_labels, combined_labels, kl_loss


# Three-dimensional features
def MS_Mix_threeD(f_a, f_t, f_v, labels, intensity_predictors, alpha=2.0, threshold=0.4, num_mix=None):
    B = f_a.size(0)
    device = f_a.device

    num_mix = num_mix if num_mix is not None else B

    I_a = intensity_predictors['audio'](f_a)
    I_t = intensity_predictors['text'](f_t)
    I_v = intensity_predictors['vision'](f_v)

    if I_a.dim() > 1:
        I_a = (I_a.max(dim=1))[0]
    if I_t.dim() > 1:
        I_t = (I_t.max(dim=1))[0]
    if I_v.dim() > 1:
        I_v = (I_v.max(dim=1))[0]

    I_a = I_a.squeeze(-1)  # [B]
    I_t = I_t.squeeze(-1)
    I_v = I_v.squeeze(-1)

    kl_loss = compute_kl_loss(I_a, I_t, I_v, labels)

    # Flatten three-dimensional features into two dimensions to calculate similarity.
    f_a_flat = f_a.view(B, -1)
    f_t_flat = f_t.view(B, -1)
    f_v_flat = f_v.view(B, -1)

    f_a_norm = F.normalize(f_a_flat, p=2, dim=1)
    f_t_norm = F.normalize(f_t_flat, p=2, dim=1)
    f_v_norm = F.normalize(f_v_flat, p=2, dim=1)

    sim_aa = torch.mm(f_a_norm, f_a_norm.t())
    sim_tt = torch.mm(f_t_norm, f_t_norm.t())
    sim_vv = torch.mm(f_v_norm, f_v_norm.t())
    total_sim = (sim_aa + sim_tt + sim_vv) / 3.0


    mask = torch.ones_like(total_sim, dtype=torch.bool)
    mask.fill_diagonal_(False)

    non_diag_values = total_sim[mask]

    if len(non_diag_values) > 0:
        min_val = non_diag_values.min()
        max_val = non_diag_values.max()

        if max_val - min_val > 1e-8:
            total_sim_normalized = (total_sim - min_val) / (max_val - min_val)
        else:
            total_sim_normalized = torch.ones_like(total_sim) * 0.5
    else:
        total_sim_normalized = total_sim.clone()

    total_sim = total_sim_normalized.fill_diagonal_(0)

    valid_pairs_mask = (total_sim > threshold)
    valid_pairs = torch.nonzero(valid_pairs_mask, as_tuple=False)

    if len(valid_pairs) < num_mix:
        triu_mask = torch.triu(torch.ones_like(total_sim), diagonal=1).bool()
        valid_sim = torch.where(triu_mask, total_sim, torch.tensor(-1.0).to(device))
        ss = valid_sim.flatten()
        _, topk_indices = torch.topk(ss, int(2 * num_mix))
        valid_pairs = torch.stack([
            topk_indices // B,
            topk_indices % B
        ], dim=1)

    selected_indices = torch.randperm(len(valid_pairs))[:num_mix]
    selected_pairs = valid_pairs[selected_indices]
    idx1 = selected_pairs[:, 0]
    idx2 = selected_pairs[:, 1]

    lam_base = torch.distributions.Beta(alpha, alpha).sample([num_mix]).to(device)  # [num_mix]

    w_a1, w_a2 = I_a[idx1], I_a[idx2]  # [num_mix]
    w_t1, w_t2 = I_t[idx1], I_t[idx2]
    w_v1, w_v2 = I_v[idx1], I_v[idx2]

    lam_a_adaptive = (w_a1 + 1e-8) / (w_a1 + w_a2 + 2e-8)
    lam_t_adaptive = (w_t1 + 1e-8) / (w_t1 + w_t2 + 2e-8)
    lam_v_adaptive = (w_v1 + 1e-8) / (w_v1 + w_v2 + 2e-8)

    lam_a = (lam_base + lam_a_adaptive) / 2
    lam_t = (lam_base + lam_t_adaptive) / 2
    lam_v = (lam_base + lam_v_adaptive) / 2

    lam_label = (lam_a + lam_t + lam_v) / 3

    # Extended dimensions for feature mixing
    # [B, T, D] → [num_mix, 1, 1]
    lam_expanded_a = lam_a.view(-1, 1, 1)
    lam_expanded_t = lam_t.view(-1, 1, 1)
    lam_expanded_v = lam_v.view(-1, 1, 1)

    if labels.dim() > 1:
        lam_expanded_labels = lam_label.view(-1, *([1] * (labels.dim() - 1)))
    else:
        lam_expanded_labels = lam_label

    mixed_f_a = lam_expanded_a * f_a[idx1] + (1 - lam_expanded_a) * f_a[idx2]
    mixed_f_t = lam_expanded_t * f_t[idx1] + (1 - lam_expanded_t) * f_t[idx2]
    mixed_f_v = lam_expanded_v * f_v[idx1] + (1 - lam_expanded_v) * f_v[idx2]
    mixed_labels = lam_expanded_labels * labels[idx1] + (1 - lam_expanded_labels) * labels[idx2]

    combined_f_a = torch.cat([f_a, mixed_f_a], dim=0)
    combined_f_t = torch.cat([f_t, mixed_f_t], dim=0)
    combined_f_v = torch.cat([f_v, mixed_f_v], dim=0)
    combined_labels = torch.cat([labels, mixed_labels], dim=0)

    return combined_f_a, combined_f_t, combined_f_v, labels[idx1], labels[
        idx2], lam_expanded_labels, combined_labels, kl_loss