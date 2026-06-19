#!/usr/bin/env python3
"""
Parameter count verification for MICCAI 2026 paper comparison table.

Defines modules inline to avoid dependency issues, then counts parameters.
All architectures match the actual source code exactly.
"""

import torch
import torch.nn as nn
import math
from functools import partial


def count_params(module):
    """Count total parameters."""
    return sum(p.numel() for p in module.parameters())


def fmt(n):
    """Format parameter count."""
    if n >= 1e6:
        return f"{n:>12,} ({n/1e6:.2f}M)"
    elif n >= 1e3:
        return f"{n:>12,} ({n/1e3:.1f}K)"
    return f"{n:>12,}"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================================
# Module Definitions (matching source code exactly)
# ============================================================================

# --- From vits.py ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

class Block(nn.Module):
    """MedST's ViT block: has DUAL MLPs (frontal mlp + lateral mlp_l)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.mlp_l = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)  # lateral MLP
        self.norm2_l = norm_layer(dim)  # lateral norm

class VisionTransformer(nn.Module):
    """MedST's modified ViT-Base with dual MLPs per block."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2  # 196
        self.patch_embed_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)


# --- From encoder.py ---
class GlobalEmbedding(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

class LocalEmbedding(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim, affine=False),
        )


# --- From anatomy_temporal_finetuner.py ---
class ROIProjection(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

class AnatomyClassifier(nn.Module):
    def __init__(self, n_input=384, n_classes=3, n_hidden=256, p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_input, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden // 2, bias=False),
            nn.BatchNorm1d(n_hidden // 2),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden // 2, n_classes),
        )

class AttentionROIPool(nn.Module):
    def __init__(self, dim=768, hidden_dim=192):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )


# --- From anatomy_modules.py ---
class CrossImageRegionalDiffAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)


# ============================================================================
# Count Parameters
# ============================================================================

section("1. ViT-Base Backbone")

vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
vit_total = count_params(vit)

# Standard DeiT-base (single MLP per block) — compute by subtraction
block0 = vit.blocks[0]
mlp_l_params = count_params(block0.mlp_l)
norm2_l_params = count_params(block0.norm2_l)
lateral_extra_per_block = mlp_l_params + norm2_l_params
lateral_extra_total = lateral_extra_per_block * 12

standard_vit_total = vit_total - lateral_extra_total

print(f"  Standard DeiT-base (1 MLP/block): {fmt(standard_vit_total)}")
print(f"  MedST ViT (dual MLPs/block):      {fmt(vit_total)}")
print(f"  Lateral MLP extra (12 blocks):     {fmt(lateral_extra_total)}")

# Detailed breakdown
patch_embed = count_params(vit.patch_embed_proj)
cls_token = vit.cls_token.numel()
pos_embed = vit.pos_embed.numel()
norm_final = count_params(vit.norm)
block_total = count_params(block0)
attn_params = count_params(block0.attn)
mlp_params = count_params(block0.mlp)

print(f"\n  Component breakdown:")
print(f"    PatchEmbed (Conv2d):         {fmt(patch_embed)}")
print(f"    cls_token:                   {fmt(cls_token)}")
print(f"    pos_embed:                   {fmt(pos_embed)}")
print(f"    Final LayerNorm:             {fmt(norm_final)}")
print(f"    Per block (total):           {fmt(block_total)}")
print(f"      Attention (QKV+proj):      {fmt(attn_params)}")
print(f"      MLP frontal:               {fmt(mlp_params)}")
print(f"      MLP lateral:               {fmt(mlp_l_params)}")
print(f"    12 blocks total:             {fmt(block_total * 12)}")


section("2. Projection Heads")

ge = GlobalEmbedding(768, 2048, 128)
le = LocalEmbedding(768, 2048, 128)

ge_total = count_params(ge)
le_total = count_params(le)

print(f"  GlobalEmbedding(768→2048→128): {fmt(ge_total)}")
print(f"  LocalEmbedding(768→2048→128):  {fmt(le_total)}")

# Breakdown
print(f"\n  GlobalEmbedding breakdown:")
for name, param in ge.named_parameters():
    print(f"    {name}: {list(param.shape)} = {fmt(param.numel())}")

four_proj = ge_total * 2 + le_total * 2
print(f"\n  4 heads (2×Global + 2×Local):  {fmt(four_proj)}")
two_proj = ge_total + le_total
print(f"  2 heads (1×Global + 1×Local):  {fmt(two_proj)}")


section("3. Full Encoders")

# ImageEncoder = ViT + 4 proj heads
img_enc_total = vit_total + four_proj
print(f"  ImageEncoder (MedST ViT + 4 proj):")
print(f"    ViT backbone:                {fmt(vit_total)}")
print(f"    4 projection heads:          {fmt(four_proj)}")
print(f"    Total:                       {fmt(img_enc_total)}")

# Also compute with standard ViT for comparison
img_enc_standard = standard_vit_total + four_proj
print(f"\n  If using standard DeiT-base:")
print(f"    Standard ViT + 4 proj:       {fmt(img_enc_standard)}")

# BertEncoder = BERT-base + 2 proj heads
# Standard BERT-base: 109,482,240 (from HuggingFace)
# BioClinicalBERT is same arch as BERT-base
bert_base_params = 109_482_240  # well-known value
bert_enc_total = bert_base_params + two_proj
print(f"\n  BertEncoder (BioClinicalBERT + 2 proj):")
print(f"    BERT-base:                   {fmt(bert_base_params)}")
print(f"    2 projection heads:          {fmt(two_proj)}")
print(f"    Total:                       {fmt(bert_enc_total)}")


section("4. Fine-tuning Layers (Our Method)")

roi_proj = ROIProjection(768, 512, 128)
classifier = AnatomyClassifier(384, 3, 256, 0.1)
attn_pool = AttentionROIPool(768, 192)
anat_emb = nn.Embedding(39, 128)
crda = CrossImageRegionalDiffAttention(128, 4, 0.1)

roi_total = count_params(roi_proj)
cls_total = count_params(classifier)
attn_total = count_params(attn_pool)
emb_total = count_params(anat_emb)
crda_total = count_params(crda)

print(f"  Required modules:")
print(f"    ROIProjection(768→512→128):    {fmt(roi_total)}")
for name, p in roi_proj.named_parameters():
    print(f"      {name}: {list(p.shape)} = {fmt(p.numel())}")

print(f"    AnatomyClassifier(384→256→128→3): {fmt(cls_total)}")
for name, p in classifier.named_parameters():
    print(f"      {name}: {list(p.shape)} = {fmt(p.numel())}")

required_total = roi_total + cls_total
print(f"    Required total:                {fmt(required_total)}")

print(f"\n  Optional modules:")
print(f"    AttentionROIPool(768→192→1):   {fmt(attn_total)}")
for name, p in attn_pool.named_parameters():
    print(f"      {name}: {list(p.shape)} = {fmt(p.numel())}")

print(f"    AnatomyEmbedding(39, 128):     {fmt(emb_total)}")
print(f"    CRDA(128, 4heads):             {fmt(crda_total)}")
for name, p in crda.named_parameters():
    print(f"      {name}: {list(p.shape)} = {fmt(p.numel())}")

optional_total = attn_total + emb_total + crda_total
print(f"    Optional total:                {fmt(optional_total)}")
all_new = required_total + optional_total
print(f"    All new layers:                {fmt(all_new)}")


section("5. MGCA Additional Components")

# MGCA has prototype layer and attention for prototype
mha = nn.MultiheadAttention(128, 1, batch_first=True)
mha_total = count_params(mha)
proto = nn.Linear(128, 500, bias=False)
proto_total = count_params(proto)
print(f"  nn.MultiheadAttention(128, 1):   {fmt(mha_total)}")
print(f"  prototype_layer(128→500):        {fmt(proto_total)}")
print(f"  MGCA extras total:               {fmt(mha_total + proto_total)}")


# ============================================================================
# VERIFICATION
# ============================================================================
section("VERIFICATION vs. PLAN")

checks = [
    ("GlobalEmbedding(768→2048→128)", ge_total, 1_841_280),
    ("LocalEmbedding(768→2048→128)", le_total, 1_841_280),
    ("ROIProjection(768→512→128)", roi_total, 460_416),
    ("AnatomyClassifier(384→256→128→3)", cls_total, 132_227),
    ("AttentionROIPool(768→192→1)", attn_total, 147_841),
    ("AnatomyEmbedding(39,128)", emb_total, 4_992),
    ("CRDA(128, 4heads)", crda_total, 66_048),
    ("Standard DeiT-base", standard_vit_total, 85_798_656),
]

all_pass = True
for name, actual, expected in checks:
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected:
        all_pass = False
    diff_str = "" if actual == expected else f"  DIFF={actual-expected:+,}"
    print(f"  [{status}] {name}: actual={actual:,} expected={expected:,}{diff_str}")

print()
if all_pass:
    print("  All checks PASSED!")
else:
    print("  Some checks FAILED — review differences above.")


# ============================================================================
# PAPER COMPARISON TABLE
# ============================================================================
section("PAPER COMPARISON TABLE")

# Key decision: report standard DeiT-base (85.8M) or MedST ViT (142.5M)?
# The lateral MLPs are loaded from checkpoint but only the frontal path is
# used in inference. For fair comparison, we should report the FULL model
# that gets loaded (142.5M for MedST ViT).

medst_total = img_enc_total + bert_enc_total
mgca_total = img_enc_total + bert_enc_total + mha_total + proto_total

# But wait — does MGCA use the same dual-MLP ViT? NO.
# MGCA uses standard ViT-base. The dual MLPs are MedST's modification.
mgca_img_enc = standard_vit_total + four_proj
mgca_total_correct = mgca_img_enc + bert_enc_total + mha_total + proto_total

print(f"""
NOTE: MedST's ViT has dual MLPs (frontal+lateral) = {vit_total/1e6:.1f}M
      Standard DeiT-base (used by MGCA) = {standard_vit_total/1e6:.1f}M
      During Stage 2 inference, only frontal MLP is used.

┌─────────────────────────────────────────────────────────────────────────────┐
│ Method      │ Vision Enc.     │ Text Enc.    │ Pre-train Total │ FT Added  │
├─────────────┼─────────────────┼──────────────┼─────────────────┼───────────┤
│ BioViL      │ 23.8M (RN50)   │ ~110M (BERT) │ ~134M           │ —         │
│ BioViL-T    │ 25.3M (RN50+T) │ ~110M (BERT) │ ~135M           │ —         │
│ MGCA        │ {mgca_img_enc/1e6:.1f}M (ViT-B) │ {bert_enc_total/1e6:.1f}M   │ {mgca_total_correct/1e6:.1f}M        │ —         │
│ MedST       │ {img_enc_total/1e6:.1f}M(ViT-B*) │ {bert_enc_total/1e6:.1f}M   │ {medst_total/1e6:.1f}M        │ —         │
│ MedSigLIP   │ ~400M (ViT-SO)  │ ~400M        │ ~800M           │ —         │
│ Ours        │ {img_enc_total/1e6:.1f}M(ViT-B*) │ {bert_enc_total/1e6:.1f}M   │ {medst_total/1e6:.1f}M        │ +{required_total/1e6:.2f}M  │
└─────────────┴─────────────────┴──────────────┴─────────────────┴───────────┘

* MedST's ViT-Base has dual MLPs per block (frontal+lateral view support),
  adding ~{lateral_extra_total/1e6:.1f}M over standard DeiT-base ({standard_vit_total/1e6:.1f}M).

Stage 2 inference parameters:
  ViT backbone (all params loaded): {vit_total/1e6:.1f}M
  + ROIProjection + Classifier:     {required_total/1e6:.2f}M
  = Total:                          {(vit_total + required_total)/1e6:.1f}M

  (Note: 4 projection heads [{four_proj/1e6:.2f}M] and text encoder [{bert_enc_total/1e6:.1f}M]
   are not used during Stage 2 inference)

  Frontal-only ViT params (actually used): {standard_vit_total/1e6:.1f}M
  + ROIProjection + Classifier:            {required_total/1e6:.2f}M
  = Active params:                         {(standard_vit_total + required_total)/1e6:.1f}M
""")


# ============================================================================
# RECOMMENDED TABLE FOR PAPER
# ============================================================================
section("RECOMMENDED TABLE FOR PAPER (LaTeX ready)")

print(r"""
% Option A: Report all model parameters (loaded into memory)
\begin{table}[t]
\centering
\caption{Model parameter comparison.}
\label{tab:params}
\begin{tabular}{lccc}
\toprule
Method & Image Enc. & Text Enc. & Total \\
\midrule""")
print(f"BioViL~\\cite{{biovil}}       & 23.8M  & 110M  & 134M  \\\\")
print(f"BioViL-T~\\cite{{biovilt}}    & 25.3M  & 110M  & 135M  \\\\")
print(f"MGCA~\\cite{{mgca}}           & {mgca_img_enc/1e6:.1f}M & {bert_enc_total/1e6:.1f}M & {mgca_total_correct/1e6:.1f}M \\\\")
print(f"MedSigLIP~\\cite{{medsiglip}} & 400M   & 400M  & 800M  \\\\")
print(f"MedST~\\cite{{medst}}         & {img_enc_total/1e6:.1f}M & {bert_enc_total/1e6:.1f}M & {medst_total/1e6:.1f}M \\\\")
print(f"\\textbf{{Ours}}              & {img_enc_total/1e6:.1f}M & {bert_enc_total/1e6:.1f}M & {medst_total/1e6:.1f}M (+{required_total/1e6:.2f}M) \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}
""")

# Also print raw numbers for easy reference
section("RAW NUMBERS (for quick reference)")
print(f"  Standard DeiT-base:      {standard_vit_total:>12,} = {standard_vit_total/1e6:.2f}M")
print(f"  MedST ViT (dual MLP):    {vit_total:>12,} = {vit_total/1e6:.2f}M")
print(f"  Lateral MLP overhead:    {lateral_extra_total:>12,} = {lateral_extra_total/1e6:.2f}M")
print(f"  GlobalEmbedding:         {ge_total:>12,} = {ge_total/1e6:.4f}M")
print(f"  LocalEmbedding:          {le_total:>12,} = {le_total/1e6:.4f}M")
print(f"  4 proj heads:            {four_proj:>12,} = {four_proj/1e6:.2f}M")
print(f"  2 proj heads (text):     {two_proj:>12,} = {two_proj/1e6:.2f}M")
print(f"  BERT-base:               {bert_base_params:>12,} = {bert_base_params/1e6:.2f}M")
print(f"  ImageEncoder (MedST):    {img_enc_total:>12,} = {img_enc_total/1e6:.2f}M")
print(f"  BertEncoder:             {bert_enc_total:>12,} = {bert_enc_total/1e6:.2f}M")
print(f"  MedST pretrain total:    {medst_total:>12,} = {medst_total/1e6:.2f}M")
print(f"  ROIProjection:           {roi_total:>12,} = {roi_total/1e6:.4f}M")
print(f"  AnatomyClassifier:       {cls_total:>12,} = {cls_total/1e6:.4f}M")
print(f"  Required FT layers:      {required_total:>12,} = {required_total/1e6:.4f}M")
print(f"  AttentionROIPool:        {attn_total:>12,} = {attn_total/1e6:.4f}M")
print(f"  AnatomyEmbedding:        {emb_total:>12,} = {emb_total/1e6:.4f}M")
print(f"  CRDA:                    {crda_total:>12,} = {crda_total/1e6:.4f}M")
print(f"  Optional FT layers:      {optional_total:>12,} = {optional_total/1e6:.4f}M")
print(f"  All new FT layers:       {all_new:>12,} = {all_new/1e6:.4f}M")
