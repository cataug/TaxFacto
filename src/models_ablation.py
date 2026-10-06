from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import functional_call

from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# ============================================================
# Helpers
# ============================================================

QV_NAMES = {
    "query",
    "value",
    "query_proj",
    "value_proj",
    "q_proj",
    "v_proj",
}


def _find_qv_suffixes(model):
    """
    Find attention query/value Linear-module suffixes automatically.

    BERT:
        query, value

    DeBERTa-like:
        query_proj, value_proj
    """
    found = set()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        leaf = name.rsplit(".", 1)[-1]

        if leaf in QV_NAMES:
            found.add(leaf)

    if not found:
        raise RuntimeError(
            "Could not identify query/value projection modules "
            "for LoRA."
        )

    print("LoRA target modules:", sorted(found))

    return sorted(found)


def _shared_lora_encoder(
    model_name,
    rank,
    alpha,
    dropout,
    id2label,
    label2id,
):
    config = AutoConfig.from_pretrained(model_name)

    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    base = AutoModel.from_pretrained(
        model_name,
        config=config,
    )

    targets = _find_qv_suffixes(base)

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )

    encoder = get_peft_model(
        base,
        lora_cfg,
    )

    return config, encoder


# ============================================================
# A. Parameter-matched shared nonlinear head
# ============================================================

class SharedLoRAWideHeadClassifier(nn.Module):
    """
    Control model:

        shared LoRA encoder
              |
             CLS
              |
        Linear(768 -> 118)
              |
             GELU
              |
        Linear(118 -> C)

    For BERT-base + r=8 this gives ~386.5k trainable
    parameters, essentially parameter-matched to RoleAdapter.
    """

    def __init__(
        self,
        model_name,
        num_labels,
        id2label,
        label2id,
        rank=8,
        alpha=16,
        dropout=0.05,
        head_width=118,
    ):
        super().__init__()

        config, encoder = _shared_lora_encoder(
            model_name=model_name,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            id2label=id2label,
            label2id=label2id,
        )

        self.encoder = encoder
        self.config = config

        hidden = config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, head_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**model_kwargs)

        h = out.last_hidden_state[:, 0, :]
        h = self.dropout(h)

        logits = self.classifier(h)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# ============================================================
# B. Nonlinear role-conditioned low-rank adapter
# ============================================================

class RoleAdapterClassifier(nn.Module):
    """
    Shared LoRA encoder plus class-specific nonlinear
    low-rank residual adapters:

        z_c = B_c GELU(A_c h)

        logit_c =
            w_c^T [ h + (alpha/r) z_c ] + b_c

    Unlike the earlier purely linear construction, this cannot
    be algebraically collapsed into an ordinary linear head.
    """

    def __init__(
        self,
        model_name,
        num_labels,
        id2label,
        label2id,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()

        config, encoder = _shared_lora_encoder(
            model_name=model_name,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            id2label=id2label,
            label2id=label2id,
        )

        self.encoder = encoder
        self.config = config

        hidden = config.hidden_size

        self.num_labels = num_labels
        self.rank = rank
        self.scale = alpha / rank

        self.dropout = nn.Dropout(dropout)

        # [class, hidden, rank]
        self.A = nn.Parameter(
            torch.empty(
                num_labels,
                hidden,
                rank,
            )
        )

        # [class, rank, hidden]
        self.B = nn.Parameter(
            torch.empty(
                num_labels,
                rank,
                hidden,
            )
        )

        self.classifier_weight = nn.Parameter(
            torch.empty(
                num_labels,
                hidden,
            )
        )

        self.classifier_bias = nn.Parameter(
            torch.zeros(num_labels)
        )

        nn.init.normal_(self.A, std=0.02)

        # Standard LoRA-style initialization:
        # residual starts at exactly zero.
        nn.init.zeros_(self.B)

        nn.init.normal_(
            self.classifier_weight,
            std=0.02,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**model_kwargs)

        h = out.last_hidden_state[:, 0, :]
        h = self.dropout(h)

        # [batch, class, rank]
        low = torch.einsum(
            "bh,chr->bcr",
            h,
            self.A,
        )

        # Crucial difference from the old linear version.
        low = F.gelu(low)

        # [batch, class, hidden]
        residual = torch.einsum(
            "bcr,crh->bch",
            low,
            self.B,
        )

        class_h = (
            h.unsqueeze(1)
            + self.scale * residual
        )

        logits = torch.einsum(
            "bch,ch->bc",
            class_h,
            self.classifier_weight,
        )

        logits = (
            logits
            + self.classifier_bias
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# ============================================================
# C. Genuine role-specific LoRA inside the encoder
# ============================================================

class TrueCategoryLoRAClassifier(nn.Module):
    """
    Genuine role-specific LoRA.

    There is ONE frozen backbone in memory.

    For each class c we create separate LoRA deltas for every
    attention Q/V projection:

        W_q,c = W_q + scale * B_q,c A_q,c
        W_v,c = W_v + scale * B_v,c A_v,c

    The encoder is evaluated once per role using torch.func's
    functional_call, so the frozen base weights are NOT copied
    seven times.

    This is deliberately a stronger/slower ablation:
    seven labels -> seven encoder forwards.

    Its purpose is to answer whether true role-specific
    adaptation inside the transformer provides benefits beyond
    a role-conditioned output adapter.
    """

    def __init__(
        self,
        model_name,
        num_labels,
        id2label,
        label2id,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)

        config.num_labels = num_labels
        config.id2label = id2label
        config.label2id = label2id

        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
        )

        # Frozen shared backbone.
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.config = config
        self.num_labels = num_labels
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

        targets = []

        for module_name, module in self.encoder.named_modules():

            if not isinstance(module, nn.Linear):
                continue

            leaf = module_name.rsplit(".", 1)[-1]

            if leaf not in QV_NAMES:
                continue

            param_name = module_name + ".weight"

            targets.append(
                (
                    param_name,
                    module.in_features,
                    module.out_features,
                )
            )

        if not targets:
            raise RuntimeError(
                "No attention Q/V Linear modules were found."
            )

        self.target_specs = targets

        print(
            "True category LoRA targets:",
            len(targets),
        )

        # One A/B tensor per transformer target.
        #
        # Each tensor additionally carries a class dimension,
        # i.e. every role owns its own LoRA delta.
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()

        for _, in_features, out_features in targets:

            A = nn.Parameter(
                torch.empty(
                    num_labels,
                    rank,
                    in_features,
                )
            )

            B = nn.Parameter(
                torch.empty(
                    num_labels,
                    out_features,
                    rank,
                )
            )

            nn.init.normal_(A, std=0.02)
            nn.init.zeros_(B)

            self.lora_A.append(A)
            self.lora_B.append(B)

        hidden = config.hidden_size

        self.classifier_weight = nn.Parameter(
            torch.empty(
                num_labels,
                hidden,
            )
        )

        self.classifier_bias = nn.Parameter(
            torch.zeros(num_labels)
        )

        nn.init.normal_(
            self.classifier_weight,
            std=0.02,
        )

    def _role_forward(
        self,
        role,
        model_kwargs,
    ):
        # References to frozen base parameters/buffers.
        params = dict(
            self.encoder.named_parameters()
        )

        buffers = dict(
            self.encoder.named_buffers()
        )

        # Start with frozen model state.
        state = {}

        state.update(params)
        state.update(buffers)

        # Override Q/V weights with role-specific LoRA deltas.
        for i, (param_name, _, _) in enumerate(
            self.target_specs
        ):
            base_weight = params[param_name]

            delta = (
                self.lora_B[i][role]
                @ self.lora_A[i][role]
            )

            state[param_name] = (
                base_weight
                + self.scale * delta
            )

        out = functional_call(
            self.encoder,
            state,
            kwargs=model_kwargs,
        )

        h = out.last_hidden_state[:, 0, :]

        return self.dropout(h)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        logits = []

        for role in range(self.num_labels):

            h = self._role_forward(
                role,
                model_kwargs,
            )

            role_logit = (
                h
                * self.classifier_weight[role]
            ).sum(dim=-1)

            role_logit = (
                role_logit
                + self.classifier_bias[role]
            )

            logits.append(role_logit)

        logits = torch.stack(
            logits,
            dim=1,
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
