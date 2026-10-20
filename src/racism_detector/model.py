import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class MultiTaskClassifier(nn.Module):
    """Shared encoder feeding two heads.

    main_head: is the comment racist (1 logit).
    aux_head:  annotator facet scores + share saying it targets race (n_aux logits).
    The aux head only gets supervision when aux_weight > 0, so aux_weight=0 is the
    single-task baseline with the same architecture.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, n_aux: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.main_head = nn.Linear(hidden_size, 1)
        self.aux_head = nn.Linear(hidden_size, n_aux)

    @classmethod
    def from_pretrained(cls, name: str, n_aux: int) -> "MultiTaskClassifier":
        encoder = AutoModel.from_pretrained(name, add_pooling_layer=False)
        return cls(encoder, encoder.config.hidden_size, n_aux)

    def forward(self, input_ids, attention_mask) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(hidden[:, 0])  # <s> token
        return self.main_head(pooled).squeeze(-1), self.aux_head(pooled)


def multitask_loss(main_logits, aux_logits, labels, aux_targets, aux_weight: float):
    """Returns (total, main, aux). Aux targets are soft labels in [0, 1], so BCE fits."""
    main = F.binary_cross_entropy_with_logits(main_logits, labels)
    aux = F.binary_cross_entropy_with_logits(aux_logits, aux_targets)
    return main + aux_weight * aux, main, aux
