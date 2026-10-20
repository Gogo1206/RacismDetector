import torch
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel

from racism_detector.model import MultiTaskClassifier, multitask_loss


def tiny_model(n_aux=3):
    config = RobertaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=40,
    )
    return MultiTaskClassifier(RobertaModel(config, add_pooling_layer=False), 16, n_aux)


def test_forward_shapes():
    model = tiny_model(n_aux=3)
    ids = torch.randint(3, 50, (4, 7))
    main, aux = model(ids, torch.ones_like(ids))
    assert main.shape == (4,)
    assert aux.shape == (4, 3)


def test_zero_aux_weight_is_single_task():
    main, aux = torch.randn(4), torch.randn(4, 3)
    labels, aux_targets = torch.tensor([0.0, 1.0, 1.0, 0.0]), torch.rand(4, 3)
    total, main_loss, _ = multitask_loss(main, aux, labels, aux_targets, aux_weight=0.0)
    assert torch.allclose(total, F.binary_cross_entropy_with_logits(main, labels))
    assert torch.allclose(total, main_loss)


def test_aux_weight_sends_gradient_to_aux_head():
    model = tiny_model()
    ids = torch.randint(3, 50, (2, 5))
    main, aux = model(ids, torch.ones_like(ids))
    total, _, _ = multitask_loss(main, aux, torch.tensor([0.0, 1.0]), torch.rand(2, 3), 1.0)
    total.backward()
    assert model.aux_head.weight.grad.abs().sum() > 0
