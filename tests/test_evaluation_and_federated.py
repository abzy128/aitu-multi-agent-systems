from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset

from fl_saf.data.loader import ClientData
from fl_saf.data.preprocessing import PreprocessingReport
from fl_saf.evaluation.comms import model_bytes, state_dict_bytes
from fl_saf.evaluation.metrics import evaluate_client
from fl_saf.training.fedavg import weighted_average
from fl_saf.training.fedbn import _non_bn_state


class OffsetModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1, 0] + 1.0


class TinyBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.head = nn.Linear(2, 1)


def test_evaluate_client_inverse_transforms_predictions_and_targets() -> None:
    x = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    report = PreprocessingReport(
        client_id="client_1",
        track="univariate",
        feature_names=["active_power"],
        target_mean=10.0,
        target_scale=2.0,
    )
    client = ClientData(
        client_id="client_1",
        feature_names=["active_power"],
        train=TensorDataset(x, y),
        val=TensorDataset(x, y),
        test=TensorDataset(x, y),
        report=report,
    )

    metrics, y_true, y_pred = evaluate_client(
        OffsetModel(), client, "test", batch_size=2, device=torch.device("cpu")
    )

    assert y_true.tolist() == [[10.0], [12.0]]
    assert y_pred.tolist() == [[12.0], [14.0]]
    assert metrics["rmse"] == 2.0
    assert metrics["mae"] == 2.0


def test_model_bytes_can_exclude_batchnorm_state() -> None:
    model = TinyBNModel()
    full_bytes = model_bytes(model)
    non_bn_bytes = model_bytes(model, exclude_bn=True)
    expected_non_bn = state_dict_bytes(
        {k: v for k, v in model.state_dict().items() if not k.startswith("bn.")}
    )

    assert non_bn_bytes == expected_non_bn
    assert 0 < non_bn_bytes < full_bytes


def test_weighted_average_excludes_batchnorm_parameters_when_requested() -> None:
    state_a = {
        "bn.weight": torch.tensor([1.0, 1.0]),
        "head.weight": torch.tensor([[2.0, 4.0]]),
    }
    state_b = {
        "bn.weight": torch.tensor([9.0, 9.0]),
        "head.weight": torch.tensor([[6.0, 8.0]]),
    }

    averaged = weighted_average([state_a, state_b], weights=[1, 3], exclude_bn=True)

    assert averaged["bn.weight"].tolist() == [1.0, 1.0]
    assert torch.allclose(averaged["head.weight"], torch.tensor([[5.0, 7.0]]))


def test_fedbn_non_bn_state_filters_all_batchnorm_entries() -> None:
    model = TinyBNModel()
    filtered = _non_bn_state(model.state_dict())

    assert filtered
    assert all(not key.startswith("bn.") for key in filtered)
    assert set(filtered) == {"head.weight", "head.bias"}
