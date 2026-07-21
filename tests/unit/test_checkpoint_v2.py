from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_core.artifacts import (
    ArtifactCompatibilityError,
    TrainingRunIdentity,
    sha256_json,
)
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig
import model_core.engine as engine_module
import training_service
from tests.unit.test_training_service_v2 import OneManager
from tests.unit.test_artifacts import artifact_identity


def run_identity(seed: int = 42) -> TrainingRunIdentity:
    return TrainingRunIdentity(run_id="1" * 32, artifact_identity=artifact_identity(seed=seed))


def engine(identity: TrainingRunIdentity | None = None) -> AlphaEngine:
    return AlphaEngine(data_manager=None, use_lord_regularization=False,
                       target_symbol="EURUSD", run_identity=identity)


def test_formal_checkpoint_operations_require_explicit_identity(tmp_path) -> None:
    empty = engine()
    with pytest.raises(ArtifactCompatibilityError, match="run_identity.*expected=.*actual="):
        empty.save_checkpoint(0, str(tmp_path / "legacy.pt"))
    torch.save({"step": 0, "vocab_version": "looks-current"}, tmp_path / "old.pt")
    old_bytes = (tmp_path / "old.pt").read_bytes()
    before = copy.deepcopy(empty.model.state_dict())
    with pytest.raises(ArtifactCompatibilityError, match="run_identity.*expected=.*actual="):
        empty.load_checkpoint(str(tmp_path / "old.pt"))
    assert all(torch.equal(before[k], v) for k, v in empty.model.state_dict().items())
    assert (tmp_path / "old.pt").read_bytes() == old_bytes


def test_checkpoint_v2_round_trip_restores_complete_state_and_rng(tmp_path) -> None:
    identity = run_identity()
    source = engine(identity)
    source.best_score = 1.25
    source.best_formula = [1, 2]
    source._best_snapshot = copy.deepcopy(source.model.state_dict())
    source.factor_pool = [(0.4, 7, torch.tensor([1.0]))]
    source._factor_pool_counter = 8
    source._elite_pool = [(0.7, 3, [1, 2], 4)]
    source._elite_counter = 5
    source._restart_count = 2
    source._best_update_step = 6
    source._stagnation_steps = 3
    source._reward_ema = 0.125
    source._reward_ema_step = 9
    source._low_entropy_streak = 4
    source._previous_initial_distribution = torch.tensor([0.2, 0.8])
    source.training_history["step"] = [1, 2]
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    path = tmp_path / identity.checkpoint_filename(9)
    source.save_checkpoint(9, str(path))
    expected = (random.random(), np.random.random(), torch.rand(3))

    restored = engine(identity)
    assert restored.load_checkpoint(str(path)) == 10
    assert restored.best_score == source.best_score
    assert restored.best_formula == source.best_formula
    assert restored._factor_pool_counter == 8
    assert restored._elite_counter == 5
    assert restored._restart_count == 2
    assert restored._best_update_step == 6
    assert restored._stagnation_steps == 3
    assert restored._reward_ema == 0.125
    assert restored._reward_ema_step == 9
    assert restored._low_entropy_streak == 4
    assert torch.equal(restored._previous_initial_distribution, torch.tensor([0.2, 0.8]))
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


@pytest.mark.parametrize("configured_device", ["cpu", "cuda"])
def test_checkpoint_deserialization_keeps_cpu_rng_on_cpu_for_configured_device(
    monkeypatch, tmp_path, configured_device
) -> None:
    identity = run_identity()
    source = engine(identity)
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.fill_(0.125)
    source.opt.zero_grad()
    sum(parameter.square().sum() for parameter in source.model.parameters()).backward()
    source.opt.step()
    expected_optimizer = _clone_nested(source.opt.state_dict())
    torch.manual_seed(12701)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(2, str(path))
    expected_next = torch.rand(4)
    target = engine(identity)
    with torch.no_grad():
        for parameter in target.model.parameters():
            parameter.zero_()
    real_load = torch.load
    real_tensor_to = torch.Tensor.to
    requested_locations = []
    optimizer_destinations = []

    def cpu_host_cuda_contract(path_arg, *, map_location, weights_only):
        requested_locations.append(str(map_location))
        payload = real_load(
            path_arg, map_location="cpu", weights_only=weights_only
        )
        if str(map_location).startswith("cuda"):
            payload["torch_cpu_rng_state"] = payload[
                "torch_cpu_rng_state"
            ].to("meta")
        return payload

    def record_tensor_destination(tensor, *args, **kwargs):
        destination = kwargs.get("device", args[0] if args else None)
        if destination is not None:
            rendered = str(destination)
            optimizer_destinations.append(rendered)
            if rendered.startswith("cuda"):
                return tensor
        return real_tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(ModelConfig, "DEVICE", torch.device(configured_device))
    monkeypatch.setattr(torch, "load", cpu_host_cuda_contract)
    monkeypatch.setattr(torch.Tensor, "to", record_tensor_destination)

    assert target.load_checkpoint(str(path)) == 3
    assert requested_locations == ["cpu"]
    assert all(
        torch.equal(target.model.state_dict()[name], value)
        for name, value in source.model.state_dict().items()
    )
    assert target.opt.state
    _assert_nested_exact(target.opt.state_dict(), expected_optimizer)
    assert optimizer_destinations
    assert all(not destination.startswith("cuda") for destination in optimizer_destinations)
    for parameter, state in target.opt.state.items():
        for value in state.values():
            if type(value) is torch.Tensor:
                assert value.device == parameter.device
    assert torch.equal(torch.rand(4), expected_next)


def test_checkpoint_install_rejects_non_cpu_cpu_rng_before_mutation(
    tmp_path
) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(2, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["torch_cpu_rng_state"] = payload["torch_cpu_rng_state"].to("meta")
    target = engine(identity)
    before = _complete_engine_snapshot(target)

    with pytest.raises(
        ArtifactCompatibilityError,
        match="torch_cpu_rng_state.*one-dimensional CPU uint8 Tensor",
    ):
        target._install_checkpoint_v2_atomically(payload, str(path), identity)
    _assert_complete_snapshot(target, before)


def test_checkpoint_identity_mismatch_is_rejected_before_mutation(tmp_path) -> None:
    source = engine(run_identity(42))
    path = tmp_path / "ckpt_v2_EURUSD_H1_fake_step_0.pt"
    source.save_checkpoint(0, str(path))
    target = engine(run_identity(43))
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(ArtifactCompatibilityError, match="training_config.*expected=.*actual="):
        target.load_checkpoint(str(path))
    assert all(torch.equal(before[k], v) for k, v in target.model.state_dict().items())


def _service_run_identity() -> TrainingRunIdentity:
    return TrainingRunIdentity(
        run_id="e" * 32,
        artifact_identity=training_service._artifact_identity(OneManager(), 42),
    )


def test_checkpoint_missing_lord_identity_is_rejected_without_rewriting_bytes(
    tmp_path,
) -> None:
    identity = _service_run_identity()
    source = AlphaEngine(
        None,
        use_lord_regularization=getattr(ModelConfig, "USE_LORD_REGULARIZATION", True),
        lord_decay_rate=getattr(ModelConfig, "LORD_DECAY_RATE", 1.0e-3),
        lord_num_iterations=getattr(ModelConfig, "LORD_NUM_ITERATIONS", 5),
        target_symbol="EURUSD",
        run_identity=identity,
    )
    path = tmp_path / "old-missing-lord.pt"
    source.save_checkpoint(0, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["run_identity"]["artifact_identity"]["training_config"]
    config.pop("lord", None)
    payload["run_identity"]["artifact_identity"]["training_config_hash"] = sha256_json(config)
    torch.save(payload, path)
    old_bytes = path.read_bytes()
    target = AlphaEngine(
        None,
        use_lord_regularization=getattr(ModelConfig, "USE_LORD_REGULARIZATION", True),
        lord_decay_rate=getattr(ModelConfig, "LORD_DECAY_RATE", 1.0e-3),
        lord_num_iterations=getattr(ModelConfig, "LORD_NUM_ITERATIONS", 5),
        target_symbol="EURUSD",
        run_identity=identity,
    )
    before = copy.deepcopy(target.model.state_dict())

    with pytest.raises(ArtifactCompatibilityError, match=r"lord|missing"):
        target.load_checkpoint(str(path))

    assert path.read_bytes() == old_bytes
    assert all(torch.equal(before[name], value) for name, value in target.model.state_dict().items())


@pytest.mark.parametrize(
    ("field", "mutant"),
    [("decay_rate", 0.5), ("num_iterations", 1)],
)
def test_checkpoint_rejects_divergent_effective_lord_optimizer_before_write(
    monkeypatch, tmp_path, field: str, mutant: object
) -> None:
    identity = _service_run_identity()
    current = AlphaEngine(
        None,
        use_lord_regularization=True,
        lord_decay_rate=1.0e-3,
        lord_num_iterations=5,
        target_symbol="EURUSD",
        run_identity=identity,
    )
    path = tmp_path / "owned-checkpoint.pt"
    current.save_checkpoint(0, str(path))
    old_bytes = path.read_bytes()
    before = copy.deepcopy(current.model.state_dict())
    real_state_dict = current.model.state_dict

    def diverge_before_serialization(*args, **kwargs):
        setattr(current.lord_opt, field, mutant)
        return real_state_dict(*args, **kwargs)

    monkeypatch.setattr(current.model, "state_dict", diverge_before_serialization)

    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"training_config\.lord\.lord_{field}",
    ):
        current.save_checkpoint(1, str(path))

    expected = current.lord_decay_rate if field == "decay_rate" else current.lord_num_iterations
    assert getattr(current.lord_opt, field) == expected
    assert path.read_bytes() == old_bytes
    for name, value in before.items():
        assert torch.equal(real_state_dict()[name], value)


def test_identical_lord_runtime_controls_save_and_load_exactly(tmp_path) -> None:
    identity = _service_run_identity()
    source = AlphaEngine(
        None,
        use_lord_regularization=True,
        lord_decay_rate=1.0e-3,
        lord_num_iterations=5,
        target_symbol="EURUSD",
        run_identity=identity,
    )
    path = tmp_path / "matching-lord.pt"
    source.save_checkpoint(0, str(path))
    target = AlphaEngine(
        None,
        use_lord_regularization=True,
        lord_decay_rate=1.0e-3,
        lord_num_iterations=5,
        target_symbol="EURUSD",
        run_identity=identity,
    )

    assert target.load_checkpoint(str(path)) == 1
    for name, value in source.model.state_dict().items():
        assert torch.equal(target.model.state_dict()[name], value)
    assert target.lord_opt.decay_rate == source.lord_opt.decay_rate == 1.0e-3
    assert target.lord_opt.num_iterations == source.lord_opt.num_iterations == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_lord_regularization", False),
        ("lord_decay_rate", 0.5),
        ("lord_num_iterations", 1),
    ],
)
def test_checkpoint_lord_mismatch_is_rejected_before_state_installation(
    tmp_path, field: str, value: object
) -> None:
    identity = _service_run_identity()
    source = AlphaEngine(
        None,
        use_lord_regularization=getattr(ModelConfig, "USE_LORD_REGULARIZATION", True),
        lord_decay_rate=getattr(ModelConfig, "LORD_DECAY_RATE", 1.0e-3),
        lord_num_iterations=getattr(ModelConfig, "LORD_NUM_ITERATIONS", 5),
        target_symbol="EURUSD",
        run_identity=identity,
    )
    path = tmp_path / f"mismatch-{field}.pt"
    source.save_checkpoint(0, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    artifact = payload["run_identity"]["artifact_identity"]
    lord = artifact["training_config"].setdefault(
        "lord",
        {
            "use_lord_regularization": True,
            "lord_decay_rate": 1.0e-3,
            "lord_num_iterations": 5,
        },
    )
    lord[field] = value
    artifact["training_config_hash"] = sha256_json(artifact["training_config"])
    torch.save(payload, path)
    target = AlphaEngine(
        None,
        use_lord_regularization=getattr(ModelConfig, "USE_LORD_REGULARIZATION", True),
        lord_decay_rate=getattr(ModelConfig, "LORD_DECAY_RATE", 1.0e-3),
        lord_num_iterations=getattr(ModelConfig, "LORD_NUM_ITERATIONS", 5),
        target_symbol="EURUSD",
        run_identity=identity,
    )
    before = copy.deepcopy(target.model.state_dict())

    with pytest.raises(ArtifactCompatibilityError, match=field):
        target.load_checkpoint(str(path))

    assert all(torch.equal(before[name], value) for name, value in target.model.state_dict().items())


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("symbol", lambda value: (
            value.__setitem__("symbol", "GBPUSD"),
            value["training_dataset"].__setitem__("symbol", "GBPUSD"),
        )),
        ("timeframe", lambda value: (
            value.__setitem__("timeframe", "D1"),
            value["training_dataset"].__setitem__("timeframe", "D1"),
            value["training_config"]["timeframe_reward"].update(
                {"timeframe": "D1", "target_bars_per_trade": 0.5}
            ),
            value.__setitem__(
                "training_config_hash", sha256_json(value["training_config"])
            ),
        )),
        ("data_fingerprint", lambda value: value["training_dataset"].__setitem__(
            "data_fingerprint", "c" * 64
        )),
        ("training_config_hash", lambda value: (
            value["training_config"].__setitem__("random_seed", 43),
            value.__setitem__(
                "training_config_hash", sha256_json(value["training_config"])
            ),
        )),
        ("vocab_version", lambda value: value.__setitem__("vocab_version", "legacy-v1")),
        ("core_semantics_version", lambda value: value.__setitem__(
            "core_semantics_version", "1"
        )),
    ],
)
def test_every_checkpoint_identity_axis_fails_before_mutation(
    tmp_path, field, mutate
) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(0, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload["run_identity"]["artifact_identity"])
    torch.save(payload, path)
    target = engine(identity)
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(ArtifactCompatibilityError) as caught:
        target.load_checkpoint(str(path))
    message = str(caught.value)
    assert field in message
    assert "expected=" in message and "actual=" in message
    assert all(torch.equal(before[key], value)
               for key, value in target.model.state_dict().items())


class _DataAccessSpy:
    def __init__(self):
        self.reads = 0

    def __getattribute__(self, name):
        if name not in {"reads", "__dict__", "__class__"}:
            object.__setattr__(self, "reads", object.__getattribute__(self, "reads") + 1)
            raise AssertionError(f"data accessed before identity validation: {name}")
        return object.__getattribute__(self, name)


class _HostileIdentity:
    def __getattribute__(self, name):
        if name == "__class__":
            return object.__getattribute__(self, name)
        raise AssertionError("hostile identity protocol invoked")

    def __repr__(self):
        raise AssertionError("hostile identity repr invoked")


def _mutate_lifetime_identity(current, original, mutation) -> None:
    if mutation == "deleted":
        del current.run_identity
    elif mutation == "none":
        current.run_identity = None
    elif mutation == "different":
        current.run_identity = run_identity(43)
    elif mutation == "hostile":
        current.run_identity = _HostileIdentity()
    else:
        replacement = TrainingRunIdentity.from_dict(original.to_dict())
        assert replacement == original and replacement is not original
        current.run_identity = replacement


@pytest.mark.parametrize("validation_callback", ["to-dict", "from-dict"])
@pytest.mark.parametrize(
    "mutation", ["deleted", "none", "different", "hostile", "equal-distinct"]
)
def test_identity_validation_callback_rejects_before_checkpoint_io(
    monkeypatch, tmp_path, validation_callback, mutation
) -> None:
    identity = run_identity()
    checkpoint = tmp_path / "existing.pt"
    checkpoint.write_bytes(b"EXISTING-CHECKPOINT-BYTES")
    checkpoint_bytes = checkpoint.read_bytes()
    current = engine(identity)
    before = _complete_engine_snapshot(current)
    real_to_dict = TrainingRunIdentity.to_dict
    real_from_dict = TrainingRunIdentity.from_dict
    mutated = False
    calls = []

    def mutate_once() -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        if mutation == "equal-distinct":
            current.run_identity = TrainingRunIdentity(
                run_id=identity.run_id,
                artifact_identity=identity.artifact_identity,
            )
        else:
            _mutate_lifetime_identity(current, identity, mutation)

    if validation_callback == "to-dict":
        def mutating_to_dict(value):
            payload = real_to_dict(value)
            mutate_once()
            return payload

        monkeypatch.setattr(TrainingRunIdentity, "to_dict", mutating_to_dict)

        def later_from_dict(cls, payload):
            calls.append("from_dict")
            return real_from_dict(payload)

        monkeypatch.setattr(
            TrainingRunIdentity, "from_dict", classmethod(later_from_dict)
        )
    else:
        def mutating_from_dict(cls, payload):
            result = real_from_dict(payload)
            mutate_once()
            return result

        monkeypatch.setattr(
            TrainingRunIdentity, "from_dict", classmethod(mutating_from_dict)
        )

    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: (
            calls.append("torch.load"),
            (_ for _ in ()).throw(AssertionError("checkpoint I/O reached")),
        )[1],
    )
    with pytest.raises(ArtifactCompatibilityError) as caught:
        current.load_checkpoint(str(checkpoint))

    assert calls == []
    assert "run_identity" in str(caught.value) and len(str(caught.value)) <= 512
    assert current.run_identity is identity
    _assert_complete_snapshot(current, before)
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "boundary", ["payload-identity", "checkpoint-filename", "load-identity"]
)
@pytest.mark.parametrize(
    "mutation", ["deleted", "none", "different", "hostile", "equal-distinct"]
)
def test_checkpoint_identity_callback_rejects_before_next_persistence_boundary(
    monkeypatch, tmp_path, boundary, mutation
) -> None:
    identity = run_identity()
    source = engine(identity)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(2, str(checkpoint))
    checkpoint_bytes = checkpoint.read_bytes()
    current = engine(identity)
    before = _complete_engine_snapshot(current)
    calls = []

    def mutate() -> None:
        if mutation == "equal-distinct":
            current.run_identity = TrainingRunIdentity(
                run_id=identity.run_id,
                artifact_identity=identity.artifact_identity,
            )
        else:
            _mutate_lifetime_identity(current, identity, mutation)

    if boundary == "payload-identity":
        real = TrainingRunIdentity.to_dict
        count = 0

        def callback(value):
            nonlocal count
            result = real(value)
            count += 1
            if count == 2:
                mutate()
            return result

        monkeypatch.setattr(TrainingRunIdentity, "to_dict", callback)
        real_save = torch.save

        def save_callback(*args, **kwargs):
            calls.append("torch.save")
            return real_save(*args, **kwargs)

        monkeypatch.setattr(torch, "save", save_callback)
        target = tmp_path / "payload.pt"
        action = lambda: current.save_checkpoint(3, str(target))
    elif boundary == "checkpoint-filename":
        real = TrainingRunIdentity.checkpoint_filename

        def callback(value, step):
            result = real(value, step)
            mutate()
            return result

        monkeypatch.setattr(TrainingRunIdentity, "checkpoint_filename", callback)
        monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", tmp_path / "checkpoints")
        real_save = torch.save

        def save_callback(*args, **kwargs):
            calls.append("torch.save")
            return real_save(*args, **kwargs)

        monkeypatch.setattr(torch, "save", save_callback)
        target = tmp_path / "checkpoints" / "untrusted.pt"
        action = lambda: current.save_checkpoint(3)
    else:
        real = TrainingRunIdentity.from_dict
        count = 0

        def callback(cls, payload):
            nonlocal count
            result = real(payload)
            count += 1
            if count == 2:
                mutate()
            return result

        monkeypatch.setattr(
            TrainingRunIdentity, "from_dict", classmethod(callback)
        )

        def install_callback(*_args, **_kwargs):
            calls.append("install")
            return 3

        monkeypatch.setattr(
            current, "_install_checkpoint_v2_atomically", install_callback
        )
        target = tmp_path / "unused.pt"
        action = lambda: current.load_checkpoint(str(checkpoint))

    with pytest.raises(ArtifactCompatibilityError) as caught:
        action()

    assert calls == []
    assert "run_identity" in str(caught.value) and len(str(caught.value)) <= 512
    assert current.run_identity is identity
    _assert_complete_snapshot(current, before)
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert not target.exists()
    assert not list(tmp_path.rglob(".*.tmp"))


@pytest.mark.parametrize("boundary", ["model-state", "torch-load"])
@pytest.mark.parametrize(
    "mutation", ["deleted", "none", "different", "hostile", "equal-distinct"]
)
def test_checkpoint_revalidates_exact_identity_after_external_state_boundary(
    monkeypatch, tmp_path, boundary, mutation
) -> None:
    identity = run_identity()
    source = engine(identity)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(2, str(checkpoint))
    checkpoint_bytes = checkpoint.read_bytes()
    current = engine(identity)
    before = _complete_engine_snapshot(current)
    target = tmp_path / "new.pt"
    if boundary == "model-state":
        real_state_dict = current.model.state_dict
        mutated = False

        def mutating_state_dict(*args, **kwargs):
            nonlocal mutated
            result = real_state_dict(*args, **kwargs)
            if not mutated:
                mutated = True
                _mutate_lifetime_identity(current, identity, mutation)
            return result

        monkeypatch.setattr(current.model, "state_dict", mutating_state_dict)
        real_optimizer_state_dict = current.opt.state_dict
        monkeypatch.setattr(
            current.opt,
            "state_dict",
            lambda: (
                real_optimizer_state_dict()
                if object.__getattribute__(current, "__dict__").get(
                    "run_identity"
                ) is identity
                else (_ for _ in ()).throw(
                    AssertionError("optimizer callback crossed identity boundary")
                )
            ),
        )
        action = lambda: current.save_checkpoint(3, str(target))
    else:
        real_load = torch.load

        def mutating_load(*args, **kwargs):
            result = real_load(*args, **kwargs)
            _mutate_lifetime_identity(current, identity, mutation)
            return result

        monkeypatch.setattr(torch, "load", mutating_load)
        action = lambda: current.load_checkpoint(str(checkpoint))
    caught = None
    try:
        action()
    except ArtifactCompatibilityError as exc:
        caught = exc
    assert (type(caught), target.exists()) == (ArtifactCompatibilityError, False)
    assert "run_identity" in str(caught) and len(str(caught)) <= 512
    assert current.run_identity is identity
    _assert_complete_snapshot(current, before)
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "value_factory",
    [lambda: None, object, _HostileIdentity],
    ids=["none", "object", "hostile"],
)
@pytest.mark.parametrize("method", ["train", "save", "load"])
def test_formal_methods_validate_exact_identity_at_entry(
    monkeypatch, tmp_path, value_factory, method
) -> None:
    spy = _DataAccessSpy()
    current = AlphaEngine(
        data_manager=spy,
        use_lord_regularization=False,
        target_symbol="EURUSD",
        run_identity=run_identity(),
    )
    current.run_identity = value_factory()
    target = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(
        torch, "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("file read before identity validation")
        ),
    )
    action = {
        "train": lambda: current.train(end_step=0),
        "save": lambda: current.save_checkpoint(0, str(target)),
        "load": lambda: current.load_checkpoint(str(target)),
    }[method]
    with pytest.raises(ArtifactCompatibilityError) as caught:
        action()
    message = str(caught.value)
    assert "run_identity" in message
    assert "expected=" in message and "actual=" in message
    assert len(message) <= 1024
    assert spy.reads == 0
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("method", ["train", "save", "load"])
def test_deleted_identity_is_rejected_before_any_external_access(
    monkeypatch, tmp_path, method
) -> None:
    spy = _DataAccessSpy()
    current = AlphaEngine(
        data_manager=spy,
        use_lord_regularization=False,
        target_symbol="EURUSD",
        run_identity=run_identity(),
    )
    del current.run_identity
    target = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(
        torch, "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("file read before identity validation")
        ),
    )
    action = {
        "train": lambda: current.train(end_step=0),
        "save": lambda: current.save_checkpoint(0, str(target)),
        "load": lambda: current.load_checkpoint(str(target)),
    }[method]
    with pytest.raises(ArtifactCompatibilityError, match="run_identity.*expected=.*actual="):
        action()
    assert spy.reads == 0
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def _clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_exact(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_exact(a, b)
    else:
        assert left == right


def _complete_engine_snapshot(current: AlphaEngine):
    return {
        "model": _clone_nested(current.model.state_dict()),
        "optimizer": _clone_nested(current.opt.state_dict()),
        "fields": _clone_nested({
            name: getattr(current, name)
            for name in (
                "best_score", "best_formula", "best_metrics", "_best_snapshot",
                "factor_pool", "factor_pool_scores", "_factor_pool_counter",
                "_elite_pool", "elite_pool_ages", "_elite_counter",
                "_restart_count", "_best_update_step", "_stagnation_steps",
                "_reward_ema", "_reward_ema_step", "_low_entropy_streak",
                "_previous_initial_distribution", "training_history",
            )
        }),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        "rank": _clone_nested(current.rank_monitor.history)
        if current.rank_monitor is not None else None,
    }


def _assert_complete_snapshot(current: AlphaEngine, expected) -> None:
    _assert_nested_exact(current.model.state_dict(), expected["model"])
    _assert_nested_exact(current.opt.state_dict(), expected["optimizer"])
    _assert_nested_exact(
        {name: getattr(current, name) for name in expected["fields"]},
        expected["fields"],
    )
    _assert_nested_exact(random.getstate(), expected["python"])
    _assert_nested_exact(np.random.get_state(), expected["numpy"])
    assert torch.equal(torch.get_rng_state(), expected["torch"])
    actual_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    _assert_nested_exact(actual_cuda, expected["cuda"])
    actual_rank = current.rank_monitor.history if current.rank_monitor is not None else None
    _assert_nested_exact(actual_rank, expected["rank"])


@pytest.mark.parametrize(
    ("field", "corrupt"),
    [
        ("model_state_dict", lambda payload: payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))),
        ("optimizer_state_dict", lambda payload: payload.__setitem__("optimizer_state_dict", {"state": {}, "param_groups": "bad"})),
        ("factor_pool", lambda payload: payload.__setitem__("factor_pool", object())),
        ("restart_count", lambda payload: payload.__setitem__("restart_count", "bad")),
        ("training_history", lambda payload: payload.__setitem__("training_history", 7)),
        ("python_random_state", lambda payload: payload.__setitem__("python_random_state", ("bad",))),
        ("numpy_random_state", lambda payload: payload.__setitem__("numpy_random_state", ("bad",))),
        ("torch_cpu_rng_state", lambda payload: payload.__setitem__("torch_cpu_rng_state", torch.tensor([1], dtype=torch.int64))),
    ],
)
def test_corrupt_checkpoint_category_is_failure_atomic(tmp_path, field, corrupt) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(3, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrupt(payload)
    torch.save(payload, path)
    target = engine(identity)
    random.seed(901); np.random.seed(902); torch.manual_seed(903)
    before = _complete_engine_snapshot(target)
    with pytest.raises(ArtifactCompatibilityError) as caught:
        target.load_checkpoint(str(path))
    message = str(caught.value)
    assert field in message
    assert len(message) <= 1024
    _assert_complete_snapshot(target, before)


@pytest.mark.parametrize("component", ["model", "optimizer"])
def test_restore_exception_rolls_back_complete_engine(
    monkeypatch, tmp_path, component
) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(2, str(path))
    target = engine(identity)
    random.seed(911); np.random.seed(912); torch.manual_seed(913)
    before = _complete_engine_snapshot(target)
    failure = (
        RuntimeError("model restore failed")
        if component == "model" else ValueError("optimizer restore failed")
    )
    if component == "model":
        def fail_model(state):
            with torch.no_grad():
                next(target.model.parameters()).add_(1)
            raise failure
        monkeypatch.setattr(target.model, "load_state_dict", fail_model)
    else:
        def fail_optimizer(state):
            target.opt.param_groups[0]["lr"] = 99.0
            raise failure
        monkeypatch.setattr(target.opt, "load_state_dict", fail_optimizer)
    with pytest.raises(
        ArtifactCompatibilityError, match=f"{component}.*expected=.*actual="
    ) as caught:
        target.load_checkpoint(str(path))
    assert caught.value.__cause__ is failure
    _assert_complete_snapshot(target, before)


class _CheckpointRestoreAbort(BaseException):
    pass


@pytest.mark.parametrize("component", ["model", "optimizer"])
@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: KeyboardInterrupt("checkpoint restore interrupted"),
        lambda: SystemExit(73),
        lambda: GeneratorExit("checkpoint generator closed"),
        lambda: _CheckpointRestoreAbort("checkpoint restore aborted"),
    ],
    ids=["keyboard-interrupt", "system-exit", "generator-exit", "custom-base"],
)
def test_restore_baseexception_rolls_back_everything_and_reraises_exact_object(
    monkeypatch, tmp_path, component, failure_factory
) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(2, str(path))
    old_bytes = path.read_bytes()
    target = engine(identity)
    target.rank_monitor = SimpleNamespace(history=["before-rank"])
    random.seed(921); np.random.seed(922); torch.manual_seed(923)
    before = _complete_engine_snapshot(target)
    failure = failure_factory()
    expected_type = type(failure)
    expected_args = failure.args

    def mutate_everything_then_raise(_state):
        with torch.no_grad():
            next(target.model.parameters()).add_(11)
        target.opt.param_groups[0]["lr"] = 88.0
        target.best_score = 77.0
        target.best_formula = [99]
        target.best_metrics = {"mutated": True}
        target._best_snapshot = None
        target.factor_pool = [(77.0, 1, torch.ones(1))]
        target.factor_pool_scores = [77.0]
        target._factor_pool_counter = 77
        target._elite_pool = [(77.0, 1, [99], 1)]
        target.elite_pool_ages = [77]
        target._elite_counter = 77
        target._restart_count = 77
        target._best_update_step = 77
        target._stagnation_steps = 77
        target._reward_ema = 77.0
        target._reward_ema_step = 77
        target._low_entropy_streak = 77
        target._previous_initial_distribution = torch.tensor([1.0])
        target.training_history = {"mutated": [77]}
        target.rank_monitor.history = ["mutated-rank"]
        random.seed(77); np.random.seed(77); torch.manual_seed(77)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(77)
        raise failure

    monkeypatch.setattr(
        target.model if component == "model" else target.opt,
        "load_state_dict",
        mutate_everything_then_raise,
    )
    with pytest.raises(BaseException) as caught:
        target.load_checkpoint(str(path))
    assert caught.value is failure
    assert type(caught.value) is expected_type
    assert caught.value.args == expected_args
    _assert_complete_snapshot(target, before)
    assert path.read_bytes() == old_bytes


class _HostileRollbackError(Exception):
    def __str__(self):
        raise AssertionError("rollback error string protocol invoked")

    def __repr__(self):
        raise AssertionError("rollback error repr protocol invoked")


def test_restore_rollback_fault_never_replaces_or_formats_primary_baseexception(
    monkeypatch, tmp_path
) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(2, str(path))
    old_bytes = path.read_bytes()
    target = engine(identity)
    target.rank_monitor = SimpleNamespace(history=["before-rank"])
    random.seed(931); np.random.seed(932); torch.manual_seed(933)
    before = _complete_engine_snapshot(target)
    primary = KeyboardInterrupt("primary checkpoint interruption")
    secondary = _HostileRollbackError("hostile rollback failure")

    def fail_restore_then_arm_rollback_fault(_state):
        with torch.no_grad():
            next(target.model.parameters()).add_(13)
        target.best_score = 13.0
        target.training_history = {"mutated": [13]}
        target.rank_monitor.history = ["mutated-rank"]
        random.seed(13); np.random.seed(13); torch.manual_seed(13)

        def fail_object_graph_rollback(_containers):
            raise secondary

        monkeypatch.setattr(
            engine_module,
            "_restore_exact_object_graph",
            fail_object_graph_rollback,
        )
        raise primary

    monkeypatch.setattr(target.model, "load_state_dict", fail_restore_then_arm_rollback_fault)
    with pytest.raises(BaseException) as caught:
        target.load_checkpoint(str(path))
    assert caught.value is primary
    assert type(caught.value) is KeyboardInterrupt
    assert caught.value.args == ("primary checkpoint interruption",)
    assert caught.value.__notes__ == [
        "checkpoint rollback incomplete: object_graph"
    ]
    assert len(caught.value.__notes__[0]) <= 128
    actual_without_model = _complete_engine_snapshot(target)
    actual_without_model.pop("model")
    before_without_model = copy.deepcopy(before)
    before_without_model.pop("model")
    _assert_nested_exact(actual_without_model, before_without_model)
    assert path.read_bytes() == old_bytes


@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: RuntimeError("checkpoint graph restore failed"),
        lambda: KeyboardInterrupt("checkpoint graph interrupted"),
        lambda: SystemExit(119),
        lambda: GeneratorExit("checkpoint graph generator closed"),
        lambda: _CheckpointRestoreAbort("checkpoint graph aborted"),
    ],
    ids=["runtime", "keyboard", "system-exit", "generator-exit", "custom-base"],
)
def test_failed_checkpoint_restore_preserves_exact_cross_component_object_graph(
    monkeypatch, tmp_path, failure_factory
) -> None:
    identity = run_identity()
    source = engine(identity)
    source.opt.zero_grad()
    sum(parameter.sum() for parameter in source.model.parameters()).backward()
    source.opt.step()
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(4, str(path))
    old_bytes = path.read_bytes()

    target = engine(identity)
    target.opt.zero_grad()
    sum(parameter.sum() for parameter in target.model.parameters()).backward()
    target.opt.step()
    parameter = target.opt.param_groups[0]["params"][0]
    optimizer_state = target.opt.state
    param_groups = target.opt.param_groups
    training_history = target.training_history
    shared = torch.tensor([1.25, 2.5])
    nested_list = [{"shared": shared}]
    nested_tuple = (nested_list, shared)
    optimizer_state[parameter]["cross"] = nested_tuple
    training_history["cross"] = nested_list
    rank_history = [nested_tuple]
    target.rank_monitor = SimpleNamespace(history=rank_history)
    before_value = shared.clone()
    before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    failure = failure_factory()

    def mutate_then_fail(_state):
        shared.add_(119.0)
        optimizer_state.clear()
        param_groups.clear()
        training_history.clear()
        rank_history.clear()
        random.random(); np.random.random(); torch.rand(1)
        raise failure

    monkeypatch.setattr(target.opt, "load_state_dict", mutate_then_fail)
    if isinstance(failure, Exception):
        with pytest.raises(ArtifactCompatibilityError) as caught:
            target.load_checkpoint(str(path))
        assert caught.value.__cause__ is failure
    else:
        with pytest.raises(BaseException) as caught:
            target.load_checkpoint(str(path))
        assert caught.value is failure

    assert target.opt.state is optimizer_state
    assert target.opt.param_groups is param_groups
    assert target.training_history is training_history
    assert target.rank_monitor.history is rank_history
    restored_tuple = optimizer_state[parameter]["cross"]
    assert restored_tuple is nested_tuple
    assert restored_tuple[0] is nested_list
    assert training_history["cross"] is nested_list
    assert rank_history[0] is nested_tuple
    assert restored_tuple[1] is shared
    assert nested_list[0]["shared"] is shared
    assert torch.equal(shared, before_value)
    assert param_groups[0]["params"][0] is parameter
    assert random.getstate() == before_rng[0]
    _assert_nested_exact(np.random.get_state(), before_rng[1])
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    assert path.read_bytes() == old_bytes


@pytest.mark.parametrize("history_size", [1, 10_000])
def test_checkpoint_install_uses_no_redundant_full_state_snapshot(
    monkeypatch, tmp_path, history_size
) -> None:
    identity = run_identity()
    source = engine(identity)
    path = tmp_path / "candidate.pt"
    source.save_checkpoint(6, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    target = engine(identity)
    target.training_history["scaling"] = list(range(history_size))
    counts = {"deepcopy": 0, "model_state": 0, "optimizer_state": 0}
    real_deepcopy = engine_module.copy.deepcopy
    real_model_state = target.model.state_dict
    real_optimizer_state = target.opt.state_dict

    def counted_deepcopy(*args, **kwargs):
        counts["deepcopy"] += 1
        return real_deepcopy(*args, **kwargs)

    def counted_model_state(*args, **kwargs):
        counts["model_state"] += 1
        return real_model_state(*args, **kwargs)

    def counted_optimizer_state(*args, **kwargs):
        counts["optimizer_state"] += 1
        return real_optimizer_state(*args, **kwargs)

    monkeypatch.setattr(engine_module.copy, "deepcopy", counted_deepcopy)
    monkeypatch.setattr(target.model, "state_dict", counted_model_state)
    monkeypatch.setattr(target.opt, "state_dict", counted_optimizer_state)

    assert target._install_checkpoint_v2_atomically(
        payload, str(path), identity
    ) == 7
    assert counts == {"deepcopy": 0, "model_state": 0, "optimizer_state": 0}

def test_checkpoint_moves_resumed_training_tensors_to_current_model_device(
    monkeypatch, tmp_path
) -> None:
    identity = run_identity()
    source = engine(identity)
    source.opt.zero_grad()
    sum(parameter.square().sum() for parameter in source.model.parameters()).backward()
    source.opt.step()
    first = torch.tensor([1.0, 2.0, 4.0, 8.0])
    second = torch.tensor([8.0, 4.0, 2.0, 1.0])
    source.factor_pool = [(0.8, 1, first), (0.7, 2, second)]
    source.factor_pool_scores = [0.8, 0.7]
    source._factor_pool_counter = 3
    source._previous_initial_distribution = torch.tensor([0.2, 0.3, 0.5])
    reward = torch.tensor([2.0])
    expected_corr = source._apply_corr_penalty(reward, first.clone())
    path = tmp_path / "cross-device.pt"
    source.save_checkpoint(4, str(path))

    cpu_target = engine(identity)
    assert cpu_target.load_checkpoint(str(path)) == 5
    assert torch.equal(
        cpu_target._apply_corr_penalty(reward, first.clone()), expected_corr
    )

    target = engine(identity)
    target.model = target.model.to("meta")
    target.opt = torch.optim.AdamW(target.model.parameters(), lr=1.0e-3)
    assert target.load_checkpoint(str(path)) == 5
    parameter_device = next(target.model.parameters()).device

    assert parameter_device.type == "meta"
    assert target.opt.state
    assert all(
        value.device == parameter.device
        for parameter, state in target.opt.state.items()
        for value in state.values()
        if type(value) is torch.Tensor
    )
    pool_tensors = [entry[2] for entry in target.factor_pool]
    assert len(pool_tensors) == 2
    assert all(value.device == parameter_device for value in pool_tensors)
    assert target._previous_initial_distribution.device == parameter_device
    assert target.factor_pool_scores == [0.8, 0.7]
    assert target._factor_pool_counter == 3

    real_any = torch.Tensor.any
    visited_devices = []

    def exact_variation(value):
        visited_devices.append(value.device)
        return True

    def meta_any(value, *args, **kwargs):
        if value.device.type == "meta":
            return torch.tensor(False)
        return real_any(value, *args, **kwargs)

    monkeypatch.setattr(engine_module, "_has_exact_variation", exact_variation)
    monkeypatch.setattr(torch.Tensor, "any", meta_any)
    meta_reward = torch.empty(1, device="meta")
    result = target._apply_corr_penalty(
        meta_reward, torch.empty(4, device="meta")
    )
    assert result is meta_reward
    assert visited_devices == [parameter_device] * 3


def test_checkpoint_pool_device_migration_failure_is_atomic(
    monkeypatch, tmp_path
) -> None:
    identity = run_identity()
    source = engine(identity)
    source.factor_pool = [
        (0.8, 1, torch.tensor([1.0, 2.0])),
        (0.7, 2, torch.tensor([2.0, 1.0])),
    ]
    source._previous_initial_distribution = torch.tensor([0.4, 0.6])
    path = tmp_path / "migration-failure.pt"
    source.save_checkpoint(6, str(path))
    old_bytes = path.read_bytes()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    failure_tensor = payload["factor_pool"][1][2]

    target = engine(identity)
    root = target.factor_pool
    prior = target._previous_initial_distribution
    before = _complete_engine_snapshot(target)
    failure = RuntimeError("second factor-pool migration failed")
    real_to = torch.Tensor.to

    def fail_second_pool_tensor(value, *args, **kwargs):
        if value is failure_tensor:
            raise failure
        return real_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_second_pool_tensor)
    with pytest.raises(ArtifactCompatibilityError) as caught:
        target._install_checkpoint_v2_atomically(payload, str(path), identity)

    assert caught.value.__cause__ is failure
    assert target.factor_pool is root
    assert target._previous_initial_distribution is prior
    _assert_complete_snapshot(target, before)
    assert path.read_bytes() == old_bytes


class _CheckpointSaveAbort(BaseException):
    pass


def test_save_checkpoint_first_callback_failure_restores_exact_entry_state(
    monkeypatch, tmp_path
) -> None:
    identity = run_identity()
    current = engine(identity)
    target = tmp_path / "entry-state.pt"
    current.save_checkpoint(0, str(target))
    old_bytes = target.read_bytes()
    history_root = current.training_history
    steps = history_root.setdefault("step", [1])
    before = _complete_engine_snapshot(current)
    entry_score = current.best_score
    failure = RuntimeError("model state callback failed")

    def mutate_then_fail():
        current.best_score = 777.0
        steps.append(91)
        random.random(); np.random.random(); torch.rand(1)
        raise failure

    real_state_dict = current.model.state_dict
    monkeypatch.setattr(current.model, "state_dict", mutate_then_fail)
    with pytest.raises(RuntimeError) as caught:
        current.save_checkpoint(1, str(target))
    monkeypatch.setattr(current.model, "state_dict", real_state_dict)

    assert caught.value is failure
    assert current.training_history is history_root
    assert current.training_history["step"] is steps
    assert current.best_score == entry_score
    _assert_complete_snapshot(current, before)
    assert target.read_bytes() == old_bytes
    assert not target.with_name(f".{target.name}.tmp").exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "model-state", "optimizer-state", "rng", "identity",
        "filename", "torch-save", "publication",
    ],
)
def test_save_checkpoint_pipeline_failure_is_exact_and_atomic(
    monkeypatch, tmp_path, boundary
) -> None:
    identity = run_identity()
    current = engine(identity)
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", tmp_path)
    target = tmp_path / identity.checkpoint_filename(3)
    current.save_checkpoint(3, str(target))
    old_bytes = target.read_bytes()
    history_root = current.training_history
    steps = history_root.setdefault("step", [3])
    shared_tensor = torch.tensor([1.5, 2.5])
    shared_array = np.array([3.5, 4.5])
    history_root["shared"] = [shared_tensor, shared_array]
    shared_list = history_root["shared"]
    parameter = next(current.model.parameters())
    parameter.grad = torch.ones_like(parameter)
    gradient = parameter.grad
    before = _complete_engine_snapshot(current)
    failure = _CheckpointSaveAbort(f"save aborted at {boundary}")

    def mutate_then_fail(*args, **kwargs):
        current.best_score = 888.0
        steps.append(92)
        with torch.no_grad():
            parameter.add_(5)
            gradient.add_(6)
            shared_tensor.add_(7)
        shared_array[:] = 9
        current.opt.param_groups[0]["lr"] = 99.0
        random.random(); np.random.random(); torch.rand(1)
        raise failure

    restore = None
    if boundary == "model-state":
        restore = (current.model, "state_dict", current.model.state_dict)
        monkeypatch.setattr(current.model, "state_dict", mutate_then_fail)
    elif boundary == "optimizer-state":
        restore = (current.opt, "state_dict", current.opt.state_dict)
        monkeypatch.setattr(current.opt, "state_dict", mutate_then_fail)
    elif boundary == "rng":
        restore = (random, "getstate", random.getstate)
        monkeypatch.setattr(random, "getstate", mutate_then_fail)
    elif boundary == "identity":
        restore = (
            TrainingRunIdentity, "to_dict", TrainingRunIdentity.to_dict,
        )
        monkeypatch.setattr(TrainingRunIdentity, "to_dict", mutate_then_fail)
    elif boundary == "filename":
        restore = (
            TrainingRunIdentity,
            "checkpoint_filename",
            TrainingRunIdentity.checkpoint_filename,
        )
        monkeypatch.setattr(
            TrainingRunIdentity, "checkpoint_filename", mutate_then_fail
        )
    elif boundary == "torch-save":
        restore = (torch, "save", torch.save)
        monkeypatch.setattr(torch, "save", mutate_then_fail)
    else:
        real_publish = current._publish_owned_artifact_temp

        def publish_then_fail(temporary, destination):
            real_publish(temporary, destination)
            mutate_then_fail()

        restore = (
            current,
            "_publish_owned_artifact_temp",
            current._publish_owned_artifact_temp,
        )
        monkeypatch.setattr(
            current, "_publish_owned_artifact_temp", publish_then_fail
        )

    save_path = None if boundary == "filename" else str(target)
    with pytest.raises(BaseException) as caught:
        current.save_checkpoint(4, save_path)
    monkeypatch.setattr(*restore)

    assert caught.value is failure
    assert current.training_history is history_root
    assert current.training_history["step"] is steps
    assert current.training_history["shared"] is shared_list
    assert shared_list[0] is shared_tensor
    assert shared_list[1] is shared_array
    assert parameter.grad is gradient
    _assert_complete_snapshot(current, before)
    assert target.read_bytes() == old_bytes
    assert not target.with_name(f".{target.name}.tmp").exists()


def test_save_checkpoint_refreshes_existing_history_content_at_entry(
    monkeypatch, tmp_path
) -> None:
    current = engine(run_identity())
    target = tmp_path / "history-entry.pt"
    steps = current.training_history.setdefault("step", [])
    steps[:] = [1]
    current.save_checkpoint(0, str(target))
    old_bytes = target.read_bytes()
    root = current.training_history
    steps[0] = 999
    before = _complete_engine_snapshot(current)
    failure = RuntimeError("history entry callback failed")

    def append_then_fail():
        steps.append(77)
        current.best_score = 77.0
        random.random(); np.random.random(); torch.rand(1)
        raise failure

    original = current.model.state_dict
    monkeypatch.setattr(current.model, "state_dict", append_then_fail)
    with pytest.raises(RuntimeError) as caught:
        current.save_checkpoint(1, str(target))
    monkeypatch.setattr(current.model, "state_dict", original)

    assert caught.value is failure
    assert current.training_history is root
    assert current.training_history["step"] is steps
    assert steps == [999]
    _assert_complete_snapshot(current, before)
    assert target.read_bytes() == old_bytes
    assert not target.with_name(f".{target.name}.tmp").exists()


def test_save_checkpoint_cleanup_failure_rolls_back_publication_and_residue(
    monkeypatch, tmp_path
) -> None:
    current = engine(run_identity())
    target = tmp_path / "cleanup.pt"
    current.best_score = 1.0
    current.save_checkpoint(0, str(target))
    old_bytes = target.read_bytes()
    current.best_score = 2.0
    before = _complete_engine_snapshot(current)
    failure = OSError("transaction backup cleanup failed")
    real_unlink = engine_module.pathlib.Path.unlink

    def fail_owned_backup_unlink(path, *args, **kwargs):
        if ".transaction-backup." in path.name:
            raise failure
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        engine_module.pathlib.Path, "unlink", fail_owned_backup_unlink
    )
    with pytest.raises(OSError) as caught:
        current.save_checkpoint(1, str(target))
    monkeypatch.setattr(engine_module.pathlib.Path, "unlink", real_unlink)

    assert caught.value is failure
    _assert_complete_snapshot(current, before)
    assert target.read_bytes() == old_bytes
    assert not target.with_name(f".{target.name}.tmp").exists()
    assert not list(tmp_path.glob("*.transaction-backup.*"))
    assert not list(tmp_path.glob("*.receipt"))


def test_save_checkpoint_registers_owned_backup_before_identity_revalidation(
    monkeypatch, tmp_path
) -> None:
    identity = run_identity()
    current = engine(identity)
    target = tmp_path / "owned.pt"
    current.save_checkpoint(0, str(target))
    old_bytes = target.read_bytes()
    before = _complete_engine_snapshot(current)
    real_snapshot = engine_module._snapshot_artifact_to_owned_backup
    real_unlink = engine_module.pathlib.Path.unlink
    created = []
    cleanup_attempts = []

    def snapshot_then_replace_identity(path):
        result = real_snapshot(path)
        if result[1] is not None:
            created.append(result[1])
            current.run_identity = run_identity(seed=43)
        return result

    def reject_owned_backup_path_unlink(path, *args, **kwargs):
        if ".transaction-backup." in path.name:
            cleanup_attempts.append(path)
            raise OSError("owned backup Path.unlink rejected")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "_snapshot_artifact_to_owned_backup",
        snapshot_then_replace_identity,
    )
    monkeypatch.setattr(
        engine_module.pathlib.Path,
        "unlink",
        reject_owned_backup_path_unlink,
    )
    with pytest.raises(ArtifactCompatibilityError) as caught:
        current.save_checkpoint(1, str(target))

    assert "run_identity" in str(caught.value)
    assert len(created) == 1
    assert cleanup_attempts == created
    assert current.run_identity is identity
    _assert_complete_snapshot(current, before)
    assert target.read_bytes() == old_bytes
    assert not target.with_name(f".{target.name}.tmp").exists()
    assert not list(tmp_path.glob("*.transaction-backup.*"))
    assert not list(tmp_path.glob("*.receipt"))
