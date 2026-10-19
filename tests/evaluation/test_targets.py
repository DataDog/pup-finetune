# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The replay's samplers: from a live training session, or a new one for the best weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pup_finetune.errors import FinetuneError, StateError
from pup_finetune.evaluation.targets import (
    best_training_checkpoint,
    run_samplers,
    session_samplers,
    tuned_checkpoint_name,
)
from pup_finetune.providers.base import AdamConfig, CheckpointRef, LoraConfig, TrainSpec
from pup_finetune.providers.fake import (
    BASE_ANSWER,
    FAKE_PROFILE,
    MAX_CHECKPOINT_NAME_LEN,
    FakeBackend,
    FakeCrash,
    FakeProvider,
    FakeRenderer,
    FakeSampler,
    FakeSession,
)
from pup_finetune.runs import RunDir
from pup_finetune.runtime import Runtime
from pup_finetune.training.records import TrainResult

LORA = LoraConfig(rank=8)
SESSION: dict[str, Any] = {"seed": 0, "max_seq_len": FAKE_PROFILE.max_seq_len}
TOOLS = [
    {"type": "function", "function": {"name": "lookup_order", "parameters": {"type": "object"}}}
]
TRAINED = [
    {"role": "user", "content": "Where is order 7?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_order", "arguments": '{"id": "7"}'},
            }
        ],
    },
]
HELD_OUT = [{"role": "user", "content": "Where is order 9?"}]


def train(session: FakeSession) -> tuple[CheckpointRef, CheckpointRef]:
    """One step on a conversation that calls the first tool after a user message."""
    example = FakeRenderer().supervised_example(TRAINED, TOOLS)
    session.forward_backward([example]).result()
    session.optim_step(AdamConfig(learning_rate=1e-4)).result()
    training = session.save_training_checkpoint("t1s1-abcd", epoch=1, step=1)
    sampler = session.save_sampler_checkpoint("s1s1-abcd", epoch=1, step=1)
    return training, sampler


def result(
    training: CheckpointRef | None,
    sampler: CheckpointRef | None,
    *,
    last: CheckpointRef | None = None,
) -> TrainResult:
    return TrainResult(
        best_epoch=None if training is None else training.epoch,
        best_val_loss=1.0,
        stopped_reason="completed",
        epochs=[],
        best_sampler_ckpt=sampler,
        best_training_ckpt=training,
        last_training_ckpt=training if last is None else last,
        warnings=[],
    )


def test_the_live_session_reuses_the_best_sampler_checkpoint() -> None:
    backend = FakeBackend()
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        training, sampler = train(session)
        saved = len(backend.saved)
        with session_samplers(
            session, result(training, sampler), run_id="ft-x", max_checkpoint_name_len=17
        ) as targets:
            assert targets.tuned_checkpoint == sampler
            assert len(backend.saved) == saved  # nothing new saved
            base, tuned = targets.samplers["base"], targets.samplers["tuned"]
            assert isinstance(base, FakeSampler)
            assert isinstance(tuned, FakeSampler)
            assert base.generate(HELD_OUT, TOOLS, max_tokens=64).message["content"] == BASE_ANSWER
            call = tuned.generate(HELD_OUT, TOOLS, max_tokens=64).message["tool_calls"][0]
            assert call["function"] == {"name": "lookup_order", "arguments": "{}"}
        assert backend.sampled == [None, sampler]
    with pytest.raises(FinetuneError, match="closed"):
        tuned.generate(HELD_OUT, TOOLS, max_tokens=64)


def test_a_new_session_recreates_the_tuned_sampler_from_the_training_checkpoint() -> None:
    backend = FakeBackend()
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        training, sampler = train(session)
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as later:
        limit = MAX_CHECKPOINT_NAME_LEN
        with session_samplers(
            later, result(training, sampler), run_id="ft-x", max_checkpoint_name_len=limit
        ) as targets:
            made = targets.tuned_checkpoint
            assert made.kind == "sampler"
            assert made.created_in_session == later.session_id
            assert made.name == tuned_checkpoint_name("ft-x", training, limit=limit)
            assert (made.epoch, made.step) == (training.epoch, training.step)
            tuned = targets.samplers["tuned"]
            message = tuned.generate(HELD_OUT, TOOLS, max_tokens=64).message
            assert message["tool_calls"][0]["function"]["name"] == "lookup_order"


def test_only_the_targets_asked_for_get_a_sampler() -> None:
    backend = FakeBackend()
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        training, sampler = train(session)
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as later:
        saved = len(backend.saved)
        with session_samplers(
            later,
            result(training, sampler),
            run_id="ft-x",
            max_checkpoint_name_len=17,
            targets=["tuned"],
        ) as targets:
            assert list(targets.samplers) == ["tuned"]
        assert None not in backend.sampled
        assert len(backend.saved) == saved + 1
        with session_samplers(
            later,
            result(training, sampler),
            run_id="ft-x",
            max_checkpoint_name_len=17,
            targets=["base"],
        ) as targets:
            assert list(targets.samplers) == ["base"]
            assert targets.tuned_checkpoint is None
        assert len(backend.saved) == saved + 1  # no ev... checkpoint for base alone


def test_tuned_checkpoint_names_are_short_and_deterministic() -> None:
    ref = CheckpointRef(
        kind="training", name="t", uri="fake://x", epoch=1, step=1, created_in_session="s"
    )
    name = tuned_checkpoint_name("ft-run", ref, limit=17)
    assert name.startswith("ev")
    assert len(name) <= 17
    assert name == tuned_checkpoint_name("ft-run", ref, limit=17)
    assert name != tuned_checkpoint_name("ft-other", ref, limit=17)
    assert len(tuned_checkpoint_name("ft-run", ref, limit=5)) == 5
    with pytest.raises(ValueError):
        tuned_checkpoint_name("ft-run", ref, limit=2)


def test_a_result_without_a_best_epoch_has_nothing_to_evaluate() -> None:
    with pytest.raises(StateError, match="no best epoch"):
        best_training_checkpoint(result(None, None))


def test_the_tuned_target_is_the_best_epoch_not_the_last() -> None:
    backend = FakeBackend()
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        best, sampler = train(session)
        example = FakeRenderer().supervised_example(TRAINED, TOOLS)
        session.forward_backward([example]).result()
        session.optim_step(AdamConfig(learning_rate=1e-4)).result()
        last = session.save_training_checkpoint("t2s2-abcd", epoch=2, step=2)
    assert (best.uri, best.epoch, best.step) != (last.uri, last.epoch, last.step)
    finished = result(best, sampler, last=last)
    assert best_training_checkpoint(finished) == best
    with (
        backend.open_session(FAKE_PROFILE, LORA, **SESSION) as later,
        session_samplers(
            later, finished, run_id="ft-x", max_checkpoint_name_len=MAX_CHECKPOINT_NAME_LEN
        ) as targets,
    ):
        made = targets.tuned_checkpoint
        assert made.name == tuned_checkpoint_name("ft-x", best, limit=MAX_CHECKPOINT_NAME_LEN)
        assert made.name != tuned_checkpoint_name("ft-x", last, limit=MAX_CHECKPOINT_NAME_LEN)
        assert (made.epoch, made.step) == (best.epoch, best.step)


def fake_run(tmp_path: Path) -> tuple[FakeProvider, RunDir, TrainResult]:
    """A fake run directory and the result of training it, in a session since closed."""
    provider = FakeProvider(Runtime(dev_mode=True))
    run = RunDir.create(
        tmp_path / "run",
        provider="fake",
        profile=FAKE_PROFILE,
        spec=TrainSpec(max_epochs=1, batch_size=1),
        dataset_dir=tmp_path / "dataset",
        dataset_sha256="d" * 64,
        rendering=FakeRenderer().identity(),
        run_id="ft-20261006123045-abcdef",
    )
    with provider.backend().open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        training, sampler = train(session)
    return provider, run, result(training, sampler)


def test_run_samplers_opens_a_session_and_ends_it_on_exit(tmp_path: Path) -> None:
    provider, run, finished = fake_run(tmp_path)
    backend = provider.backend()
    with run_samplers(provider, run, finished) as targets:
        session_id = backend.opened[-1]
        assert backend.is_live(session_id)
        assert targets.tuned_checkpoint.created_in_session == session_id
        samplers = dict(targets.samplers)
        assert samplers["base"].generate(HELD_OUT, TOOLS, max_tokens=64).message
    assert not backend.is_live(session_id)
    for sampler in samplers.values():
        with pytest.raises(FinetuneError, match="closed"):
            sampler.generate(HELD_OUT, TOOLS, max_tokens=64)


def test_run_samplers_ends_the_session_when_the_body_fails(tmp_path: Path) -> None:
    provider, run, finished = fake_run(tmp_path)
    backend = provider.backend()
    with pytest.raises(RuntimeError, match="replay broke"), run_samplers(provider, run, finished):
        raise RuntimeError("replay broke")
    assert backend.closed[-1] == backend.opened[-1]
    assert not backend.is_live(backend.opened[-1])


@pytest.mark.parametrize("step", ["load_weights", "save_sampler_checkpoint"])
def test_the_base_sampler_is_closed_when_the_tuned_one_cannot_be_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    provider, run, finished = fake_run(tmp_path)
    made: list[FakeSampler] = []
    real_sampler = FakeSession.sampler

    def sampler(self: FakeSession, ckpt: CheckpointRef | None) -> FakeSampler:
        made.append(real_sampler(self, ckpt))
        return made[-1]

    def fail(self: FakeSession, *args: Any, **kwargs: Any) -> Any:
        raise FakeCrash(f"{step} failed")

    monkeypatch.setattr(FakeSession, "sampler", sampler)
    monkeypatch.setattr(FakeSession, step, fail)
    with pytest.raises(FakeCrash, match=step), run_samplers(provider, run, finished):
        pytest.fail("the body must not run")
    assert len(made) == 1  # the base sampler only
    with pytest.raises(FinetuneError, match="closed"):
        made[0].generate(HELD_OUT, TOOLS, max_tokens=64)
    assert not provider.backend().is_live(provider.backend().opened[-1])


def test_a_sampler_that_fails_to_close_is_noted(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: FakeSampler) -> None:
        raise FakeCrash("Bearer abcdef0123456789secret")

    monkeypatch.setattr(FakeSampler, "close", broken)
    backend = FakeBackend()
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        training, sampler = train(session)
        with session_samplers(
            session, result(training, sampler), run_id="ft-x", max_checkpoint_name_len=17
        ):
            pass
    err = capsys.readouterr().err
    assert "closing the base sampler failed" in err
    assert "closing the tuned sampler failed" in err
    assert "abcdef0123456789secret" not in err
