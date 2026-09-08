"""Verified in-memory continuation hooks for the frozen trainer.

The frozen file is never edited. Its exact ``run_training`` source is compiled
with three narrowly registered insertions and installed only for the duration
of the validation-only call.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Any, Iterator

from lap_gnn_tf.training import trainer

from .continuation import EpochBoundaryContinuationManager, ExactContinuationError


RESTORE_ANCHOR = "    history = []\n"
LOOP_ANCHOR = "    for epoch in range(1, max_epochs + 1):\n"
PERSIST_ANCHOR = "        _print_epoch_summary(row, policy)\n"

RESTORE_INSERTION = """    history = []
    _continuation_manager = globals().get("_lap_ls005_continuation_manager")
    _continuation_start_epoch = 1
    if _continuation_manager is not None:
        _continuation_start_epoch, history = _continuation_manager.restore_or_initialize(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early=early,
            policy=policy,
            train_data=train_data,
            config=config,
            output_dir=output_dir,
        )
"""
LOOP_INSERTION = (
    "    for epoch in range(_continuation_start_epoch, max_epochs + 1):\n"
)
PERSIST_INSERTION = """        _print_epoch_summary(row, policy)
        if _continuation_manager is not None:
            _continuation_manager.persist_epoch_boundary(
                completed_epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                early=early,
                policy=policy,
                train_data=train_data,
                config=config,
                history=history,
                output_dir=output_dir,
            )
"""


def transformed_run_training_source() -> str:
    """Return the frozen function with exactly the registered hook insertions."""

    source = inspect.getsource(trainer.run_training)
    replacements = (
        (RESTORE_ANCHOR, RESTORE_INSERTION),
        (LOOP_ANCHOR, LOOP_INSERTION),
        (PERSIST_ANCHOR, PERSIST_INSERTION),
    )
    for anchor, replacement in replacements:
        if source.count(anchor) != 1:
            raise ExactContinuationError(
                f"Frozen trainer continuation anchor drift: {anchor!r}"
            )
        source = source.replace(anchor, replacement, 1)
    return source


def compile_transformed_run_training() -> Any:
    namespace: dict[str, Any] = {}
    exec(
        compile(
            transformed_run_training_source(),
            str(inspect.getsourcefile(trainer.run_training)),
            "exec",
        ),
        trainer.__dict__,
        namespace,
    )
    transformed = namespace.get("run_training")
    if not callable(transformed):
        raise ExactContinuationError("Transformed trainer compilation failed")
    return transformed


@contextmanager
def continuation_enabled_trainer(
    manager: EpochBoundaryContinuationManager,
) -> Iterator[None]:
    """Install hooks temporarily and restore every binding on all exits."""

    original = trainer.run_training
    if hasattr(trainer, "_lap_ls005_continuation_manager"):
        raise ExactContinuationError("Continuation trainer is already active")
    transformed = compile_transformed_run_training()
    trainer._lap_ls005_continuation_manager = manager
    trainer.run_training = transformed
    try:
        yield
    finally:
        trainer.run_training = original
        del trainer._lap_ls005_continuation_manager
