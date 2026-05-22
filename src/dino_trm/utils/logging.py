"""Thin wandb wrapper that degrades to a no-op when logging is disabled.

Lets train.py call ``logger.log({...})`` unconditionally without sprinkling
``if use_wandb`` everywhere. Project defaults to ``dino-trm``.
"""

from __future__ import annotations

from typing import Any


class Logger:
    def __init__(
        self,
        enabled: bool = True,
        project: str = "dino-trm",
        run_name: str | None = None,
        config: dict | None = None,
        mode: str = "online",
    ) -> None:
        self.enabled = enabled
        self._wandb = None
        if enabled:
            import wandb

            self._wandb = wandb
            wandb.init(project=project, name=run_name, config=config or {}, mode=mode)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log(data, step=step)

    def log_image(self, key: str, figure, step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log({key: self._wandb.Image(figure)}, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
