from typing import Any, Mapping, Sequence


class NoOpLogger:
    enabled = False

    def log(self, metrics: Mapping[str, Any]) -> None:
        del metrics

    def finish(self) -> None:
        return None


class WandbLogger:
    enabled = True

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        config: Mapping[str, Any],
        group: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=dict(config),
            group=group,
            tags=list(tags) if tags is not None else None,
            reinit=True,
        )

    def log(self, metrics: Mapping[str, Any]) -> None:
        self._wandb.log(dict(metrics))

    def finish(self) -> None:
        self._wandb.finish()


def create_run_logger(
    *,
    use_wandb: bool,
    project: str,
    entity: str | None,
    name: str,
    config: Mapping[str, Any] | None = None,
    group: str | None = None,
    tags: Sequence[str] | None = None,
):
    if not use_wandb:
        return NoOpLogger()
    return WandbLogger(
        project=project,
        entity=entity,
        name=name,
        config=config or {},
        group=group,
        tags=tags,
    )
