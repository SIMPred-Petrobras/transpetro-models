"""
Checkpoint/resume do AutoML (grid search).
==========================================

Persiste o progresso do grid (trials já concluídos) para que uma task que caia no
ClearML (falta de energia, abort) possa ser re-enfileirada e **retomar de onde parou**
em vez de recomeçar do zero.

Estratégia:
  - Cada trial tem uma ASSINATURA estável (hash determinístico do TrialConfig).
  - O estado salvo é `{"rows": [...], "attempted": [sig, ...]}`: as linhas de resultado
    já computadas + as assinaturas já tentadas (concluídas, puladas ou com erro).
  - No ClearML, o checkpoint é um artifact (`progress_checkpoint`) ancorado no NOME
    estável da task. No startup busca-se: (1) o artifact da própria task (re-enfileiramento
    do mesmo id) e, se ausente, (2) a task mais recente com o mesmo nome que tenha o artifact
    (re-run com id novo). Em run local, é um pickle em disco.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

CHECKPOINT_ARTIFACT = "progress_checkpoint"


def trial_signature(trial: Any) -> str:
    """Hash determinístico do TrialConfig — estável entre execuções com os mesmos args."""
    def norm(v):
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, tuple):
            return list(v)
        return v

    payload = {k: norm(v) for k, v in asdict(trial).items()}
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def make_payload(rows: list[dict], attempted: set[str]) -> dict:
    return {"rows": rows, "attempted": sorted(attempted)}


# ── disco local ──────────────────────────────────────────────────────────────
def save_local(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)  # escrita atômica (não corrompe o checkpoint se cair no meio)


def load_local(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ── ClearML ──────────────────────────────────────────────────────────────────
def _payload_from_task(task) -> dict | None:
    try:
        if task is not None and CHECKPOINT_ARTIFACT in (task.artifacts or {}):
            obj = task.artifacts[CHECKPOINT_ARTIFACT].get()
            if isinstance(obj, dict) and "rows" in obj:
                return obj
    except Exception:
        pass
    return None


def fetch_resume_payload(task, explicit_task_id: str | None = None) -> tuple[dict | None, str]:
    """Retorna (payload, origem). origem é uma string descritiva ('' se nada encontrado).

    Resume só a partir da PRÓPRIA task: re-enfileirar a mesma task abortada traz o seu
    artifact `progress_checkpoint`. `explicit_task_id` permite apontar outra task manualmente."""
    if explicit_task_id:
        from clearml import Task
        p = _payload_from_task(Task.get_task(task_id=explicit_task_id))
        return p, f"task {explicit_task_id}" if p else ""

    p = _payload_from_task(task)
    return (p, "própria task (re-enfileirada)") if p else (None, "")


def upload_to_task(task, payload: dict) -> None:
    """Sobe/atualiza o artifact de checkpoint na task (bloqueia até subir; payload é pequeno)."""
    task.upload_artifact(CHECKPOINT_ARTIFACT, artifact_object=payload, wait_on_upload=True)
