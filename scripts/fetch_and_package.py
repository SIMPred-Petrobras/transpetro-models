"""
Baixa os artefatos de uma task de AutoML concluída no ClearML e empacota o
equipamento no padrão SIMPred (Drive), em um só comando.

Pré-requisitos:
  - a task precisa ter sido treinada com o automl.py atual, que sobe os artifacts:
    `best_model`, `preprocessing`, `best_trial` (e `best_full_scores`, `automl_results`).
  - ClearML configurado localmente (clearml.conf / clearml-init).

Uso:
    # por task_id (recomendado — copie do dashboard ClearML):
    uv run python scripts/fetch_and_package.py --equipment B-8802B --task-id <ID>

    # ou pelo nome da task:
    uv run python scripts/fetch_and_package.py --equipment B-8802B \
        --task-name automl-b8802b-deploy

Saída: deploy/Transpetro/<EQUIP>/ (depois é só arrastar para o Drive).
"""

import argparse
import pickle
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # para importar package_for_drive

from clearml import Task  # noqa: E402

from package_for_drive import REGISTRY, package  # noqa: E402
from transpetro_modelos.training.automl import TrialConfig  # noqa: E402

_SKLEARN = ("ocsvm", "isolation_forest", "lof")


def _get_task(task_id: str | None, task_name: str | None, project: str) -> Task:
    if task_id:
        return Task.get_task(task_id=task_id)
    if task_name:
        return Task.get_task(project_name=project, task_name=task_name)
    raise SystemExit("Informe --task-id ou --task-name.")


def _reconstruct_trial(trial_dict: dict) -> TrialConfig:
    """Reconstrói o TrialConfig a partir do dict subido pelo automl.py.

    O empacotamento só usa model/preset/dense_layers/seq_len/lstm_*/latent_dim/
    debounce; val_start (string ISO no artifact) não é usado, então é zerado."""
    valid = {f.name for f in fields(TrialConfig)}
    kwargs = {k: v for k, v in trial_dict.items() if k in valid}
    kwargs["val_start"] = None
    if isinstance(kwargs.get("dense_layers"), list):
        kwargs["dense_layers"] = tuple(kwargs["dense_layers"])
    return TrialConfig(**kwargs)


def fetch_to_artifacts_dir(task: Task, dest: Path) -> str:
    """Baixa best_model/preprocessing/best_trial para `dest` no formato que o
    package_for_drive espera (best_model.pt|pkl, preprocessing.pkl, best_trial.pkl).
    Retorna o model_type."""
    dest.mkdir(parents=True, exist_ok=True)
    arts = task.artifacts
    for need in ("best_model", "preprocessing", "best_trial"):
        if need not in arts:
            raise SystemExit(
                f"A task {task.id} não tem o artifact '{need}'. Ela foi treinada com o "
                f"automl.py atual (que sobe best_model/preprocessing/best_trial)?"
            )

    best_trial = arts["best_trial"].get()           # {"trial": dict, "results": dict}
    trial = _reconstruct_trial(best_trial["trial"])
    results = best_trial["results"]
    model_type = trial.model

    # best_trial.pkl no formato do package_for_drive (objeto TrialConfig + results)
    with (dest / "best_trial.pkl").open("wb") as f:
        pickle.dump({"trial": trial, "results": results}, f)

    # preprocessing.pkl (objeto PreprocessingArtifacts)
    preprocessing = arts["preprocessing"].get()
    with (dest / "preprocessing.pkl").open("wb") as f:
        pickle.dump(preprocessing, f)

    # best_model (arquivo) -> best_model.pt | best_model.pkl
    model_local = Path(arts["best_model"].get_local_copy())
    target = dest / ("best_model.pkl" if model_type in _SKLEARN else "best_model.pt")
    target.write_bytes(model_local.read_bytes())

    print(f"  artefatos baixados: best_model ({model_type}), preprocessing, best_trial")
    return model_type


def main():
    ap = argparse.ArgumentParser(description="Baixa artefatos do ClearML e empacota p/ o Drive.")
    ap.add_argument("--equipment", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--task-id", default=None, help="ID da task ClearML concluída.")
    ap.add_argument("--task-name", default=None, help="Nome da task (alternativa ao --task-id).")
    ap.add_argument("--project", default="Transpetro", help="Projeto ClearML (default: Transpetro).")
    ap.add_argument("--out", default=None, help="Raiz de saída (default: deploy/Transpetro).")
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="Mantém a pasta temporária com os artefatos baixados.")
    ap.add_argument("--from-clearml", action="store_true",
                    help="Baixa os dados do ClearML Dataset (quando o feather não está local).")
    args = ap.parse_args()

    task = _get_task(args.task_id, args.task_name, args.project)
    print(f"Task: {task.id}  ({task.name})")

    tmp = Path(tempfile.mkdtemp(prefix=f"fetch_{args.equipment}_"))
    fetch_to_artifacts_dir(task, tmp)
    package(args.equipment, artifacts_dir=tmp, out_root=args.out, from_clearml=args.from_clearml)

    if args.keep_artifacts:
        print(f"  (artefatos brutos mantidos em {tmp})")
    else:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
