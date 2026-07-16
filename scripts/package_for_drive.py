"""
Empacota um equipamento no padrão de pastas SIMPred (Google Drive).
====================================================================

Gera, para um equipamento, a árvore:

    Transpetro/<EQUIP>/
    ├── metadata.csv                       # sensores (equipment, tag, tag_name, value_type, category)
    ├── documentos/                        # documentação técnica (análises + overview)
    ├── dados/<periodo>/
    │   └── data_<inicio>_<fim>_raw.csv    # série temporal bruta
    ├── modelos/
    │   └── model_<inicio>_<fim>_<ARQ>/    # bundle de inferência (se --artifacts-dir for dado)
    │       ├── model.pt | model.pkl
    │       ├── preprocessing.pkl
    │       ├── pipeline.json
    │       └── alarm.json
    └── scripts/
        └── <equip>-exemplo.py             # script de inferência (carrega → processa → infere)

Uso:
    # Estrutura + dados + metadata + doc + script (sem o modelo ainda):
    uv run python scripts/package_for_drive.py --equipment B-8802B

    # Idem + bundle do modelo a partir dos artefatos de um treino (após retreinar):
    uv run python scripts/package_for_drive.py --equipment B-8802B \
        --artifacts-dir results/automl_B-8802B

Saída: deploy/Transpetro/<EQUIP>/  (depois é só arrastar para o Drive).
"""

import argparse
import json
import pickle
import re
import shutil
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps  # noqa: E402
from transpetro_modelos.data.loading import load_equipment_data  # noqa: E402


# Registro dos equipamentos a publicar: config a usar, sigla da arquitetura, tipo, doc extra.
# equipment_type é uma melhor-estimativa — CONFIRME com o cadastro oficial antes de publicar.
REGISTRY = {
    "B-8802B":       {"config": "B-8802B",       "arch": "DENSE", "type": "Bomba",
                      "docs": ["auditoria_pdm.md"]},
    "B-6511502A":    {"config": "B-6511502A",    "arch": "DENSE", "type": "Bomba",
                      "docs": ["auditoria_pdm.md"]},
    "B-4064A":       {"config": "B-4064A-prod",  "arch": "DENSE", "type": "Bomba",
                      "docs": ["analise_b4064a_regime.md", "auditoria_pdm.md"]},
    "B-0302C":       {"config": "B-0302C",       "arch": "LSTM",  "type": "Bomba",
                      "docs": ["analise_b0302c.md"]},
    "B-4703.24001B": {"config": "B-4703.24001B", "arch": "VAE",   "type": "Bomba",
                      "docs": []},
    "B-3403C":       {"config": "B-3403C_interpolated", "arch": "DENSE", "type": "Bomba",
                      "docs": []},
}


# model_type (interno) -> sigla da arquitetura usada no nome da pasta do bundle.
_ARCH_LABEL = {
    "dense": "DENSE", "lstm": "LSTM", "vae": "VAE",
    "ocsvm": "OCSVM", "isolation_forest": "IFOREST", "lof": "LOF",
}


def _arch_from_artifacts(artifacts_dir: Path | None, default: str) -> str:
    """Lê o model_type real do best_trial.pkl p/ rotular a pasta; cai no default se ausente."""
    if not artifacts_dir:
        return default
    trial_path = Path(artifacts_dir) / "best_trial.pkl"
    if not trial_path.exists():
        return default
    with trial_path.open("rb") as f:
        best = pickle.load(f)
    return _ARCH_LABEL.get(best["trial"].model, default)


def slugify(name: str) -> str:
    """'Pressão Sucção' -> 'pressao_succao' (tag no padrão metadata.csv)."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^\w\s-]", "", ascii_str).strip().lower()
    return re.sub(r"[-\s]+", "_", ascii_str)


def category_of(tag_name: str) -> str:
    n = tag_name.lower()
    if "vibra" in n:
        return "Vibração"
    if "temp" in n:
        return "Temperatura"
    if "press" in n:
        return "Pressão"
    if "corrente" in n:
        return "Elétrico"
    if "desloc" in n or "axial" in n:
        return "Deslocamento"
    if "vaz" in n:
        return "Vazão"
    return "Processo"


def build_metadata(equipment: str, columns: list[str], eq_type: str) -> pd.DataFrame:
    """metadata.csv: uma linha por sensor."""
    rows = []
    for col in columns:
        rows.append({
            "equipment": equipment,
            "equipment_type": eq_type,
            "tag": slugify(col),
            "tag_name": col,
            "value_type": "float",
            "category": category_of(col),
        })
    return pd.DataFrame(rows)


def assemble_model_bundle(
    bundle_dir: Path, equipment: str, config_key: str, artifacts_dir: Path
) -> dict:
    """Monta o bundle de inferência a partir dos artefatos de um treino (automl.py)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "best_trial.pkl").open("rb") as f:
        best = pickle.load(f)
    trial = best["trial"]
    results = best["results"]
    model_type = trial.model

    # modelo: sklearn = pickle inteiro; PyTorch = reconstrói a arquitetura, carrega o
    # state_dict e re-salva o MÓDULO INTEIRO (load_bundle no deploy não precisa rebuildar).
    if model_type in ("ocsvm", "isolation_forest", "lof"):
        shutil.copy(artifacts_dir / "best_model.pkl", bundle_dir / "model.pkl")
    else:
        import torch
        from transpetro_modelos.training.automl import build_model
        state = torch.load(artifacts_dir / "best_model.pt", map_location="cpu", weights_only=True)
        # n_features = entrada da 1ª camada (dense/vae: encoder.0.weight [h, in];
        # lstm: encoder.weight_ih_l0 [4h, in]).
        first_w = state.get("encoder.0.weight")
        if first_w is None:
            first_w = state["encoder.weight_ih_l0"]
        n_features = int(first_w.shape[1])
        model = build_model(
            model_type, n_features,
            dense_layers=list(trial.dense_layers) if trial.dense_layers else None,
            seq_len=trial.seq_len,
            lstm_hidden_dim=trial.lstm_hidden_dim,
            lstm_num_layers=trial.lstm_num_layers,
            latent_dim=trial.latent_dim,
        )
        model.load_state_dict(state)
        model.eval()
        torch.save(model, bundle_dir / "model.pt")

    # preprocessing (scaler/clip/coefs ajustados no treino) — gerado pelo automl.py atual
    pp_src = artifacts_dir / "preprocessing.pkl"
    if not pp_src.exists():
        raise FileNotFoundError(
            f"{pp_src} não existe. Retreine com o automl.py atual (que persiste o "
            f"preprocessing.pkl) antes de empacotar o modelo."
        )
    shutil.copy(pp_src, bundle_dir / "preprocessing.pkl")

    # pipeline de preprocessing congelado = pre_split_steps + preset vencedor
    config = EQUIPMENT_CONFIGS[config_key]
    pipeline_steps = list(config.pre_split_steps) + get_preprocessing_steps(config_key, preset=trial.preset)
    (bundle_dir / "pipeline.json").write_text(json.dumps(pipeline_steps, ensure_ascii=False, indent=2))

    # parâmetros de alarme
    select_feats = next(
        (s["features"] for s in config.pre_split_steps if s.get("step") == "select_features"), []
    )
    alarm = {
        "model_type": model_type,
        "threshold": float(results["threshold"]),
        "debounce_consecutive": int(trial.debounce_consecutive),
        "seq_len": int(trial.seq_len),
        "preset": trial.preset,
        "features": select_feats,
        "trained_metrics": {
            k: results.get(k) for k in (
                "prefailure_alert_rate", "normal_alert_rate", "val_fp_rate_heldout", "composite_score"
            ) if k in results
        },
    }
    (bundle_dir / "alarm.json").write_text(json.dumps(alarm, ensure_ascii=False, indent=2))
    return {"model_type": model_type, "preset": trial.preset, "threshold": alarm["threshold"]}


def write_example_script(scripts_dir: Path, equipment: str, bundle_name: str, data_name: str) -> None:
    """Gera o <equip>-exemplo.py seguindo o padrão SIMPred (paths no topo, checagens,
    main() linha-a-linha), mas usando o modelo MULTIVARIADO (todos os sensores juntos,
    erro de reconstrução vs threshold) — diferente do exemplo per-sensor do app."""
    slug = equipment.lower().replace(".", "").replace("-", "")
    script = f'''"""
Exemplo de inferência — {equipment} (padrão SIMPred).

Partes:
    1. Carregamento dos dados
    2. Carregamento do modelo
    3. Transformações e processamento
    4. Inferência do modelo

OBS: diferente do exemplo per-sensor (`model.predict([[valor]])` -> -1/1), este
equipamento usa um modelo MULTIVARIADO (todos os sensores juntos, na escala do treino).
A inferência devolve um erro de reconstrução por instante; o alarme dispara quando o erro
supera o `threshold` calibrado (com debounce). Por isso o preprocessing (scaler/clip/
resample) é parte essencial — ver `preprocessing.pkl` e `pipeline.json` no bundle.

Requisito: o pacote `transpetro_modelos` instalado (fonte do preprocessing e das
arquiteturas). O módulo `simpred_inference.py` acompanha este pacote de deploy.
"""

import os
import sys
from pathlib import Path

# simpred_inference.py acompanha este script (mesma pasta scripts/). Fallback: raiz do deploy.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2]))  # raiz acima de Transpetro/, se existir
from simpred_inference import load_bundle, predict

# Caminhos dos arquivos
HERE = Path(__file__).parent
EQUIP_DIR = HERE.parent  # .../{equipment}
BUNDLE_DIR = EQUIP_DIR / "modelos" / "{bundle_name}"
DATA_FILE = EQUIP_DIR / "dados"  # procura o CSV bruto abaixo desta pasta


def _find_data():
    matches = list(DATA_FILE.rglob("{data_name}"))
    return matches[0] if matches else None


def main():
    # Verifica se os arquivos e diretórios existem
    if not BUNDLE_DIR.exists():
        print(f"Erro: bundle {{BUNDLE_DIR}} não encontrado.")
        return
    data_file = _find_data()
    if data_file is None:
        print(f"Erro: dados não encontrados em {{DATA_FILE}}.")
        return

    # Partes 1+2 — carrega o bundle (modelo multivariado + preprocessing + alarme)
    print(f"Carregando bundle de: {{BUNDLE_DIR}}...")
    bundle = load_bundle(BUNDLE_DIR)

    # Partes 3+4 — processa os dados e infere (erro de reconstrução + flag de anomalia)
    print(f"Lendo dados de {{data_file}}...")
    result = predict(bundle, data_file)

    # Itera sobre cada instante pontuado (apenas os alertas, para não poluir a saída)
    print("\\nIniciando inferência (mostrando os alertas):")
    for ts, row in result[result["is_anomaly"]].iterrows():
        status = "ANOMALIA DETECTADA"
        print(f"Hora: {{ts}} | Erro: {{row['reconstruction_error']:10.4f}} "
              f"| Threshold: {{bundle.threshold:.4f}} | Resultado: {{status}}")

    n_alertas = int(result["is_anomaly"].sum())
    print(f"\\n{equipment}: {{len(result)}} instantes pontuados, {{n_alertas}} alertas.")

    # Persistir o resultado (opcional):
    # result.to_csv(HERE / "{slug}_inferencia.csv")


if __name__ == "__main__":
    main()
'''
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / f"{slug}-exemplo.py").write_text(script)


def package(equipment: str, artifacts_dir: str | Path | None = None,
            out_root: str | Path | None = None, from_clearml: bool = False) -> Path:
    """Monta a árvore SIMPred de um equipamento. Retorna o diretório do equipamento.

    Se `artifacts_dir` for dado (pasta com best_model/preprocessing/best_trial), monta
    também a pasta `modelos/`; senão, gera tudo menos o bundle do modelo.
    `from_clearml=True` baixa os dados do ClearML Dataset (útil quando o feather do
    equipamento não está local, ex.: os datasets interpolados da Lara)."""
    if equipment not in REGISTRY:
        raise ValueError(f"Equipamento {equipment!r} não está no REGISTRY: {list(REGISTRY)}")
    reg = REGISTRY[equipment]
    config_key = reg["config"]
    out_root = Path(out_root) if out_root else PROJECT_ROOT / "deploy" / "Transpetro"
    eq_dir = out_root / equipment
    eq_dir.mkdir(parents=True, exist_ok=True)

    # ── dados: feather -> dados/<periodo>/data_<inicio>_<fim>_raw.csv ──
    df = load_equipment_data(config_key, from_clearml=from_clearml)
    lo, hi = str(df.index.min())[:10], str(df.index.max())[:10]
    period = f"{lo[:4]}_{hi[:4]}"
    dados_dir = eq_dir / "dados" / period
    dados_dir.mkdir(parents=True, exist_ok=True)
    data_name = f"data_{lo}_{hi}_raw.csv"
    df.to_csv(dados_dir / data_name)
    print(f"  dados   : dados/{period}/{data_name}  ({len(df)} linhas)")

    # ── metadata.csv ──
    metadata = build_metadata(equipment, list(df.columns), reg["type"])
    metadata.to_csv(eq_dir / "metadata.csv", index=False)
    print(f"  metadata: metadata.csv  ({len(metadata)} sensores, type='{reg['type']}' — CONFIRMAR)")

    # ── documentos/ ──
    docs_dir = eq_dir / "documentos"
    docs_dir.mkdir(exist_ok=True)
    for doc in ["OVERVIEW.md", *reg["docs"]]:
        src = PROJECT_ROOT / "docs" / doc
        if src.exists():
            shutil.copy(src, docs_dir / doc)
    print(f"  docs    : documentos/  ({len(list(docs_dir.iterdir()))} arquivos)")

    # ── modelos/ (opcional, requer artifacts_dir) ──
    # a sigla da arquitetura vem do modelo realmente treinado (não do palpite do REGISTRY).
    arch = _arch_from_artifacts(Path(artifacts_dir) if artifacts_dir else None, reg["arch"])
    bundle_name = f"model_{lo}_{hi}_{arch}"
    if artifacts_dir:
        info = assemble_model_bundle(
            eq_dir / "modelos" / bundle_name, equipment, config_key, Path(artifacts_dir)
        )
        print(f"  modelos : modelos/{bundle_name}/  ({info['model_type']}, thr={info['threshold']:.5g})")
    else:
        print(f"  modelos : (pulado — rode com --artifacts-dir após retreinar; bundle = {bundle_name})")

    # ── scripts/ (exemplo + simpred_inference.py para o bundle viajar autossuficiente) ──
    scripts_dir = eq_dir / "scripts"
    write_example_script(scripts_dir, equipment, bundle_name, data_name)
    inf_src = PROJECT_ROOT / "deploy" / "simpred_inference.py"
    if inf_src.exists():
        shutil.copy(inf_src, scripts_dir / "simpred_inference.py")
    # wheel do pacote (engenharia instala uma vez) — pega o mais recente em dist/
    wheels = sorted((PROJECT_ROOT / "dist").glob("transpetro_modelos-*.whl"))
    if wheels:
        shutil.copy(wheels[-1], scripts_dir / wheels[-1].name)
    print(f"  scripts : scripts/{equipment.lower().replace('.', '').replace('-', '')}-exemplo.py + simpred_inference.py")

    print(f"\n✓ {equipment} empacotado em: {eq_dir}")
    return eq_dir


def main():
    ap = argparse.ArgumentParser(description="Empacota um equipamento no padrão SIMPred (Drive).")
    ap.add_argument("--equipment", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--artifacts-dir", default=None,
                    help="Pasta com os artefatos do treino (best_model/preprocessing/best_trial). "
                         "Se omitido, gera tudo menos o bundle do modelo.")
    ap.add_argument("--out", default=None, help="Raiz de saída (default: deploy/Transpetro)")
    args = ap.parse_args()
    package(args.equipment, artifacts_dir=args.artifacts_dir, out_root=args.out)


if __name__ == "__main__":
    main()
