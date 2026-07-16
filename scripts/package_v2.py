"""
package_v2.py — GERADOR do pacote de inferência autocontido `deploy_v2/`.

A partir dos bundles já treinados em `deploy/Transpetro/<EQUIP>/`, produz o
entregável `deploy_v2/` que o time de engenharia integra — SEM depender da lib
`transpetro_modelos`:

    deploy_v2/
    └── Transpetro/
        ├── simpred_inference.py        # módulo compartilhado (VAE + 4 passos)
        ├── requirements.txt
        ├── README.md
        └── <EQUIP>/
        ├── modelos/<bundle>/           # model_state.pt, model_arch.json, scaler.pkl,
        │                                 clip_bounds.json, pipeline.json, alarm.json
        ├── dados/<periodo>/*.csv
        └── scripts/<equip>_exemplo.py  # script fino: aponta o bundle e chama os 4 passos

O que faz por equipamento:
  1. Converte model.pt (módulo inteiro, exige a lib) -> model_state.pt (pesos) + model_arch.json.
  2. Converte preprocessing.pkl (objeto da lib) -> scaler.pkl (RobustScaler puro) + clip_bounds.json.
  3. Copia alarm.json, pipeline.json e os dados brutos.
  4. Escreve o script fino do equipamento.

NÃO retreina nada: reempacota o modelo já validado.
NÃO sobrescreve o README.md (documentação mantida à mão) — só o cria se faltar.

Uso (com a venv do repo, que tem torch + a lib para ler os model.pt originais):
    .venv/bin/python scripts/package_v2.py
    .venv/bin/python scripts/package_v2.py --equip B-8802B   # só um
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "deploy" / "Transpetro"
OUT_ROOT = REPO / "deploy_v2"
OUT = OUT_ROOT / "Transpetro"

# equipamento -> pasta do bundle em deploy/Transpetro/<EQUIP>/modelos/
EQUIPS = {
    "B-8802B":    "model_2022-05-15_2022-07-21_VAE",
    "B-6511502A": "model_2022-05-17_2023-05-17_VAE",
    "B-3403C":    "model_2023-01-01_2023-09-15_DENSE",
}


def infer_arch(model, model_type: str) -> dict:
    """Extrai input_dim / encoding_layers (+ latent_dim se VAE) de um modelo carregado."""
    import torch.nn as nn
    enc_linears = [m for m in model.encoder if isinstance(m, nn.Linear)]
    arch = {
        "model_type": model_type,
        "input_dim": int(enc_linears[0].in_features),
        "encoding_layers": [int(l.out_features) for l in enc_linears],
    }
    if model_type == "vae":
        arch["latent_dim"] = int(model.fc_mean.out_features)
    return arch


# ════════════════════════════════════════════════════════════════════════════
# requirements.txt
# ════════════════════════════════════════════════════════════════════════════
REQUIREMENTS = """# Dependências do pacote de inferência SIMPred (deploy_v2).
# Versões testadas no ambiente de treino (Python 3.12).
#
# IMPORTANTE: scaler.pkl é um pickle de sklearn.preprocessing.RobustScaler, então
# a versão do scikit-learn precisa ser compatível com a usada no treino (1.8.x).
# Para torch, ajuste a build conforme o ambiente (CPU vs CUDA) — a inferência roda em CPU.

numpy>=2.4,<3
pandas>=3.0,<4
scikit-learn>=1.8,<1.9
torch>=2.11,<3
"""


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO COMPARTILHADO — deploy_v2/simpred_inference.py (verbatim)
# ════════════════════════════════════════════════════════════════════════════
MODULE_SOURCE = r'''"""
SIMPred — módulo de inferência de anomalias (COMPARTILHADO e AUTOCONTIDO).

Não depende de nenhuma biblioteca interna nossa. Requisitos: pandas, numpy, torch,
scikit-learn. É importado pelos scripts de exemplo de cada equipamento
(`Transpetro/<EQUIP>/scripts/<equip>_exemplo.py`), que só apontam o bundle e chamam
os 4 passos abaixo.

Os 4 passos do padrão SIMPred:
    1) carregar_dados(csv)        — lê o CSV bruto dos sensores (índice temporal)
    2) preprocessar(bundle, df)   — reaplica EXATAMENTE o preprocessing do treino
    3) carregar_modelo(bundle)    — reconstrói a arquitetura e carrega os pesos treinados
    4) prever(bundle, model, df)  — erro de reconstrução -> atenção/alarme (com debounce)

Um "bundle" é a pasta `modelos/model_<inicio>_<fim>_<ARQ>/`:
    model_state.pt      pesos do modelo treinado
    model_arch.json     hiperparâmetros da arquitetura (input_dim / encoding_layers / latent_dim)
    scaler.pkl          RobustScaler do sklearn ajustado no treino
    clip_bounds.json    limites de clip por sensor (percentil do treino)
    pipeline.json       passos de preprocessing (ordem + parâmetros)
    alarm.json          model_type, threshold, atenção, debounce, features
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# ════════════════════════════════════════════════════════════════════════════
# ARQUITETURA DO MODELO (VAE)
# ════════════════════════════════════════════════════════════════════════════
class VAE(nn.Module):
    """Autoencoder variacional. Em inferência é determinístico (usa a média do latente)."""

    def __init__(self, input_dim, encoding_layers, latent_dim):
        super().__init__()
        enc_dims = [input_dim] + encoding_layers
        enc = []
        for i in range(len(enc_dims) - 1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i + 1]),
                    nn.BatchNorm1d(enc_dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        self.fc_mean = nn.Linear(encoding_layers[-1], latent_dim)
        self.fc_log_var = nn.Linear(encoding_layers[-1], latent_dim)

        dec_dims = [latent_dim] + list(reversed(encoding_layers))
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec += [nn.BatchNorm1d(dec_dims[i + 1]), nn.ReLU()]
        dec.append(nn.Linear(dec_dims[-1], input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)          # inferência: usa a média (sem sampling)
        return self.decoder(mean), mean


class DenseAutoencoder(nn.Module):
    """Autoencoder denso (determinístico). Erro de reconstrução = MSE(entrada, saída)."""

    def __init__(self, input_dim, encoding_layers):
        super().__init__()
        enc_dims = [input_dim] + encoding_layers
        enc = []
        for i in range(len(enc_dims) - 1):
            enc.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:           # sem BN/ReLU no gargalo
                enc += [nn.BatchNorm1d(enc_dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        dec_dims = list(reversed(encoding_layers)) + [input_dim]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:           # sem ativação na saída
                dec += [nn.BatchNorm1d(dec_dims[i + 1]), nn.ReLU()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded


# ════════════════════════════════════════════════════════════════════════════
# 1) CARREGAR DADOS
# ════════════════════════════════════════════════════════════════════════════
def carregar_dados(csv_path) -> pd.DataFrame:
    """Lê o CSV bruto; a 1ª coluna (ou 'datetime') vira o índice temporal."""
    df = pd.read_csv(csv_path)
    col_tempo = "datetime" if "datetime" in df.columns else df.columns[0]
    df = df.set_index(col_tempo)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        raise ValueError(f"coluna de tempo '{col_tempo}' não é uma data válida: {e}") from e
    return df.sort_index()


def achar_csv(dados_dir, nome) -> Path:
    """Localiza o CSV bruto abaixo de dados/ (aceita nome exato ou glob)."""
    caminhos = list(Path(dados_dir).rglob(nome))
    if not caminhos:
        raise FileNotFoundError(f"CSV bruto '{nome}' nao encontrado em {dados_dir}")
    return caminhos[0]


# ════════════════════════════════════════════════════════════════════════════
# 2) PRE-PROCESSAMENTO
#    Passos NUMERICOS (clip + normalizacao) -> sklearn.Pipeline.
#    Passos TEMPORAIS (descartam linhas / dependem do indice) -> funcoes explicitas
#    (nao encaixam no contrato do sklearn: n_amostras entra != n_amostras sai).
# ════════════════════════════════════════════════════════════════════════════
def remove_sensor_errors(df, error_values):
    """Troca códigos de erro de sensor (ex.: 0.0, 32767.0) por NaN."""
    for v in error_values:
        df = df.replace(v, np.nan)
    return df

def filter_running(df, column, threshold):
    """Mantém só as linhas com a bomba LIGADA (coluna acima do limiar)."""
    if column not in df.columns:
        return df
    return df[df[column] > threshold].copy()

def remove_transients(df, minutes=10, gap_minutes=5):
    """Remove os primeiros `minutes` após cada religamento (gap > gap_minutes no tempo)."""
    if len(df) == 0:
        return df
    gap = pd.Timedelta(minutes=gap_minutes)
    reinicios = df.index[df.index.to_series().diff() > gap]
    mask = pd.Series(True, index=df.index)
    mask[df.index < df.index[0] + pd.Timedelta(minutes=minutes)] = False
    for t in reinicios:
        mask[(df.index >= t) & (df.index < t + pd.Timedelta(minutes=minutes))] = False
    return df[mask].copy()

def resample(df, freq="5min", agg="last"):
    """Reamostra a série COV para uma grade regular (último valor da janela)."""
    r = df.resample(freq)
    return r.last() if agg == "last" else getattr(r, agg)()

def ffill(df, limit=4):
    """Preenche NaN para frente até `limit` períodos; descarta o que sobrar."""
    return df.ffill(limit=limit).dropna()

def moving_average(df, window=3, min_periods=1, columns=None):
    """Média móvel causal (suavização) nas colunas indicadas (default: todas)."""
    if columns is None:
        columns = list(df.columns)
    df = df.copy()
    df[columns] = df[columns].rolling(window=window, min_periods=min_periods, center=False).mean()
    return df

def select_features(df, features):
    """Seleciona (e ordena) as colunas usadas pelo modelo (erro amigável se faltar sensor)."""
    faltando = [c for c in features if c not in df.columns]
    if faltando:
        raise ValueError(
            "Sensores ausentes no CSV: " + ", ".join(faltando)
            + "\nSensores esperados: " + ", ".join(features)
        )
    return df[features].copy()


class Clipper(BaseEstimator, TransformerMixin):
    """Transformer sklearn: winsoriza cada sensor nos limites [min, max] do treino.

    Os limites vêm de clip_bounds.json (percentil calculado no treino); aqui só
    aplicamos, nunca reajustamos -> sem vazamento.
    """
    def __init__(self, bounds):
        self.bounds = bounds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X chega como DataFrame (colunas = sensores); clipa por coluna e devolve
        # ndarray para o próximo passo (o RobustScaler foi ajustado sobre arrays).
        X = X.copy()
        for col in X.columns:
            lo, hi = self.bounds[col]
            X[col] = X[col].clip(lo, hi)
        return X.to_numpy()


# despacho dos passos temporais (nome no pipeline.json -> função)
_STEPS_TEMPORAIS = {
    "remove_sensor_errors": remove_sensor_errors,
    "filter_running": filter_running,
    "remove_transients": remove_transients,
    "resample": resample,
    "ffill": ffill,
    "moving_average": moving_average,
    "select_features": select_features,
}


def preprocessar(bundle_dir, df: pd.DataFrame) -> pd.DataFrame:
    bundle_dir = Path(bundle_dir)
    steps = json.loads((bundle_dir / "pipeline.json").read_text())
    bounds = {c: tuple(v) for c, v in json.loads((bundle_dir / "clip_bounds.json").read_text()).items()}
    with (bundle_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)  # RobustScaler já ajustado no treino

    # Pipeline sklearn para o bloco numérico (montado a partir das peças do bundle)
    prep = Pipeline([("clip", Clipper(bounds)), ("scale", scaler)])

    # Guarda: este módulo aplica clip+normalize JUNTOS, ao final, via `prep`. Isso só é
    # correto se forem os dois últimos passos do pipeline.json. Se um dia houver um passo
    # depois deles, a ordem real não seria respeitada -> falha explícita em vez de silêncio.
    nomes = [s["step"] for s in steps]
    if "clip" in nomes or "normalize" in nomes:
        if nomes[-2:] != ["clip", "normalize"]:
            raise ValueError(
                "Este módulo assume 'clip' e 'normalize' como os dois últimos passos do "
                f"pipeline.json (aplicados juntos no sklearn.Pipeline). Ordem recebida: {nomes}"
            )

    for s in steps:
        nome = s["step"]
        params = {k: v for k, v in s.items() if k != "step"}
        if nome in _STEPS_TEMPORAIS:
            df = _STEPS_TEMPORAIS[nome](df, **params)
        elif nome in ("clip", "normalize"):
            continue  # aplicados juntos pelo Pipeline `prep` abaixo
        else:
            raise ValueError(f"passo de preprocessing desconhecido: {nome}")

    if len(df) == 0:
        raise ValueError(
            "nenhuma linha restou após os filtros (bomba desligada em todo o período, "
            "ou janela sem dados válidos). Verifique o CSV de entrada."
        )

    vals = prep.transform(df)  # só transform: o scaler já vem ajustado
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


# ════════════════════════════════════════════════════════════════════════════
# 3) CARREGAR MODELO
# ════════════════════════════════════════════════════════════════════════════
def carregar_modelo(bundle_dir):
    bundle_dir = Path(bundle_dir)
    arch = json.loads((bundle_dir / "model_arch.json").read_text())
    tipo = arch.get("model_type", "vae")
    if tipo == "vae":
        model = VAE(arch["input_dim"], arch["encoding_layers"], arch["latent_dim"])
    elif tipo == "dense":
        model = DenseAutoencoder(arch["input_dim"], arch["encoding_layers"])
    else:
        raise NotImplementedError(
            f"model_type '{tipo}' não suportado neste módulo (só 'vae' e 'dense'). "
            "Para OCSVM/IsolationForest/LOF, carregue model.pkl via pickle.load."
        )
    # weights_only=True: carregamos apenas os pesos (state_dict), sem executar pickle
    # arbitrário — mais seguro e comunica a intenção.
    state = torch.load(bundle_dir / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# 4) PREVER
# ════════════════════════════════════════════════════════════════════════════
def prever(bundle_dir, model, df_proc: pd.DataFrame) -> pd.DataFrame:
    """Erro de reconstrução (MSE) por instante -> severity (normal/atencao/alarme)."""
    bundle_dir = Path(bundle_dir)
    alarm = json.loads((bundle_dir / "alarm.json").read_text())
    threshold = float(alarm["threshold"])              # nível de ALARME
    atencao = alarm.get("threshold_attention")         # nível de ATENÇÃO (< alarme)
    debounce = int(alarm.get("debounce_consecutive", 1))

    x = torch.tensor(df_proc.values, dtype=torch.float32)
    with torch.no_grad():
        recon, _ = model(x)
        erro = F.mse_loss(recon, x, reduction="none").mean(dim=1).numpy()

    out = pd.DataFrame({"reconstruction_error": erro}, index=df_proc.index)
    acima = out["reconstruction_error"] > threshold

    # debounce: exige `debounce` pontos consecutivos acima do limite
    if debounce > 1:
        cont = acima.astype(int).rolling(debounce, min_periods=debounce).sum()
        acima = (cont >= debounce).fillna(False)
    out["is_anomaly"] = acima

    sev = pd.Series("normal", index=out.index, dtype=object)
    if atencao is not None:
        sev[out["reconstruction_error"] > float(atencao)] = "atencao"
    sev[out["is_anomaly"]] = "alarme"
    out["severity"] = sev
    return out
'''


# ════════════════════════════════════════════════════════════════════════════
# SCRIPT FINO POR EQUIPAMENTO — tokens @@...@@ trocados por equipamento
# ════════════════════════════════════════════════════════════════════════════
SCRIPT_TEMPLATE = r'''"""
SIMPred — Inferência de anomalias do @@EQUIP@@.

Script FINO: só aponta o bundle deste equipamento e executa os 4 passos, todos
implementados no módulo compartilhado `simpred_inference.py` (na pasta Transpetro/).
Não depende de nenhuma biblioteca interna nossa.

    python3 @@SCRIPTNAME@@
"""
import sys
from pathlib import Path

# torna o módulo compartilhado (Transpetro/simpred_inference.py) importável
_RAIZ = Path(__file__).resolve().parents[2]   # .../deploy_v2/Transpetro
sys.path.insert(0, str(_RAIZ))
import simpred_inference as si

# ── Bundle e dados deste equipamento (tudo relativo a esta pasta) ────────────
EQUIP = "@@EQUIP@@"
EQUIP_DIR = Path(__file__).resolve().parent.parent      # .../Transpetro/@@EQUIP@@
BUNDLE_DIR = EQUIP_DIR / "modelos" / "@@BUNDLE@@"
DADOS_DIR = EQUIP_DIR / "dados"
CSV_NOME = "@@CSV@@"


def main():
    print(f"[{EQUIP}] 1/4 carregando dados...")
    csv = si.achar_csv(DADOS_DIR, CSV_NOME)
    df = si.carregar_dados(csv)
    print(f"        {len(df)} linhas brutas")

    print(f"[{EQUIP}] 2/4 pre-processando...")
    df_proc = si.preprocessar(BUNDLE_DIR, df)
    print(f"        {len(df_proc)} instantes, {df_proc.shape[1]} features")

    print(f"[{EQUIP}] 3/4 carregando modelo...")
    model = si.carregar_modelo(BUNDLE_DIR)

    print(f"[{EQUIP}] 4/4 inferindo...")
    res = si.prever(BUNDLE_DIR, model, df_proc)

    # ── Resumo (sem despejar todos os alarmes na tela) ──
    n_alarme = int((res["severity"] == "alarme").sum())
    n_atencao = int((res["severity"] == "atencao").sum())
    alarmes = res[res["is_anomaly"]]
    print(f"\n{EQUIP}: {len(res)} instantes | {n_atencao} atenção | {n_alarme} alarme")
    if len(alarmes):
        print(f"        1º alarme: {alarmes.index.min()}  |  último: {alarmes.index.max()}")

    # Salva o resultado completo (para inspeção / integração)
    nome_csv = EQUIP.lower().replace("-", "").replace(".", "") + "_inferencia.csv"
    saida = Path(__file__).parent / nome_csv
    res.to_csv(saida)
    print(f"        resultado salvo em: {saida.name}")


if __name__ == "__main__":
    main()
'''


def convert(equip: str, bundle: str):
    src_bundle = SRC / equip / "modelos" / bundle
    if not src_bundle.exists():
        raise FileNotFoundError(f"bundle de origem não encontrado: {src_bundle}")
    out_equip = OUT / equip
    out_bundle = out_equip / "modelos" / bundle
    out_bundle.mkdir(parents=True, exist_ok=True)
    (out_equip / "scripts").mkdir(parents=True, exist_ok=True)

    # 1) modelo: módulo inteiro -> state_dict + arch (tipo vem do alarm.json)
    model_type = json.loads((src_bundle / "alarm.json").read_text()).get("model_type", "vae")
    model = torch.load(src_bundle / "model.pt", map_location="cpu", weights_only=False)
    model.eval()
    arch = infer_arch(model, model_type)
    torch.save(model.state_dict(), out_bundle / "model_state.pt")
    (out_bundle / "model_arch.json").write_text(json.dumps(arch, indent=2))

    # 2) preprocessing -> scaler.pkl (sklearn puro) + clip_bounds.json
    with (src_bundle / "preprocessing.pkl").open("rb") as f:
        art = pickle.load(f)
    with (out_bundle / "scaler.pkl").open("wb") as f:
        pickle.dump(art.scaler, f)
    (out_bundle / "clip_bounds.json").write_text(
        json.dumps({c: list(v) for c, v in (art.clip_bounds or {}).items()}, indent=2)
    )

    # 3) copia jsons + dados brutos
    shutil.copy2(src_bundle / "alarm.json", out_bundle / "alarm.json")
    shutil.copy2(src_bundle / "pipeline.json", out_bundle / "pipeline.json")
    src_dados = SRC / equip / "dados"
    for csv in src_dados.rglob("*.csv"):
        dst = out_equip / "dados" / csv.relative_to(src_dados)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv, dst)
    data_name = next(src_dados.rglob("*.csv")).name

    # 4) script fino
    equip_low = equip.lower().replace("-", "").replace(".", "")
    script_name = f"{equip_low}_exemplo.py"
    script = (SCRIPT_TEMPLATE
              .replace("@@EQUIP@@", equip)
              .replace("@@BUNDLE@@", bundle)
              .replace("@@CSV@@", data_name)
              .replace("@@SCRIPTNAME@@", script_name))
    (out_equip / "scripts" / script_name).write_text(script)
    lat = f"->{arch['latent_dim']}" if "latent_dim" in arch else ""
    print(f"OK {equip}: {model_type} arch={arch['input_dim']}->{arch['encoding_layers']}{lat}"
          f" | features={len(art.clip_bounds)} | dados={data_name}")


def main():
    ap = argparse.ArgumentParser(description="Gera o pacote de inferência deploy_v2/.")
    ap.add_argument("--equip", choices=list(EQUIPS), help="gera só este equipamento")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "simpred_inference.py").write_text(MODULE_SOURCE)
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    print(f"módulo   -> {OUT / 'simpred_inference.py'}")
    print(f"reqs     -> {OUT / 'requirements.txt'}")
    if not (OUT / "README.md").exists():
        print("aviso: README.md não existe (mantido à mão) — crie-o manualmente.")

    alvos = {args.equip: EQUIPS[args.equip]} if args.equip else EQUIPS
    for eq, bd in alvos.items():
        convert(eq, bd)
    print(f"\npronto: {OUT}")


if __name__ == "__main__":
    main()
