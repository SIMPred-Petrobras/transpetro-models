"""
Outras arquiteturas no modelo de anomalia do B-8802B (out/2026), com a mesma bateria de fe_exploracao_b8802b.py.

O VAE olha um instante de cada vez. Aqui:
  pca            PCA com 2 ou 3 componentes (reconstrução linear) — linha de base: se empatar, o VAE é desnecessário
  dense          autoencoder denso com gargalo de 2 (menor que a entrada, para não copiar a entrada)
  dense_janela   autoencoder denso sobre a janela das últimas 2 h achatada (24 leituras × entradas)
  lstm           autoencoder LSTM sobre a janela das últimas 2 h
Nos de janela, o score no instante t é o erro médio da janela que termina em t; a janela não atravessa paradas (no
início de cada trecho de operação, repete a primeira leitura). Entradas: "vib_press" (vibração + pressões) e "base"
(os 5 sensores de produção). Os VAEs de referência estão em results/fe_exploracao (mesma bateria).

    python scripts/arquiteturas_b8802b.py                      # local
    python scripts/arquiteturas_b8802b.py --remote             # worker do ClearML
    python scripts/arquiteturas_b8802b.py --baixar <task id>   # traz o resultado para --out
"""
import argparse, os, shutil, sys, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_exploracao_b8802b as fx  # noqa: E402
from transpetro_modelos.models.autoencoder import DenseAutoencoder, LSTMAutoencoder  # noqa: E402
from transpetro_modelos.training.train import train_autoencoder  # noqa: E402

L = 24                    # janela de 2 h (24 leituras de 5 min)
SEEDS = (0, 1, 2)
ENTRADAS = ["vib_press", "base"]


def janelas(X: pd.DataFrame) -> np.ndarray:
    """(n, L, entradas): para cada instante, as L últimas leituras do mesmo trecho de operação."""
    v = X.to_numpy(dtype="float32")
    out = np.empty((len(v), L, v.shape[1]), dtype="float32")
    seg = fx.segmentos(X.index)
    for g in np.unique(seg):
        pos = np.flatnonzero(seg == g)
        bloco = np.concatenate([np.repeat(v[pos[:1]], L - 1, axis=0), v[pos]])
        out[pos] = sliding_window_view(bloco, (L, v.shape[1]))[:, 0]
    return out


def erro_rede(m, Z: np.ndarray) -> np.ndarray:
    m.eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(Z), 4096):
            x = torch.tensor(Z[i:i + 4096])
            r = m(x)[0]
            res.append(((r - x) ** 2).reshape(len(x), -1).mean(dim=1).numpy())
    return np.concatenate(res)


def treinar(m, Zfit: np.ndarray, Zva: np.ndarray, seed: int, lr: float = 1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = lambda Z, sh: DataLoader(TensorDataset(torch.tensor(Z)), batch_size=256, shuffle=sh)
    return train_autoencoder(m, dl(Zfit, True), dl(Zva, False), epochs=60, learning_rate=lr, weight_decay=1e-5, patience=10)


def modelos(n_in: int):
    """(nome, seed, construtor, usa janela, achatar)."""
    yield from ((f"pca{k}", 0, ("pca", k), False, False) for k in (2, 3))
    for s in SEEDS:
        yield "dense", s, lambda: DenseAutoencoder(n_in, [32, 16, 2]), False, False
        yield "dense_janela", s, lambda: DenseAutoencoder(L * n_in, [64, 32, 8]), True, True
        yield "lstm", s, lambda: LSTMAutoencoder(n_in, hidden_dim=32, num_layers=1, seq_len=L), True, False


def rodar(out: Path, entradas: list[str]):
    torch.set_num_threads(4)
    ctx = fx.contexto()
    arq = out / "avaliacao.csv"
    linhas = pd.read_csv(arq).to_dict("records") if arq.exists() else []
    feitos = {(r["variante"], r["cand"]) for r in linhas}
    for var in entradas:
        F = fx.features(ctx.pre, var)
        tr = F[(F.index >= fx.TS) & (F.index < fx.TE)]
        n_val = int(len(tr) * 0.15)
        esc = fx.Escala().fit(tr.iloc[:-n_val])
        X = esc(F)
        fit_idx, va_idx = X.index.get_indexer(tr.index[:-n_val]), X.index.get_indexer(tr.index[-n_val:])
        W = janelas(X)
        for nome, seed, cons, usa_janela, achatar in modelos(X.shape[1]):
            cand = f"{nome}_s{seed}"
            if (f"{var}·{nome}", cand) in feitos:
                continue
            t0 = time.time()
            if isinstance(cons, tuple):              # PCA
                pca = PCA(n_components=cons[1]).fit(X.iloc[fit_idx])
                def pontuar(p, pca=pca):
                    Xp = esc(fx.features(p, var))
                    rec = pca.inverse_transform(pca.transform(Xp))
                    return pd.Series(((Xp.to_numpy() - rec) ** 2).mean(axis=1), index=Xp.index)
            else:
                forma = (lambda Z: Z.reshape(len(Z), -1)) if achatar else (lambda Z: Z)
                Z = forma(W) if usa_janela else X.to_numpy(dtype="float32")
                Zfit = Z[fit_idx][::3] if usa_janela else Z[fit_idx]   # janelas vizinhas quase iguais: 1 em 3
                m = treinar(cons(), Zfit, Z[va_idx], seed)
                torch.save(m.state_dict(), out / "cands" / f"{var}__{cand}.pt")
                def pontuar(p, m=m, forma=forma, usa_janela=usa_janela):
                    Xp = esc(fx.features(p, var))
                    Zp = forma(janelas(Xp)) if usa_janela else Xp.to_numpy(dtype="float32")
                    return pd.Series(erro_rede(m, Zp), index=Xp.index)
            linhas += fx.avaliar(ctx, f"{var}·{nome}", cand, X.shape[1], pontuar)
            pd.DataFrame(linhas).to_csv(arq, index=False)
            print(f"[{var}·{cand}] {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/arquiteturas")
    ap.add_argument("--entradas", nargs="*", default=ENTRADAS)
    ap.add_argument("--remote", action="store_true", help="enfileira no ClearML e sai")
    ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None, help="id da task: baixa o artifact `arquiteturas` para --out e sai")
    ap.add_argument("--clearml-task-name", default="arquiteturas-b8802b")
    a = ap.parse_args()
    if a.baixar:
        from clearml import Task
        out = fx.ROOT / a.out
        z = Task.get_task(task_id=a.baixar).artifacts["arquiteturas"].get_local_copy(extract_archive=False)
        out.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(z).extractall(out)
        print(f"resultado da task {a.baixar} extraído em {out}"); return 0

    task = None
    if a.remote or os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        Task.add_requirements("pyarrow"); Task.add_requirements("torch", package_version="")
        task = Task.init(project_name="Transpetro", task_name=a.clearml_task_name, output_uri=True, reuse_last_task_id=False)
        task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
        task.connect(vars(a))
        task.execute_remotely(queue_name=a.queue)       # daqui para baixo, só no worker
        from replay_pipeline import _baixar_csvs
        _baixar_csvs()

    out = fx.ROOT / a.out
    (out / "cands").mkdir(parents=True, exist_ok=True)
    try:
        tab = rodar(out, a.entradas)
        print(tab.groupby(["variante", "modo"]).median(numeric_only=True).round(3).to_string())
    finally:   # sobe o que tiver, mesmo se algo falhar
        if task is not None:
            if (out / "avaliacao.csv").exists():
                task.upload_artifact("avaliacao", artifact_object=out / "avaliacao.csv")
            task.upload_artifact("arquiteturas", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
