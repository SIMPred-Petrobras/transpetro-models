# Geradores do deck executivo (SimPred · Transpetro)

Scripts que geram o material de apresentação. **Os decks em si (`.pptx/.pdf/.html`) não são
versionados** (são binários pesados e regeneráveis — ver `.gitignore`); versionamos só estes
geradores. As saídas caem em `deploy/` ao rodar.

## Arquivos

| Script | Produz |
|---|---|
| `gen_exec_v2.py` | `deploy/slides_executivo_transpetro_v2.html` + as figuras (PNG) em `$DECK_WORK/figs_exec_v2/` |
| `build_exec_pptx_v2.py` | `deploy/slides_executivo_transpetro_editavel_v2.pptx` (nativo, editável) reusando as figuras |
| `add_notes_v2.py` | embute as notas do apresentador no `.pptx` + gera `deploy/notas_apresentador_v2.html` |

## Ordem de execução

```bash
export DECK_WORK=/caminho/para/pasta_de_trabalho   # default: deploy/decks/_work (gitignored)
uv run python deploy/decks/gen_exec_v2.py           # 1º: figuras + HTML
uv run python deploy/decks/build_exec_pptx_v2.py    # 2º: pptx (lê as figuras)
uv run python deploy/decks/add_notes_v2.py          # 3º: notas do apresentador
# PDFs:
soffice --headless --convert-to pdf --outdir deploy deploy/slides_executivo_transpetro_editavel_v2.pptx
google-chrome --headless --no-pdf-header-footer \
  --print-to-pdf=deploy/notas_apresentador_v2.pdf deploy/notas_apresentador_v2.html
```

## Dependências de dados (não versionadas)

- **Bundles empacotados** em `deploy/Transpetro/{B-8802B,B-6511502A}/` — `gen_exec_v2.py` roda a
  inferência (`simpred_inference`) para desenhar os gráficos de resultado.
- **Assets de imagem em `$DECK_WORK/`** (não gerados por estes scripts):
  - `lara_chart_2..5.png` — recortes das páginas 2–5 do PDF de análise da Lara (slides do apêndice B-24001B).

Os gráficos dos slides **4–5** (curva detecção×FP de B-3403C/B-90001A) são gerados por `tradeoff_fig()`
a partir dos números do *sweep* impressos nos notebooks `automl_model_optimized.ipynb` (branch `Lara`) —
não usam asset externo. Números auditados; ver "Regras de honestidade" abaixo.

## Estrutura do deck (15 slides)

1. Capa · 2. Pipeline (6 etapas) · 3. Resultados (B-8802B, B-6511502A — prontos)
· **4. B-3403C (detecção validada)** · **5. B-90001A (requer validação)**
· 6. Qualidade de dados · 7. B-0302C · 8. B-4064A · 9. B-24001B · 10. Placar (8 equipamentos)
· 11. Divisória do apêndice · 12–15. Estratégias de threshold da Lara (B-24001B).

## Regras de honestidade dos números (NÃO violar)

Auditamos os notebooks da Lara antes de colocar os números no slide. Ao editar:

- **B-90001A: nunca reportar "98% @ 0,7%".** É o `best_trial` do AutoML (p95/debounce 1) e **não
  reproduz** nas células. Use o melhor do grid manual: **85% @ 1,6% FP** (p97,5 + debounce 15/20).
- **Não exibir "FP 0,00%"** como métrica: o threshold é o percentil da própria janela normal e o FP
  é medido nessa mesma janela (**in-sample**). Reporte o trade-off detecção×FP do sweep.
- **Não reportar contagens de alerta por timestamp** (ex.: "6.309 alertas") — são um único regime
  elevado contado milhares de vezes em dados interpolados, não episódios de alarme.
- Sempre marcar detecção como **validada em 1 evento histórico** e os dados como **interpolados**.
