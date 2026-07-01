# Auditoria do primeiro deploy SIMPred

Data: 24/jun/2026

## Resumo

A estrutura de deploy existe e os scripts de exemplo carregam os bundles sem erro para:

- `B-8802B`
- `B-6511502A`
- `B-4064A`

Cada equipamento tem `metadata.csv`, `dados/`, `documentos/`, `modelos/model_*/` e `scripts/*-exemplo.py`.
Os bundles tambem contem os arquivos essenciais: `model.pt`, `preprocessing.pkl`, `pipeline.json` e
`alarm.json`.

## Resultado da validacao local

| Equipamento | Bundle atual | Tipo real | Normal FP | Pre-falha | Total alertas | Observacao |
|---|---:|---|---:|---:|---:|---|
| B-8802B | `model_2022-05-15_2022-07-21_VAE` | VAE | 0,05% | 44,10% | 736 / 10.792 | OK tecnico para primeiro deploy |
| B-6511502A | `model_2022-05-17_2023-05-17_VAE` | VAE | 0,16% | 14,77% | 311 / 36.387 | OK tecnico; lead aparece em 05/mai/2023 |
| B-4064A | `model_2024-01-01_2026-03-27_VAE` | VAE | 1,00% | 10,28% | 6.151 / 11.694 | **Nao recomendado como esta para 2025+** |

Para o B-4064A, o problema aparece quando se olha a operacao posterior ao reparo:

- taxa de alarme em 2025+: **85,38%**;
- causa conhecida: mudanca de regime termico pos-reparo, ja documentada em
  `docs/analise_b4064a_regime.md`;
- conclusao: o bundle carrega, mas nao esta pronto para operacao continua no regime 2025+ sem
  re-baseline no regime saudavel atual ou sem trocar para a estrategia de producao validada no
  notebook `notebooks/b4064a/automl_b4064a_prod.ipynb`.

## Pontos de atencao antes de subir ao Drive

1. A documentacao de deploy ainda fala em arquitetura DENSE para os tres bons, mas os bundles
   atualmente empacotados sao **VAE**. Isso precisa ser alinhado com o resultado que sera entregue.
2. `B-8802B` e `B-6511502A` estao tecnicamente empacotados e executaveis. Ainda vale confirmar com o
   time o `equipment_type` do `metadata.csv` antes de publicar.
3. `B-4064A` nao deve ser publicado como detector operacional de 2025+ no estado atual; ele gera
   alarme quase permanente no regime pos-reparo.
4. O exemplo atual imprime todos os alertas e pode gerar saida muito grande. Para entrega ao time de
   engenharia, convem limitar a impressao ou salvar CSV de resultado.

## Comandos usados

```bash
timeout 25 uv run python deploy/Transpetro/B-8802B/scripts/b8802b-exemplo.py
timeout 25 uv run python deploy/Transpetro/B-6511502A/scripts/b6511502a-exemplo.py
timeout 25 uv run python deploy/Transpetro/B-4064A/scripts/b4064a-exemplo.py
```

Tambem foi feita validacao programatica com `deploy/simpred_inference.py` para calcular as taxas por
janela normal, pre-falha e pos-reparo.
