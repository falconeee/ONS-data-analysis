# ONS Energy — previsão de carga e detecção de anomalias multivariada no SIN

*English: [README.md](README.md)*

Projeto de portfólio de ponta a ponta com os dados abertos do **ONS** (Operador Nacional do Sistema
Elétrico):

1. **Previsão de carga.** Carga horária para o dia seguinte por subsistema (SE/CO, S, NE, N). Modelos:
   Temporal Fusion Transformer, N-BEATS (genérico e interpretável) e N-HiTS. Baselines: naive sazonal,
   LightGBM e a **carga programada do próprio ONS**. Os erros são segmentados por *dias especiais*:
   feriados, pontes, Carnaval, jogos da Seleção na Copa e apagões.
2. **Detecção de anomalias multivariada.** Variáveis: carga, geração por fonte, intercâmbio e
   armazenamento dos reservatórios. Detectores: PCA com T² de Hotelling e SPE/Q (limites F, Box,
   Jackson-Mudholkar, KDE e EVT), PCA dinâmico, PCA em janela móvel e **MSCVAE**, um VAE convolucional
   multiescala sobre matrizes de assinatura, com limiar SPOT e análise de causa raiz. Todos são
   avaliados contra eventos conhecidos: a crise hídrica de 2021 e os apagões de 15/08/2023, 21/03/2018
   e outros.

Documento de desenho: [`docs/PLAN.md`](docs/PLAN.md). Convenções: [`CLAUDE.md`](CLAUDE.md).

## Como rodar

```bash
uv sync                              # Python 3.12; grupos core + dev + dl
uv run ons ingest                    # arquivos do ONS via CKAN (manifesto ETag/sha256), API de carga, Open-Meteo
uv run ons build                     # tabelas Parquet canônicas em data/processed/ (validação estrita de schema)
uv run jupyter lab notebooks/        # depois rode os notebooks em ordem
uv run pytest                        # suíte offline (testes de rede ficam de fora)
```

`ons ingest -s ons|api|weather` baixa só uma fonte. Os modelos e detectores rodam **nos notebooks**, que
importam todos os algoritmos de `src/ons_energy`. Tabelas e gráficos, incluindo os de carga real vs
prevista, ficam nas saídas dos notebooks:

| Notebook | Conteúdo | Tempo (M4) |
|---|---|---|
| `01_data_eda.ipynb` | histórico de carga, matriz de geração, EAR/ENA, efeito de feriados, qualidade dos dados | < 1 min |
| `02_forecast_baselines.ipynb` | SeasonalNaive, LightGBM e carga programada do ONS: backtest, real vs previsto, erro por tipo de dia, DM, SHAP; ablação de clima opcional | ~15 min |
| `03_forecast_deep.ipynb` | N-HiTS, N-BEATSx, N-BEATS-I (CPU) e TFT (MPS): backtests, intervalos, interpretabilidade e **comparação final de todos os modelos** | ~1,5 h |
| `04_anomaly_detection.ipynb` | PCA / DPCA / RPCA / MSCVAE, horário e diário: cartas de controle nos apagões, contribuições, crise hídrica de 2021, FAR igualado | ~1 h com MSCVAE |

LightGBM e torch não podem dividir o mesmo processo no macOS. Por isso o notebook 02 (LightGBM) e os
notebooks 03/04 (torch) usam kernels separados. Os backtests ficam em cache em
`data/interim/forecasts/`, e é assim que o notebook 03 compara todos os modelos sem retreinar. Os
experimentos são registrados no MLflow.

## Desenho experimental (resumo)

- **Previsão.**
  - Emitida em **D-1 às 12h**, com dados até as 11h, para as 24 horas de D. É o mesmo conjunto de
    informação da carga programada do ONS.
  - Treino desde 2015, validação a partir de mai/2023 e **teste de 2024-01-01 até o último dia**
    (inclui a Copa de 2026).
  - Intervalos: quantis nativos no TFT; nos demais modelos, conformal causal com janela móvel.
  - Comparações por teste de Diebold-Mariano e segmentação por tipo de dia, com IC por bootstrap.
  - Clima em três cenários: `none`, `perfect` (reanálise) e `honest` (previsão do dia anterior).
- **MSPC.**
  - Duas escalas: diária (anomalias lentas, com EAR e ENA) e horária (eventos rápidos).
  - Pré-processamentos: bruto, resíduos de calendário e resíduos de linha de base semanal (causal).
  - Detectores: PCA, DPCA, RPCA e MSCVAE, todos pela mesma interface.
  - Contribuições parciais e **RBC**.
  - Comparação também em **FAR igualado em 5%**.

## Notas de engenharia

- **torch e LightGBM** não podem dividir o mesmo processo no macOS (há duas `libomp` e dá segfault);
  ficam em notebooks (kernels) diferentes.
- **A transição energética envelhece referências estáticas.** Variáveis negligenciáveis (< 1% da
  carga) são descartadas e a padronização tem um piso.
- **O RPCA com guarda irrestrita trava** depois de uma mudança permanente. A guarda é limitada por
  `exclude_ratio` e `accept_after`.
- **Limites paramétricos sob autocorrelação** alarmam ~10% da referência com α = 1%. KDE e SPOT
  recalibram para ~1–2%.
- **O limiar SPOT do MSCVAE** calibrado em dados normais fora da amostra difere do calibrado in-sample.
  Os dois são reportados.

Os resultados, tabelas e figuras ficam nas saídas dos notebooks.

Dados © ONS, Portal de Dados Abertos; clima © Open-Meteo (CC BY 4.0).
