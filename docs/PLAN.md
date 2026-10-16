# Plano — Projeto de portfólio com dados abertos do ONS

## Contexto
Repositório vazio (só `.gitignore`, commit inicial). O objetivo é um projeto de portfólio com dois módulos:
**(1)** previsão horária de carga por subsistema (TFT, N-BEATS e N-HiTS contra baselines) com análise por tipo de dia, e
**(2)** detecção e diagnóstico de anomalias multivariadas (PCA com T² e SPE, DPCA, RPCA) cruzadas com eventos conhecidos.
Este plano saiu de uma inspeção real do portal (API CKAN, `dados.ons.org.br`) e de amostras dos arquivos no S3, baixadas em 14/10/2026.

**Decisões já tomadas com você:**
- Alvo da previsão: curva-carga com dummies de regime metodológico.
- Setup principal: day-ahead realista, comparado com a carga programada do ONS.
- Clima: Open-Meteo.
- MVP: subsistema SE/CO.
- MSPC em duas escalas (diária e horária).
- Treino só no Mac M4 com 16 GB (MPS).
- Código em inglês, README em inglês e português.
- **Sem** ruff, pre-commit, pandera ou GitHub Actions por enquanto.
- Incluir o **MSCVAE** (`MSCVAE.py`, seu) como detector no Módulo 2, avaliado junto com o PCA na mesma interface e com os mesmos eventos e métricas.

---

## 1. Inventário dos datasets (verificado)

Arquivos em `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/<pasta>/`. A maioria tem CSV, XLSX e **Parquet** por ano, mais um dicionário de dados em JSON e PDF. Todos são **atualizados diariamente (D-1) e podem ser revisados retroativamente**.

| Dataset (CKAN id) | Pasta S3 / arquivo | Granularidade | Período | Colunas-chave | Uso |
|---|---|---|---|---|---|
| `curva-carga` | `curva-carga-ho/CURVA_CARGA_{AAAA}.parquet` | horária, subsistema | 2000 → D-1 | `id_subsistema, nom_subsistema, din_instante, val_cargaenergiahomwmed` | **Alvo M1**, M2 |
| `balanco-energia-subsistema` | `balanco_energia_subsistema_ho/BALANCO_ENERGIA_SUBSISTEMA_{AAAA}.parquet` | horária, subsistema + SIN | 2000 → D-1 | `val_gerhidraulica, val_gertermica, val_gereolica, val_gersolar, val_carga, val_intercambio` | **Base M2** (a carga confere com curva-carga) |
| `geracao-usina-2` | `geracao_usina_2_ho/GERACAO_USINA-2_{AAAA}.parquet` (até 2021), depois `_{AAAA}_{MM}` | horária, usina | 2000 → D-1 | `nom_tipousina` ∈ {HIDROELÉTRICA, TÉRMICA, EOLIELÉTRICA, FOTOVOLTAICA, NUCLEAR}, `cod_modalidadeoperacao`, `id_estado`, `val_geracao` | separar a **nuclear** (Angra 1/2) da térmica; reconciliação |
| `ear-diario-por-subsistema` | `ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_{AAAA}.parquet` | diária | 2000 → D-1 | `ear_data, ear_max_subsistema, ear_verif_subsistema_mwmes, ear_verif_subsistema_percentual` | M2 diário |
| `ena-diario-por-subsistema` | `ena_subsistema_di/…` (Parquet só nos anos recentes; os demais em CSV) | diária | 2000 → D-1 | (a ler do dicionário) | M2 diário |
| `intercambio-nacional` | `intercambio_nacional_ho/…` (Parquet só recente) | horária | 2000 → D-1 | (a ler do dicionário) | M2 (opcional, o balanço já tem o líquido) |
| `interrupcao_carga` | `interrupcao_carga/INTERRUPCAO_CARGA.parquet` | por evento | 2007 → D-1 | `din_interrupcaocarga, id_subsistema, id_estado, val_cargainterrompida_mw, val_energianaosuprida_mwh` | **rótulos de eventos** (contém 10/11/2009, 21/03/2018 e 15/08/2023 às 08:30) |
| `carga-energia` (diária) | `carga_energia_di/…` | diária | 2000 → D-1 | — | só para consistência |
| API carga verificada | `GET https://apicarga.ons.org.br/prd/cargaverificada?dat_inicio&dat_fim&cod_areacarga` | semi-horária, **UTC**, área (SECO, S, NE, N, SIN, estados) | ~2018 → | `val_cargaglobal, val_cargammgd, val_cargaglobalsmmgd, din_atualizacao…` | validar as quebras; pesos por estado |
| API carga programada | `GET …/prd/cargaprogramada?…` | semi-horária, UTC | ~2018 → D+1 | `val_cargaglobalprogramada` | **benchmark "ONS"** |

A API tem limite de 3 meses por chamada. Ainda falta confirmar a data inicial exata (a sondagem achou dados em 05/01/2018 e nada em 12/2017).

**Problemas encontrados que a ingestão precisa tratar:**
1. **Dtype drift:** anos antigos guardam valores numéricos como string decimal (`"0E-8"`, `"5337.70000000"`) e usam string vazia para ausente. Anos recentes já vêm como `float64`.
2. **Nomenclatura:** `nom_subsistema` aparece como `SUDESTE` em alguns arquivos e `SUDESTE/CENTRO-OESTE` em outros. O `id_subsistema` aparece com espaço sobrando (`"N "`, `"S "` no EAR de 2026). Normalizar sempre pelo `id` sem espaços, para um enum `SE|S|NE|N|SIN`.
3. **Horário de verão:** `din_instante` é horário local de Brasília sem fuso. Na entrada do horário de verão falta a hora 00:00 (o dia tem 23 linhas, ex.: 08/10/2000). Na saída, a hora repetida aparece fundida, sem timestamps duplicados. O horário de verão acabou em 2019, então isso só afeta dados de 2019 para trás.
4. **Quebras metodológicas da carga:** em mar/2021 entra a geração não despachada e em **29/04/2023 entra a MMGD**, ambas mudando o nível da série.
5. **Zeros estruturais vs ausentes:** eólica e solar vêm vazias em 2000. Isso significa que a fonte não existia, não que o dado falta: preencher com 0 e manter uma flag.
6. **Revisões:** a CKAN `metadata_modified` muda todo dia e a API expõe `din_atualizacao`. Os arquivos dos anos fechados podem mudar.

### Estratégia de ingestão
- `catalog.yaml` declara cada dataset (CKAN id, padrão de arquivo, formato preferido Parquet > CSV, chave temporal, tipo de granularidade).
- O downloader resolve os recursos pela **API CKAN** (`package_show`) em vez de URLs fixas, já que os padrões mudam (anual vs mensal). Ele baixa para `data/raw/<dataset>/<arquivo>` e grava um **manifesto** (`data/raw/_manifest.parquet`: url, ETag, Last-Modified, sha256, bytes, data do download).
- Modo incremental: refaz o download do ano ou mês corrente sempre, e dos arquivos antigos só se o ETag mudar. Assim as revisões entram e ficam registradas.
- Cliente da API de carga com paginação em janelas de 3 meses, retry com backoff e cache por janela.
- Limpeza com polars em lazy mode, arquivo por arquivo: coerção numérica estrita, normalização dos subsistemas e tratamento do horário de verão. Para isso a coluna local vira `ts_local` e se cria `ts_utc` via `America/Sao_Paulo`; a hora que falta é imputada por interpolação com flag `is_imputed`. Depois vem uma validação de schema simples, sem pandera: `data/schemas.py` guarda um dicionário de colunas e dtypes esperados e uma função `validate()` que falha se uma coluna for renomeada ou sumir. A saída é gravada em `data/processed/<tabela>.parquet`.
- `geracao-usina-2` (estimativa de 1 a 2 GB) é agregada logo na ingestão por (subsistema, tipo de usina, hora). Só o agregado fica em `processed/`.
- Snapshot: cada build de `processed/` registra a data da extração, que também vai como parâmetro para o MLflow.

## 2. Estrutura de diretórios
```
ONS-data-analysis/
├── pyproject.toml / uv.lock / .python-version (3.12)
├── README.md / README.pt-BR.md / CLAUDE.md
├── configs/
│   ├── data/catalog.yaml                 # datasets ONS + Open-Meteo
│   ├── forecast/{base,seasonal_naive,lgbm,nbeats,nhits,tft}.yaml
│   └── anomaly/{base,pca_static,pca_resid,dpca,rpca,mscvae_hourly,mscvae_daily,lstm_ae}.yaml
├── resources/                            # versionado no git (pequeno, curado)
│   ├── events.csv                        # jogos da Seleção, apagões, COVID, greves…
│   └── subsystem_states.yaml             # UF→subsistema, capitais/coords para clima
├── data/{raw,interim,processed}/         # fora do git
├── src/ons_energy/
│   ├── cli.py                            # typer: ingest | build | backtest | fit-mspc | report
│   ├── config.py                         # pydantic models para os YAML
│   ├── data/{ckan.py,download.py,api_carga.py,clean.py,schemas.py,catalog.py}
│   ├── calendar/{holidays.py,events.py,segments.py}
│   ├── weather/openmeteo.py
│   ├── features/{forecast.py,mspc.py}
│   ├── forecast/{models.py,backtest.py,metrics.py,evaluate.py,interpret.py,stats_tests.py}
│   ├── anomaly/
│   │   ├── base.py                       # interface Detector: fit / score / contributions
│   │   ├── pca.py, limits.py, dpca.py, rpca.py, deseason.py, contributions.py
│   │   ├── mscvae.py                     # SEU MSCVAE.py movido para cá (só o import do logger muda)
│   │   ├── mscvae_adapter.py             # MSCVAE → interface Detector (padding, calibração, RCA)
│   │   ├── evaluate.py, lstm_ae.py
│   ├── tracking.py                       # helpers MLflow
│   ├── viz/                              # figuras padronizadas p/ README
│   └── utils/{seed.py,io.py,logging.py}
├── notebooks/                            # só EDA/relatórios; importam ons_energy
│   ├── 01_eda_load.ipynb  02_eda_generation_ear.ipynb  03_data_quality.ipynb
│   ├── 10_forecast_report.ipynb  20_mspc_report.ipynb
├── tests/{fixtures/,test_ingest.py,test_clean.py,test_calendar.py,test_features_leakage.py,
│          test_metrics.py,test_pca_stats.py,test_limits.py,test_contributions.py,
│          test_spot.py,test_mscvae_adapter.py}
├── reports/figures/                      # versionado (figuras do README)
└── mlruns/                               # fora do git
```

## 3. Stack (com as escolhas justificadas)
- **Nixtla (NeuralForecast + MLForecast + StatsForecast + utilsforecast) em vez de Darts.** As quatro bibliotecas usam o mesmo formato longo (`unique_id, ds, y`) e o mesmo `cross_validation` para todos os modelos, então os baselines e os modelos de DL passam pelo mesmo backtest, o que evita vazamento por implementações diferentes. O NeuralForecast tem TFT, NBEATS (interpretável), NBEATSx e NHITS sobre Lightning, `MQLoss` e covariáveis `futr`, `hist` e `stat`. Ele também é mais rápido para modelos globais multi-série.
  - **Spike da Fase 0 (concluído, neuralforecast 3.3.0):** o `TFT` expõe `feature_importances()` (VSN do passado, do futuro e estática), `attention_weights()` e `feature_importance_correlations()`. O NBEATS interpretável tem `decompose()`. O Darts não é necessário.
  - Ressalva: os pesos ficam em `interpretability_params` do **último batch predito**, como média do batch. Para comparar dia normal e feriado, o `interpret.py` precisa predizer separadamente as janelas de cada segmento e coletar os pesos de cada uma.
- **polars** para ETL (lazy, Parquet) e pandas só onde a biblioteca exigir. O MSCVAE já recebe e devolve pandas.
- **scikit-learn** para PCA e escalonamento. As estatísticas T², SPE, limites e contribuições são implementadas por mim, porque são o diferencial do projeto e precisam de teste. **scipy** para F, χ², KDE e o SPOT do MSCVAE.
- **typer** para a CLI, **pydantic** com YAML para as configs (Hydra é pesado demais para o escopo), **MLflow** com backend sqlite local e pytest.
- **PyTorch no M4 (MPS):** `PYTORCH_ENABLE_MPS_FALLBACK=1`, accelerator configurável (`mps|cpu`) e float32. Como a máquina é limitada, o histórico de treino do DL é configurável (padrão de 2015 em diante) e o refit é trimestral.

## 4. Pipeline de dados e features

**Tabelas processadas:**
- `load_hourly` (subsistema, `ts_local`, `ts_utc`, y, flags)
- `balance_hourly` (+ nuclear separada)
- `hydro_daily` (EAR e ENA)
- `events` (interrupções agregadas)
- `ons_programmed_load` (horária, convertida de semi-horária UTC para local)
- `weather_hourly`

**Calendário (`calendar/`):**
- Biblioteca `holidays`: feriados nacionais (`holidays.BR`) e estaduais (`subdiv=UF`). Carnaval e Corpus Christi entram pela categoria `OPTIONAL`, a confirmar na versão instalada.
- **Feriado estadual ponderado:** a soma da participação de carga de cada UF com feriado no subsistema. Os pesos vêm da API carga verificada por UF, com a população como alternativa.
- Ponte (dia útil entre um feriado e o fim de semana), véspera e pós-feriado, festas de fim de ano (24/12 a 01/01) e Carnaval (seg, ter e manhã de quarta de cinzas).
- `resources/events.csv` curado à mão: jogos da Seleção nas Copas de 2014, 2018, 2022 e **2026** (jun–jul/2026, já dentro dos dados) e talvez Copa América, com o horário do jogo. Também apagões (via `interrupcao_carga` acima de um limiar em MW), COVID (mar–jun/2020), greve dos caminhoneiros (mai/2018) e o racionamento de 2001.
- A partir disso, `segments.py` atribui a cada hora um rótulo `day_type`, usado só na avaliação, e as flags usadas como covariáveis.

**Covariáveis da previsão:**
- Futuras conhecidas: hora, dia da semana, mês e Fourier anual; flags de feriado, ponte e evento (jogo como janela de −1h a +2h); dummies de regime metodológico (`regime_2021_03`, `regime_2023_04`); horário de verão (antes de 2019).
- Históricas: a própria carga.
- Estáticas: id do subsistema.
- Lags no LightGBM: 24 a 48, 168, 336 e médias móveis respeitando o gap do day-ahead. Os lags são calculados a partir da origem e **nunca usam dados posteriores a D-1 11h**, o que é coberto por teste.

**Clima (Open-Meteo):**
- Temperatura, umidade e radiação horárias em 3 a 6 capitais por subsistema, ponderadas pelo peso de carga por UF. Daí saem também CDD e HDD.
- Dois cenários **separados e rotulados**:
  - (a) *perfect-forecast*, com a reanálise ERA5 como se fosse previsão. É um teto de desempenho e vale para todo o histórico.
  - (b) *honest*, com a **Historical Forecast API** (previsões arquivadas, ~2021 em diante, a confirmar). É o cenário reportado como principal no período de teste.

**MSPC:**
- Diário: carga, hidráulica, térmica, nuclear, eólica, solar, intercâmbio, EAR% e ENA por subsistema, ou seja, cerca de 9×4 variáveis.
- Horário: o mesmo conjunto sem EAR e ENA.

## 5. Desenho experimental

### Módulo 1 — Previsão
- **Tarefa:** a cada dia, origem às 12h de D-1 local, com dados até 11h, prevendo as 24h de D (horizonte de 13 a 36 h; a avaliação usa as horas de D).
- **Splits:**
  - treino até 31/12/2022;
  - validação para hiperparâmetros em mai–dez/2023, depois da quebra da MMGD;
  - **teste em 2024-01-01 → último dado** (inclui a Copa de 2026);
  - backtest com rolling origin diário e janela expansiva; LightGBM refeito mensalmente, DL trimestralmente.
- **Modelos:**
  - SeasonalNaive (m = 168);
  - LightGBM (MLForecast), pontual e por quantis;
  - N-BEATS (genérico e interpretável), N-HiTS e TFT (`MQLoss` com quantis 0.05, 0.1, 0.5, 0.9 e 0.95);
  - **benchmark externo: carga programada do ONS**, alinhada à mesma definição de carga do período (a confirmar na Fase 1 que a série programada e a curva-carga têm a mesma base depois de 2023).
- **Métricas:**
  - MAPE, sMAPE e RMSE;
  - MASE (escala SeasonalNaive-168 calculada no treino);
  - pinball loss média e cobertura empírica de 80% e 90% para TFT e LightGBM-quantil;
  - teste de **Diebold-Mariano** entre pares de modelos.
- **Segmentação:** métricas por `day_type` (normal, feriado nacional, feriado estadual ponderado, ponte, Carnaval, fim de ano, jogo da Seleção, evento atípico), mais por subsistema e por hora do horizonte. Segmentos raros levam IC por bootstrap e o número de dias aparece em toda tabela.
- **Interpretabilidade:**
  - TFT: importâncias da variable selection (encoder e decoder) e atenção média por horizonte, em dia normal vs feriado;
  - N-BEATS-I: decomposição em tendência e sazonalidade;
  - LightGBM: SHAP (bônus barato).
- **Ablações:** sem clima, clima perfect-forecast e clima honest; sem as dummies de regime.

### Módulo 2 — MSPC
- **Pré-processamento:** a variante principal é PCA sobre **resíduos dessazonalizados**, com dois candidatos para gerar os resíduos:
  - regressão calendário + Fourier;
  - resíduos do LightGBM do Módulo 1 para a carga, o que liga os dois módulos.
  
  Também entram padronização pela média e desvio do período de referência e remoção de variáveis com variância quase zero no período (a solar antes de 2017).
- **Período de referência "normal":** candidato principal **2022-06 a 2023-04**, depois da crise hídrica, antes da quebra da MMGD, sem apagões grandes, com no mínimo cerca de 300 dias no modelo diário. Segunda referência para o regime pós-MMGD: 2023-09 a 2024-08, excluindo o apagão.
- **Variantes comparadas:**
  1. PCA estático sobre dados brutos padronizados (mostra por que não funciona: alarme o tempo todo por causa da tendência e da sazonalidade);
  2. PCA sobre resíduos;
  3. **DPCA** com l lags escolhido por análise paralela ou autocorrelação dos scores;
  4. **RPCA** com janela móvel ou fator de esquecimento e uma zona de guarda (atualiza o modelo só com observações sem alarme).
  
  O trade-off central é que a RPCA absorve falhas lentas como a crise hídrica. Por isso a comparação é feita nas duas escalas.
- **Estatísticas e limites:**
  - k componentes por variância acumulada e validação cruzada (PRESS);
  - T² com limite F para observação nova, k(n²−1)/(n(n−k))·F(k, n−k, α);
  - SPE com limite de Box (g·χ²_h, com g = θ₂/θ₁ e h = θ₁²/θ₂), e Jackson-Mudholkar como alternativa;
  - limite empírico por KDE como verificação, porque a autocorrelação infla os falsos alarmes;
  - regra de persistência (m de n) para reduzir alarmes isolados.
- **Diagnóstico:** contribuições parciais para T² e SPE e **contribuições baseadas em reconstrução (RBC, Alcala & Qin)**, que têm menos smearing. Os heatmaps de contribuição ficam ao longo dos eventos.
- **Validação:**
  - eventos rotulados (crise hídrica de 2021, apagões de 10/11/2009, 21/03/2018 e 15/08/2023, COVID de 2020, racionamento de 2001);
  - métricas: taxa de falso alarme em períodos normais de hold-out, atraso de detecção, fração de eventos detectados e se a variável de maior contribuição faz sentido físico (ex.: EAR e térmica em 2021, carga do NE em 15/08/2023).
- **Interface comum (`anomaly/base.py`):** todo detector implementa `fit(reference: list[DataFrame])`, `score(df) -> DataFrame[ts, score(s), threshold, alarm]` e `contributions(df, windows) -> DataFrame[variable, subsystem, score, %]`. Assim PCA, DPCA, RPCA, MSCVAE e LSTM-AE passam pelo mesmo `evaluate.py`, com a mesma tabela de eventos, as mesmas métricas e o mesmo pré-processamento.

#### MSCVAE (seu algoritmo) como segundo detector principal
O que ele é: matrizes de assinatura `M = XXᵀ/w` em várias escalas, encoder CNN com núcleo VAE, transformer espaço-temporal sobre `seq_len` janelas e decoder duplo (matriz e valores brutos). O score φ combina as duas rotas, ponderadas pelas precisões aprendidas, e o limiar vem do **SPOT/EVT**. O RCA usa diagonal, fora da diagonal e valor, padronizados por mediana e MAD do período normal.

- **Integração:**
  - `MSCVAE.py` é movido para `src/ons_energy/anomaly/mscvae.py`. A única mudança é trocar `from src.utils import logger` pelo logger do pacote. O algoritmo fica intacto.
  - Tudo o que for específico do projeto vai para `mscvae_adapter.py`:
    - monta o `df_sistema` (`VARIAVEL`, `DESC`, `SISTEMA` = subsistema) a partir de um registro de variáveis;
    - alinha os timestamps;
    - converte `contribution()` para o formato comum.
- **Escalas** (as janelas do seu código são em passos de tempo):
  - horária: `window_sizes = (6, 24, 168)` h (intradiário, diário e semanal) e `seq_len` de 5 a 12;
  - diária: `(7, 30, 90)` dias.
  - Os valores ficam no YAML e são ajustados na validação.
- **Mesmos insumos do PCA:** as variantes "bruto padronizado" e "resíduos dessazonalizados", para separar o efeito do modelo do efeito do pré-processamento. O z-score interno do MSCVAE não remove sazonalidade.
- **Ajustes no adapter, sem editar o seu código:**
  1. **Warm-up:** `generate()` e `SequenceMatrixDataset` preenchem o início repetindo a primeira linha ou matriz, o que gera janelas artificiais. Ao pontuar um período, o adapter **prepende as últimas `max(w)+seq_len` linhas reais** anteriores e depois descarta essa parte. No treino, as primeiras janelas de cada período são mascaradas.
  2. **Limiar fora da amostra:** o seu `fit()` calibra o SPOT sobre os scores do próprio treino, onde o erro de reconstrução é otimista, o que infla os falsos alarmes. O adapter separa um trecho final do período normal para validação, refaz o limiar com `_pot_eval(_get_anomaly_scores(val))` e também calibra ali a baseline do RCA. As duas versões são reportadas.
  3. **Múltiplos períodos normais:** o seu `fit` aceita uma lista de DataFrames sem sequências cruzando fronteiras. Isso é ótimo para o modelo diário, que sozinho teria só ~300 janelas, pouco para uma CNN-VAE. Lá uso vários trechos normais disjuntos (ex.: 2016–2019 e 2022-06 a 2023-04) e a mesma lista para o PCA, para ficar justo.
  4. **NaN vira 0 (a média):** o adapter faz a imputação antes e registra uma flag, para dado faltante não passar como normal em silêncio.
- **Comparação cruzada de limiares:** aplicar também o SPOT aos scores de T² e SPE (além de F, χ² e Box). Assim a tabela compara modelos com o mesmo critério de limiar.
- **Comparação de diagnóstico:** o RCA do MSCVAE contra as contribuições RBC do PCA nos mesmos eventos. Pergunta: as duas apontam a mesma variável física (ex.: carga do NE e intercâmbio em 15/08/2023)?
- **Opcional:** LSTM-AE com a mesma interface. Fica em segundo plano, já que o MSCVAE cobre a comparação com deep learning.

## 6. Riscos e decisões em aberto
1. **Definição da carga programada vs curva-carga:** podem diferir em MMGD e áreas. Se não houver alinhamento limpo, a comparação com o ONS vira análise qualitativa ou fica restrita a uma janela compatível.
2. **Previsões arquivadas do Open-Meteo:** cobertura e qualidade antes de 2022 incertas. Se falharem, o cenário honest usa só o período coberto.
3. **TFT no MPS:** possível falta de memória ou operações sem suporte. Mitigação: hidden size menor, batch menor, histórico de treino a partir de 2018 para o TFT e fallback para CPU em operações específicas.
4. **DST no fall-back:** não está claro como a hora repetida foi agregada. Verificar com a API UTC em 2018–2019 e documentar.
5. **Tamanho do `geracao-usina-2`** e eventual diferença na reconciliação entre usinas somadas e o balanço. A nuclear pode estar incluída em `val_gertermica`; verificar antes de subtrair.
6. **Período de referência do MSPC** pode precisar de ajuste depois da EDA.
7. **Pontos de atenção do MSCVAE** (para decidir se você quer corrigir no seu código ou deixar como está):
   - `_pot_eval` tem um `while True` que reduz o `level` sem limite de iterações. Com scores quase constantes, pode entrar em loop infinito. O adapter vai chamar com timeout ou com limite.
   - `set_deterministic` não fixa a seed do MPS nem chama `torch.use_deterministic_algorithms`, então os scores podem variar um pouco no M4.
   - Memória: `generate()` materializa todas as matrizes (janelas × escalas × N²). No horário de ~6 anos, com N≈28 e stride 1, são cerca de 0,5 GB. Cabe nos 16 GB, mas o treino usa stride > 1 ou um período limitado.
8. **Tamanho do período normal diário** para o MSCVAE (veja o item 3 dos ajustes acima).

## 7. Roadmap
| Fase | Entregável |
|---|---|
| **0. Setup** | `uv init`, pyproject (deps em grupos `dev`, `dl`, `weather`), pytest, **CLAUDE.md**, esqueleto de `src/ons_energy` e CLI. Mover `MSCVAE.py` para `anomaly/mscvae.py` e adicionar `.DS_Store` ao `.gitignore`. Spike da interpretabilidade do TFT no NeuralForecast. |
| **1. Ingestão + qualidade** | `ons ingest` (CKAN → raw com manifesto) e `ons build` (processed com validação de schema), cobrindo curva-carga, balanço, EAR, ENA, interrupções, geração por usina e as APIs verificada e programada. Notebook `03_data_quality` com quebras, DST, faltantes e reconciliação. Testes com fixtures pequenas. |
| **2. MVP previsão SE/CO** | Calendário e eventos, backtest day-ahead, SeasonalNaive e LightGBM, métricas e segmentação, MLflow. Primeira tabela de resultados. |
| **3. DL no SE/CO** | N-HiTS, N-BEATS-I e TFT com `MQLoss` no mesmo backtest; interpretabilidade; comparação com a carga programada do ONS; teste DM. |
| **4. Escala e clima** | 4 subsistemas como modelo global, Open-Meteo nos dois cenários, ablações, relatório `10_forecast_report`. |
| **5. MVP MSPC diário** | PCA sobre resíduos com referência fixa, T², SPE e limites, contribuições, validação na crise de 2021 (SE/CO e SIN). |
| **6. MSPC completo + MSCVAE** | DPCA e RPCA. Modelo horário para apagões. **MSCVAE horário e diário pelo adapter**, comparado ao PCA, DPCA e RPCA na mesma tabela (falso alarme, atraso de detecção, eventos detectados) e no RCA contra RBC. LSTM-AE opcional. Relatório `20_mspc_report`. |
| **7. Portfólio** | README EN e PT com figuras em `reports/figures`, resumo dos resultados e como reproduzir. |

## 8. CLAUDE.md (criado na Fase 0 com este conteúdo)
```markdown
# ONS Energy — project conventions

## Stack & environment
- Python 3.12, managed with `uv` (`uv sync --all-groups`; run everything via `uv run`).
- Package: `src/ons_energy` (src layout). CLI: `uv run ons <command>` (typer).
- Hardware: Apple M4 / MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`; accelerator from config (`mps|cpu`).
- Forecasting: Nixtla (neuralforecast, mlforecast, statsforecast, utilsforecast). ETL: polars; pandas only at library boundaries.
- Tracking: MLflow (sqlite, `mlruns/`, gitignored).

## Layout rules
- Logic lives in `src/`; notebooks only for EDA/reports and must import from `ons_energy` (no business logic in notebooks).
- Every experiment = CLI command + YAML in `configs/` validated by pydantic. No hard-coded paths/params.
- `data/` is never committed. Small curated tables live in `resources/` (versioned). README figures go in `reports/figures/`.

## Data conventions
- Never assume ONS column names: resolve via CKAN `package_show` and the dataset JSON dictionary; validate against the expected-schema dicts in `data/schemas.py`. Schema drift must fail loudly.
- Canonical columns: `subsystem` ∈ {SE, S, NE, N, SIN} (normalize from stripped `id_subsistema`, never from `nom_subsistema`), `ts_local` (America/Sao_Paulo wall clock), `ts_utc`, values as float64 in MW-med.
- Old files store numbers as decimal strings and missing as ""; coerce strictly. Structural zeros (source non-existent) ≠ missing: fill 0 + flag.
- DST (pre-2019): missing spring-forward hour is imputed with `is_imputed=True`.
- Methodology breaks in load: 2021-03 (non-dispatched gen) and 2023-04-29 (MMGD). Model them with regime dummies; never silently mix.
- ONS data is revised: raw downloads are tracked in `data/raw/_manifest.parquet` (ETag, sha256, timestamp); log the extraction date to MLflow.

## Modeling rules
- No leakage: features at forecast origin use only data available at origin (day-ahead: origin D-1 12:00, data ≤ D-1 11:00). Covered by `tests/test_features_leakage.py`.
- Temporal splits only (train ≤ 2022, val 2023-05..12, test 2024+); rolling-origin backtests via `cross_validation`. No random CV.
- Observed weather as future covariate = "perfect-forecast" scenario; always label it. "Honest" scenario uses archived forecasts.
- MSPC: scaling/PCA/limits fitted on the reference period only. T²/SPE/limits/contributions are our own implementation and must be unit-tested.
- All anomaly detectors implement the `Detector` interface in `anomaly/base.py` and are evaluated by the same `anomaly/evaluate.py` (same events, same preprocessing variants).
- `anomaly/mscvae.py` is the author's MSCVAE algorithm: do NOT refactor or "fix" it without asking. Project-specific behavior (warm-up with real history, out-of-sample SPOT calibration, `df_sistema` mapping, NaN handling) lives in `anomaly/mscvae_adapter.py`.
- Seeds fixed via `ons_energy.utils.seed.set_seed` (numpy, torch, lightning, lightgbm).

## Code style
- English identifiers, docstrings and commit messages; keep ONS domain terms (EAR, ENA, MMGD, SIN) as-is.
- Type hints on public functions; match surrounding style. No linters/CI configured yet.
- Tests: `uv run pytest`; network tests marked `@pytest.mark.network` and skipped by default (`-m "not network"`); use small fixtures in `tests/fixtures/`.
```

## Verificação
- **Fase 0:** `uv sync`, `uv run pytest`, `uv run ons --help` e `uv run python -c "from ons_energy.anomaly.mscvae import MSCVAE"`.
- **Fase 1:**
  - `uv run ons ingest --datasets curva-carga --years 2000 2017 2023 2026` baixa os arquivos e gera o manifesto;
  - `uv run ons build` passa na validação de schema;
  - checagens: 4 subsistemas, 8760/8784 horas por ano (23 nos dias de entrada do horário de verão, com imputação), carga do balanço igual à curva-carga, soma das usinas próxima ao balanço por fonte.
- **Testes de estatística:** em dados gaussianos sintéticos, a taxa de alarme de T² e SPE fica perto de α; uma falha injetada numa variável aparece com contribuição máxima nessa variável; o limite de Box confere com uma simulação Monte Carlo.
- **MSCVAE:**
  - `test_spot.py`: em amostras exponenciais/GPD com quantil analítico conhecido, o limiar do SPOT fica próximo dele.
  - `test_mscvae_adapter.py`: smoke test com 2 a 3 épocas em dados sintéticos, CPU e seed fixa. Verifica o shape e o alinhamento dos timestamps (score[i] corresponde a ts[i]), que o warm-up com histórico real dá o mesmo score de pontuar a série inteira, e que uma falha injetada na variável j coloca j no topo do RCA (tolerância top-3).
- **Previsão:** o backtest reproduz o MASE do SeasonalNaive em ~1 por construção, o teste de vazamento passa e a cobertura empírica é reportada.
- **MSPC:** alarmes visíveis em 2021 (SE/CO) e em 15/08/2023 (NE, modelo horário), com a figura de contribuições do PCA (RBC) e do MSCVAE (RCA) lado a lado.

---

## 9. Status da implementação (15/10/2026)

Todas as fases (0–7) estão implementadas e estruturadas. A execução completa fica com o autor, pelos
notebooks (ver §10). Ajustes em relação ao plano original, decididos durante a implementação:

- **TFT:** fica em uma config própria (`seco_tft`, `global_tft_honest`), menor (hidden 32, input 168)
  e com refit a cada 180 dias. No M4, a versão com hidden 64 faz ~1 passo/s.
- **N-HiTS e família N-BEATS:** rodam em CPU, que no M4 é mais rápida que o MPS para eles.
- **`ONS_adj`:** carga programada do ONS ajustada de nível de forma causal, porque a carga global da
  API fica ~0,9% acima da curva-carga.
- **Clima honesto:** usa a previous-runs API do Open-Meteo (`*_previous_day1`), que cobre ~98% das
  horas de teste; as lacunas usam a reanálise de 48 h antes.
- **MSPC:**
  - Variáveis com menos de 1% da carga na referência são descartadas e a padronização tem piso de
    1 unidade.
  - O RPCA ganhou `exclude_ratio` e `accept_after`; o horário roda sem exclusão.
  - Os detectores também são comparados com FAR igualado em 5%.
- **Eventos:** uma linha por perturbação, sem duplicar os apagões curados.
- **LSTM-AE (opcional):** não implementado, porque o MSCVAE cobre a comparação com deep learning.

## 10. Simplificação: experimentos nos notebooks (15/10/2026)

Esta seção substitui o layout da §3 e as convenções de CLI/YAML das seções anteriores.

- **Linha de comando só para os dados:** `uv run ons ingest` e `uv run ons build`. Os comandos
  `backtest`, `interpret`, `fit-mspc` e `report` foram removidos.
- **Execução nos notebooks.** O código dos algoritmos continua em `src/ons_energy` e é importado:
  - `01_data_eda`: EDA e qualidade dos dados.
  - `02_forecast_baselines`: SeasonalNaive, LightGBM e ONS.
  - `03_forecast_deep`: N-HiTS, N-BEATS e TFT, mais a comparação final.
  - `04_anomaly_detection`: PCA, DPCA, RPCA e MSCVAE.
- **Gráficos e tabelas** ficam nas saídas dos notebooks, incluindo os gráficos de carga real vs
  prevista.
- **`configs/` removido.**
  - Cada experimento é um objeto pydantic (`BacktestConfig`, `MSPCConfig` via `mspc_config`)
    definido numa célula.
  - O catálogo de dados virou a constante `CATALOG` em `data/catalog.py`.
- **`scripts/`, `reports/` e `viz/report.py` removidos.** As funções de `viz/` recebem DataFrames e
  devolvem a figura.
- **MLflow mantido:** `tracking.log_backtest` e `tracking.log_mspc` são chamados pelos notebooks.
- **Cache de previsões em `data/interim/forecasts/`.** Os backtests salvos permitem comparar
  LightGBM e os modelos torch, que precisam de kernels separados (libomp), sem retreinar.


## 11. Enxugamento: só PCA e MSCVAE, sem testes (15/10/2026)

Objetivo: menos código e menos conceitos, mantendo o mesmo protocolo de avaliação.

- **Detecção de anomalias:**
  - Ficam só dois detectores, ambos na escala horária: **PCA** (T²/SPE, limites empíricos por KDE,
    contribuições RBC) e **MSCVAE**.
  - Saíram:
    - DPCA, RPCA e a escala diária;
    - os pré-processamentos `raw` e `calendar`;
    - os limites F/Box/Jackson-Mudholkar/SPOT do PCA e as contribuições CDC.
  - O pré-processamento é fixo: resíduos de linha de base semanal.
  - A crise hídrica de 2021 fica na EDA (EAR/ENA).
  - O pacote `anomaly/` tem agora `base`, `pca`, `mscvae`, `mscvae_adapter` e `experiment`, que
    absorveu a avaliação por eventos.
  - O `MSPCConfig` ficou com `name`, `detector` (`pca`|`mscvae`), `params` e defaults da referência
    horária.
- **Outras simplificações:**
  - `utils/` e `config.py` viraram um único `ons_energy/utils.py`.
  - O teste de Diebold-Mariano foi para `forecast/metrics.py`.
  - O modo de validação do `BacktestConfig` foi removido; `eval_start` é o início do teste.
  - `schemas.py` valida só os arquivos brutos.
- **Testes:** removidos a pedido do autor. A verificação é feita importando os módulos, executando o
  notebook 01 e validando as configurações dos notebooks 02–04.
- **Mantidos:** clima (Open-Meteo, cenários `perfect`/`honest`) e experimentos globais em 4
  subsistemas, como opções nos notebooks.
