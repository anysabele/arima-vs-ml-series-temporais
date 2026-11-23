# Previsão de Séries Temporais: Modelos Estatísticos × Aprendizado de Máquina

Comparação de modelos estatísticos (ARIMA/SARIMA) e de aprendizado de máquina (MLP, SVR e Árvore de Decisão) na **previsão 1 passo à frente** de três séries temporais reais. 

Trabalho da disciplina de Aprendizado de Máquina II (CIn/UFPE).

## Séries analisadas

| Série | Frequência | Fonte | Ciclo principal |
|---|---|---|---|
| Consumo horário de energia (PJME) | Horária (recorte de 05/2009 a 08/2009) | [Kaggle: Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) | Diário (24 h) |
| CO₂ atmosférico em Mauna Loa | Mensal (03/1958 a 12/2025) | [NOAA GML](https://gml.noaa.gov/ccgg/trends/data.html) | Anual (12 meses) |
| Manchas solares | Anual (1700 a 2008) | `statsmodels.datasets.sunspots` | ~11 anos |

## Modelos

**Estatísticos**
- **Box & Jenkins**: identificação por ACF/PACF, comparação de candidatos por AIC/BIC e diagnóstico dos resíduos (Ljung-Box).
- **Auto-ARIMA** (`pmdarima`): seleção automática das ordens.
- Energia e manchas solares usam ARIMA; CO₂ usa SARIMA (`D = 1`, `s = 12`).

**Aprendizado de máquina** (janelas deslizantes com `L` defasagens)
- **MLP**, **SVR** (kernel RBF; linear também testado no CO₂) e **Árvore de Decisão**.
- Hiperparâmetros e número de defasagens escolhidos por grid search na validação.

## Metodologia

1. **Análise exploratória**: estatísticas descritivas, tendência linear, decomposição STL (força de tendência e sazonalidade), testes ADF e KPSS e correlogramas (ACF/PACF).
2. **Pré-processamento**: diferenciação sazonal e simples. Em energia, log + diferença de 24; em CO₂, normalização Min-Max seguida de diferenças de 12 e 1. A normalização é ajustada **apenas no treino**, para evitar vazamento de informação.
3. **Hold-out temporal**: 50% treino, 25% validação e 25% teste, sem embaralhar. Nos modelos estatísticos, treino e validação são concatenados (75% iniciais).
4. **Previsão 1 passo à frente** no teste: cada previsão usa apenas valores observados até o instante anterior.
5. **Repetições**: os modelos de ML são avaliados em 10 sementes (0 a 9), e os resultados são reportados como média ± desvio padrão.
6. **Métricas**: RMSE, MAE e MAPE (energia e CO₂), na escala original e normalizada, e RMSE e MAE (manchas solares).
7. **Referências simples** para contexto (CO₂): modelo *naive* e ΔΔ₁₂ = 0. Um modelo só "aprendeu algo" se superar essas referências.

### Configurações finais: manchas solares

| Modelo | Configuração |
|---|---|
| Box & Jenkins | ARIMA(2, 1, 2), menor AIC |
| Auto-ARIMA | ARIMA(2, 1, 3) |
| MLP | `n_lags=6`, `hidden_layer_sizes=(32, 16)`, `alpha=1e-4` |
| SVR | `n_lags=3`, `C=1`, `gamma='scale'`, `epsilon=0.01` |
| Árvore de Decisão | `n_lags=15`, `max_depth=10`, `min_samples_leaf=1` |

As tabelas comparativas completas e os gráficos (real × previsto, erro de previsão, RMSE por modelo) são gerados ao executar o código.

## Como executar

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
pip install numpy pandas matplotlib scipy statsmodels scikit-learn pmdarima kagglehub
```

O código foi desenvolvido no **Google Colab**, e o caminho mais simples é abri-lo lá. Os dados são baixados automaticamente:
- **Energia**: via `kagglehub` (pode pedir autenticação no Kaggle).
- **CO₂**: direto da NOAA, com o período fixado em 03/1958 a 12/2025 para reprodutibilidade.
- **Manchas solares**: embutida no `statsmodels`.

## Estrutura

```
.
├── time_series_analysis.py   # análise e experimentos das três séries
├── apresentação_timeseriesanalysis.pdf
└── README.md
```

## Tecnologias

Python · pandas · NumPy · statsmodels · pmdarima · scikit-learn · SciPy · Matplotlib

## Autoria

Anysabele de Paula Barbosa Santos · Maria Clara Pereira Gonçalves ·
Millena Ferreira Marçal das Neves / Inteligência Artificial, CIn/UFPE
