# Trend Intelligence for Digital Marketing

Analisis de tendencias digitales para marketing analytics, enfocado en el sector fintech colombiano (Nu, Nequi, Daviplata, Lulo Bank).

## Estructura

```
├── src/
│   ├── config.py      # Configuracion (marcas, parametros)
│   ├── extract.py     # Extraccion desde Google Trends (pytrends)
│   ├── transform.py   # Limpieza y transformacion de datos
│   ├── analyze.py     # Metricas, crecimiento, KPIs de marketing
│   ├── visualize.py   # Graficos matplotlib y plotly
│   └── export.py      # Exportacion a CSV
├── notebooks/
│   └── 01_analisis_tendencias.ipynb
├── data/              # CSV generados
├── dashboard/         # Archivos Power BI
├── images/            # Graficos exportados
├── main.py            # Pipeline completo
├── requirements.txt
└── Trends.ipynb       # Notebook original
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

**Pipeline completo:**
```bash
python main.py
```

**Notebook (Google Colab o local):**
Abrir `notebooks/01_analisis_tendencias.ipynb`

## KPIs Calculados

- CTR = Clicks / Impressions * 100
- CPC = Spend / Clicks
- Conversion Rate = Conversions / Clicks * 100
- ROI = (Revenue - Spend) / Spend * 100
