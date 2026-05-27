"""
Trend Intelligence for Digital Marketing
Pipeline completo de extraccion, transformacion, analisis profundo y visualizacion.
Fuente: Google Trends (pytrends)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.extract import fetch_interest_over_time, fetch_interest_by_region
from src.transform import (
    melt_trends,
    add_time_features,
    compute_moving_average,
    compute_volatility,
    compute_period_comparison,
    compute_brand_correlation,
    compute_market_share,
)
from src.analyze import (
    compute_average_interest,
    compute_growth,
    compute_peaks,
    simulate_marketing_kpis,
    compute_monthly_summary,
    compute_growth_rate,
    compute_trend_direction,
    compute_seasonality,
    compute_brand_momentum,
    compute_market_dominance,
    compute_competitive_landscape,
    generate_insights,
)
from src.visualize import (
    plot_trends_over_time,
    plot_individual_brands,
    plot_market_share,
    plot_growth_comparison,
    plot_dominance,
    plot_trend_arrows,
    plot_momentum,
    plot_seasonality,
    plot_monthly_heatmap,
    plot_correlation,
    plot_top_regions,
    plot_kpis_dashboard,
    plot_executive_summary,
)
from src.export import save_csv


def main():
    print("=" * 70)
    print("  TREND INTELLIGENCE FOR DIGITAL MARKETING")
    print("  Analisis Profundo - Fintech en Colombia")
    print("  Fuente: Google Trends")
    print("=" * 70)

    # ==================================================================
    # 1. EXTRACCION
    # ==================================================================
    print("\n[1/7] EXTRACCION DE DATOS\n")

    print("  [Google Trends] Extrayendo interes por marca (12 meses)...")
    trends_df = fetch_interest_over_time()
    print(f"  OK: {len(trends_df)} registros semanales obtenidos\n")
    save_csv(trends_df, "google_trends_raw.csv", index=True)

    # ==================================================================
    # 2. TRANSFORMACION
    # ==================================================================
    print("\n[2/7] TRANSFORMACION DE DATOS\n")

    trends_long = melt_trends(trends_df)
    print(f"  Formato largo: {trends_long.shape[0]} filas")
    save_csv(trends_long, "trends_long.csv")

    trends_long_full = add_time_features(trends_long)
    print("  Caracteristicas temporales agregadas: Anio, Mes, Semana, Trimestre")

    ma4 = compute_moving_average(trends_df, window=4)
    volatility = compute_volatility(trends_df)
    print("  Media movil (4 semanas) y volatilidad calculadas")

    market_share = compute_market_share(trends_df)
    print("\n  Share de busqueda:")
    print(market_share.to_string(index=False))
    save_csv(market_share, "market_share.csv")

    period_comp = compute_period_comparison(trends_df)
    print("\n  Comparacion Q1 vs Q4:")
    print(period_comp.to_string(index=False))
    save_csv(period_comp, "period_comparison.csv")

    # ==================================================================
    # 3. ANALISIS DESCRIPTIVO
    # ==================================================================
    print("\n[3/7] ANALISIS DESCRIPTIVO\n")

    print("  >> Promedio de interes por marca:")
    avg = compute_average_interest(trends_long)
    print(avg.to_string())
    save_csv(avg.reset_index(), "average_interest.csv")

    print("\n  >> Crecimiento (primer vs ultimo valor):")
    growth = compute_growth(trends_df)
    print(growth.to_string(index=False))
    save_csv(growth, "growth_metrics.csv")

    print("\n  >> Picos, valles y amplitud:")
    peaks = compute_peaks(trends_df)
    print(peaks.to_string(index=False))
    save_csv(peaks, "peak_metrics.csv")

    print("\n  >> Resumen mensual:")
    monthly = compute_monthly_summary(trends_df)
    print(monthly.head(15))
    save_csv(monthly, "monthly_summary.csv")

    print("\n  >> Tasa de crecimiento semanal:")
    growth_rate = compute_growth_rate(trends_df)
    print(growth_rate.to_string(index=False))
    save_csv(growth_rate, "growth_rate.csv")

    # ==================================================================
    # 4. ANALISIS DE TENDENCIA
    # ==================================================================
    print("\n[4/7] ANALISIS DE TENDENCIA\n")

    print("  >> Direccion de tendencia (regresion lineal):")
    trend_dir = compute_trend_direction(trends_df)
    print(trend_dir.to_string(index=False))
    save_csv(trend_dir, "trend_direction.csv")

    print("\n  >> Estacionalidad:")
    seasonal = compute_seasonality(trends_df)
    print(seasonal.to_string(index=False))
    save_csv(seasonal, "seasonality.csv")

    print("\n  >> Momentum (reciente vs historico):")
    momentum = compute_brand_momentum(trends_df)
    print(momentum.to_string(index=False))
    save_csv(momentum, "momentum.csv")

    print("\n  >> Dominancia de mercado:")
    dominance = compute_market_dominance(trends_df)
    print(dominance.to_string(index=False))
    save_csv(dominance, "market_dominance.csv")

    print("\n  >> Correlacion entre marcas:")
    corr = compute_brand_correlation(trends_df)
    print(corr.to_string())
    save_csv(corr, "brand_correlation.csv")

    # ==================================================================
    # 5. ANALISIS COMPETITIVO
    # ==================================================================
    print("\n[5/7] ANALISIS COMPETITIVO\n")

    competitive = compute_competitive_landscape(trends_df)
    print(f"  Marca lider en busqueda: {competitive['marca_lider']}")
    print(f"  Mayor momentum: {competitive['mayor_momentum']}")
    print(f"  Marcas crecientes: {', '.join(competitive['crecientes']) if competitive['crecientes'] else 'Ninguna'}")
    print(f"  Marcas decrecientes: {', '.join(competitive['decrecientes']) if competitive['decrecientes'] else 'Ninguna'}")

    # KPIs simulados
    print("\n  >> Simulacion de KPIs de marketing:")
    kpi_df = simulate_marketing_kpis()
    save_csv(kpi_df, "marketing_kpis.csv")
    kpi_summary = kpi_df.groupby("Marca").agg({
        "CTR": "mean",
        "CPC": "mean",
        "ROI": "mean",
        "Conversions": "sum",
    }).round(2)
    print(kpi_summary.to_string())
    save_csv(kpi_summary, "kpi_summary.csv")

    # ==================================================================
    # 6. INSIGHTS
    # ==================================================================
    print("\n[6/7] GENERANDO INSIGHTS\n")

    insights = generate_insights(trends_df, growth, peaks, trend_dir, momentum, dominance)
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")

    # Datos regionales
    print("\n  >> Datos por region...")
    try:
        regions = fetch_interest_by_region()
        save_csv(regions, "google_trends_regions.csv")
        print(f"  OK: {len(regions)} registros regionales obtenidos")
    except Exception as e:
        print(f"  No se pudieron obtener datos regionales: {e}")
        regions = None

    # ==================================================================
    # 7. VISUALIZACION
    # ==================================================================
    print("\n[7/7] GENERANDO VISUALIZACIONES (13 graficos)\n")

    plt.style.use("ggplot")

    print("  [1/13] Tendencia temporal...")
    plot_trends_over_time(trends_df)

    print("  [2/13] Marcas individuales con volatilidad...")
    plot_individual_brands(trends_df)

    print("  [3/13] Share de busqueda (pie)...")
    plot_market_share(market_share)

    print("  [4/13] Comparacion de crecimiento...")
    plot_growth_comparison(growth)

    print("  [5/13] Dominancia de mercado...")
    plot_dominance(dominance)

    print("  [6/13] Direccion de tendencia...")
    plot_trend_arrows(trend_dir)

    print("  [7/13] Momentum...")
    plot_momentum(momentum)

    print("  [8/13] Estacionalidad...")
    plot_seasonality(seasonal, trends_df)

    print("  [9/13] Heatmap mensual...")
    plot_monthly_heatmap(trends_df)

    print("  [10/13] Correlacion entre marcas...")
    plot_correlation(corr)

    if regions is not None and not regions.empty:
        print("  [11/13] Top regiones...")
        plot_top_regions(regions)
    else:
        print("  [11/13] Top regiones... (omitido)")

    print("  [12/13] Dashboard de KPIs...")
    plot_kpis_dashboard(kpi_df)

    print("  [13/13] Resumen ejecutivo...")
    plot_executive_summary(growth, dominance, momentum, trend_dir)

    # ==================================================================
    # RESUMEN FINAL
    # ==================================================================
    print("\n" + "=" * 70)
    print("  RESUMEN DEL ANALISIS")
    print("=" * 70)
    print(f"\n  Marcas analizadas: {', '.join(BRANDS)}")
    print(f"  Periodo: {trends_df.index[0].strftime('%Y-%m-%d')} a {trends_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Registros totales: {len(trends_df)} semanas")
    print(f"  Archivos CSV generados: 12")
    print(f"  Graficos generados: 13 (en carpeta images/)")
    print(f"\n  LIDER DE MERCADO: {competitive['marca_lider']}")
    print(f"  MAYOR MOMENTUM: {competitive['mayor_momentum']}")
    print(f"\n  Pipeline completado exitosamente.")
    print("=" * 70)


if __name__ == "__main__":
    from src.config import BRANDS
    main()
