import numpy as np
import pandas as pd
from scipy import stats

from .config import BRANDS


def compute_average_interest(trends_long: pd.DataFrame) -> pd.Series:
    return (
        trends_long
        .groupby("Marca")["Interes"]
        .mean()
        .sort_values(ascending=False)
    )


def compute_growth(trends_wide: pd.DataFrame) -> pd.DataFrame:
    records = []
    for brand in trends_wide.columns:
        values = trends_wide[brand].dropna()
        if len(values) < 2:
            records.append({"Marca": brand, "Crecimiento(%)": np.nan})
            continue
        initial = values.iloc[0]
        final = values.iloc[-1]
        if initial == 0:
            growth = np.nan
        else:
            growth = ((final - initial) / initial) * 100
        records.append({"Marca": brand, "Crecimiento(%)": round(growth, 2)})
    return pd.DataFrame(records)


def compute_peaks(trends_wide: pd.DataFrame) -> pd.DataFrame:
    records = []
    for brand in trends_wide.columns:
        values = trends_wide[brand].dropna()
        peak_val = values.max()
        peak_date = values.idxmax()
        valley_val = values.min()
        valley_date = values.idxmin()
        amplitude = peak_val - valley_val
        records.append({
            "Marca": brand,
            "Pico_Max": peak_val,
            "Fecha_Pico": peak_date,
            "Valle_Min": valley_val,
            "Fecha_Valle": valley_date,
            "Amplitud": amplitude,
        })
    return pd.DataFrame(records)


def simulate_marketing_kpis(
    brands: list[str] | None = None,
    seed: int = 42,
    n_weeks: int = 12,
) -> pd.DataFrame:
    if brands is None:
        brands = BRANDS

    rng = np.random.default_rng(seed)

    rows = []
    for marca in brands:
        impressions = rng.integers(50_000, 200_000, size=n_weeks)
        ctr = rng.uniform(0.5, 3.0, size=n_weeks)
        clicks = (impressions * ctr / 100).astype(int)
        cpc = rng.uniform(500, 3000, size=n_weeks)
        spend = (clicks * cpc).astype(int)
        conversions = rng.integers(50, 500, size=n_weeks) * (ctr / ctr.mean())
        conversions = conversions.astype(int)
        revenue = (conversions * rng.integers(10_000, 50_000, size=n_weeks)).astype(int)

        for w in range(n_weeks):
            rows.append({
                "Marca": marca,
                "Semana": w + 1,
                "Impressions": impressions[w],
                "Clicks": clicks[w],
                "CTR": round(ctr[w], 2),
                "Spend": spend[w],
                "CPC": round(cpc[w], 0),
                "Conversions": conversions[w],
                "Revenue": revenue[w],
            })

    kpi_df = pd.DataFrame(rows)
    kpi_df["Conversion_Rate"] = round(
        (kpi_df["Conversions"] / kpi_df["Clicks"]) * 100, 2
    )
    kpi_df["ROI"] = round(
        ((kpi_df["Revenue"] - kpi_df["Spend"]) / kpi_df["Spend"]) * 100, 2
    )
    return kpi_df


def compute_monthly_summary(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Resumen mensual: promedio, min, max por marca y mes."""
    df = trends_wide.copy()
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("ME").agg(["mean", "min", "max", "std"])
    records = []
    for brand in df.columns:
        for date in monthly.index:
            records.append({
                "Marca": brand,
                "Fecha": date,
                "Promedio": round(monthly[(brand, "mean")][date], 2),
                "Minimo": round(monthly[(brand, "min")][date], 2),
                "Maximo": round(monthly[(brand, "max")][date], 2),
                "Desviacion": round(monthly[(brand, "std")][date], 2)
                              if not np.isnan(monthly[(brand, "std")][date]) else 0,
            })
    return pd.DataFrame(records)


def compute_growth_rate(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Calcula tasa de crecimiento semanal por marca."""
    records = []
    for brand in trends_wide.columns:
        values = trends_wide[brand].dropna()
        if len(values) < 2:
            continue
        pct_change = values.pct_change().dropna() * 100
        records.append({
            "Marca": brand,
            "Crecimiento_Semanal_Promedio(%)": round(pct_change.mean(), 2),
            "Crecimiento_Max(%)": round(pct_change.max(), 2),
            "Crecimiento_Min(%)": round(pct_change.min(), 2),
            "Semanas_Crecimiento": int((pct_change > 0).sum()),
            "Semanas_Decrecimiento": int((pct_change < 0).sum()),
        })
    return pd.DataFrame(records)


def compute_trend_direction(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Determina la direccion de la tendencia usando regresion lineal."""
    records = []
    for brand in trends_wide.columns:
        values = trends_wide[brand].dropna().values
        if len(values) < 3:
            continue
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        r_squared = r_value ** 2

        if slope > 0.5:
            direccion = "Creciente"
        elif slope < -0.5:
            direccion = "Decreciente"
        else:
            direccion = "Estable"

        records.append({
            "Marca": brand,
            "Pendiente": round(slope, 4),
            "R_Cuadrado": round(r_squared, 4),
            "P_Valor": round(p_value, 6),
            "Direccion": direccion,
            "Significativo": "Si" if p_value < 0.05 else "No",
        })
    return pd.DataFrame(records)


def compute_seasonality(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Analisis de estacionalidad: patron promedio por mes."""
    df = trends_wide.copy()
    df.index = pd.to_datetime(df.index)
    df["Mes"] = df.index.month

    seasonal = df.groupby("Mes").mean()
    records = []
    for brand in df.columns[:-1]:
        brand_data = seasonal[brand]
        mes_max = brand_data.idxmax()
        mes_min = brand_data.idxmin()
        amplitude = brand_data.max() - brand_data.min()
        cv = (brand_data.std() / brand_data.mean() * 100) if brand_data.mean() > 0 else 0
        records.append({
            "Marca": brand,
            "Mes_Max": mes_max,
            "Mes_Min": mes_min,
            "Amplitud_Estacional": round(amplitude, 2),
            "Coef_Variacion(%)": round(cv, 2),
        })
    return pd.DataFrame(records)


def compute_brand_momentum(trends_wide: pd.DataFrame, recent_weeks: int = 6) -> pd.DataFrame:
    """Calcula momentum: comparacion reciente vs historico."""
    records = []
    for brand in trends_wide.columns:
        values = trends_wide[brand].dropna()
        if len(values) < recent_weeks + 4:
            continue
        recent = values.iloc[-recent_weeks:].mean()
        historical = values.iloc[:-recent_weeks].mean()
        momentum = ((recent - historical) / historical * 100) if historical > 0 else 0

        if momentum > 10:
            estado = "Acelerando"
        elif momentum < -10:
            estado = "Desacelerando"
        else:
            estado = "Estable"

        records.append({
            "Marca": brand,
            "Promedio_Reciente": round(recent, 2),
            "Promedio_Historico": round(historical, 2),
            "Momentum(%)": round(momentum, 2),
            "Estado": estado,
        })
    return pd.DataFrame(records)


def compute_market_dominance(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Calcula dominancia de mercado: semanas lider por marca."""
    records = []
    for brand in trends_wide.columns:
        leader_count = 0
        total_weeks = len(trends_wide)
        for idx in trends_wide.index:
            row = trends_wide.loc[idx]
            if row.idxmax() == brand:
                leader_count += 1
        dominance = (leader_count / total_weeks * 100) if total_weeks > 0 else 0
        records.append({
            "Marca": brand,
            "Semanas_Lider": leader_count,
            "Dominancia(%)": round(dominance, 2),
        })
    return pd.DataFrame(records).sort_values("Dominancia(%)", ascending=False)


def compute_competitive_landscape(trends_wide: pd.DataFrame) -> dict:
    """Analisis competitivo completo."""
    dominance = compute_market_dominance(trends_wide)
    momentum = compute_brand_momentum(trends_wide)
    trend = compute_trend_direction(trends_wide)

    top_dominant = dominance.iloc[0] if not dominance.empty else None
    top_momentum = momentum.sort_values("Momentum(%)", ascending=False).iloc[0] if not momentum.empty else None
    fastest_growing = trend[trend["Direccion"] == "Creciente"].sort_values("Pendiente", ascending=False)
    declining = trend[trend["Direccion"] == "Decreciente"]

    return {
        "dominancia": dominance,
        "momentum": momentum,
        "tendencias": trend,
        "marca_lider": top_dominant["Marca"] if top_dominant is not None else "N/A",
        "mayor_momentum": top_momentum["Marca"] if top_momentum is not None else "N/A",
        "crecientes": fastest_growing["Marca"].tolist() if not fastest_growing.empty else [],
        "decrecientes": declining["Marca"].tolist() if not declining.empty else [],
    }


def generate_insights(
    trends_wide: pd.DataFrame,
    growth: pd.DataFrame,
    peaks: pd.DataFrame,
    trend_dir: pd.DataFrame,
    momentum: pd.DataFrame,
    dominance: pd.DataFrame,
) -> list[str]:
    """Genera insights accionables basados en el analisis."""
    insights = []

    # Lider de mercado
    if not dominance.empty:
        leader = dominance.iloc[0]
        insights.append(
            f"MERCADO: {leader['Marca']} domina con {leader['Dominancia(%)']}% "
            f"de semanas como lider en busqueda."
        )

    # Crecimiento
    if not growth.empty:
        fastest = growth.sort_values("Crecimiento(%)", ascending=False).iloc[0]
        slowest = growth.sort_values("Crecimiento(%)").iloc[0]
        insights.append(
            f"CRECIMIENTO: {fastest['Marca']} lidera con {fastest['Crecimiento(%)']}% "
            f"de crecimiento. {slowest['Marca']} tiene el menor ({slowest['Crecimiento(%)']}%)."
        )

    # Momentum
    if not momentum.empty:
        accel = momentum[momentum["Estado"] == "Acelerando"]
        decel = momentum[momentum["Estado"] == "Desacelerando"]
        if not accel.empty:
            brands_accel = ", ".join(accel["Marca"].tolist())
            insights.append(f"MOMENTUM: {brands_accel} esta(n) acelerando(s) en las ultimas semanas.")
        if not decel.empty:
            brands_decel = ", ".join(decel["Marca"].tolist())
            insights.append(f"MOMENTUM: {brands_decel} esta(n) desacelerando(s).")

    # Tendencia
    if not trend_dir.empty:
        growing = trend_dir[trend_dir["Direccion"] == "Creciente"]
        declining = trend_dir[trend_dir["Direccion"] == "Decreciente"]
        if not growing.empty:
            names = ", ".join(growing["Marca"].tolist())
            insights.append(f"TENDENCIA: {names} muestra(n) tendencia creciente sostenida.")
        if not declining.empty:
            names = ", ".join(declining["Marca"].tolist())
            insights.append(f"TENDENCIA: {names} muestra(n) tendencia decreciente.")

    # Estacionalidad
    if not peaks.empty:
        peak_info = peaks[["Marca", "Fecha_Pico"]].copy()
        insights.append(
            f"PICOS: Los picos de busqueda ocurren en meses diferentes, "
            f"sugiriendo oportunidades de campañas escalonadas."
        )

    # Oportunidad
    insights.append(
        "OPORTUNIDAD: Las marcas con momentum positivo y baja dominancia "
        "tienen potencial de crecimiento rapido con inversion en campañas digitales."
    )

    return insights
