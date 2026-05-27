import pandas as pd
import numpy as np

_MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

_DIAS = {
    0: "Lunes", 1: "Martes", 2: "Miercoles",
    3: "Jueves", 4: "Viernes", 5: "Sabado", 6: "Domingo",
}


def melt_trends(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.melt(
        id_vars=date_col,
        var_name="Marca",
        value_name="Interes",
    )
    df.rename(columns={date_col: "Fecha"}, inplace=True)
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "Fecha") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Anio"] = df[date_col].dt.year
    df["Mes"] = df[date_col].dt.month
    df["Nombre_Mes"] = df[date_col].dt.month.map(_MESES)
    df["Semana"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Dia_Semana"] = df[date_col].dt.dayofweek.map(_DIAS)
    df["Trimestre"] = df[date_col].dt.quarter
    return df


def compute_moving_average(
    trends_wide: pd.DataFrame,
    window: int = 4,
) -> pd.DataFrame:
    """Calcula media movil para cada marca."""
    ma = trends_wide.rolling(window=window, min_periods=1).mean()
    ma.columns = [f"{col}_MA{window}" for col in ma.columns]
    return ma


def compute_volatility(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Calcula volatilidad (desviacion estandar rolling) por marca."""
    vol = trends_wide.rolling(window=4, min_periods=1).std()
    vol.columns = [f"{col}_Volatilidad" for col in vol.columns]
    return vol


def compute_period_comparison(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Compara periodos: primer trimestre vs ultimo trimestre."""
    records = []
    for brand in trends_wide.columns:
        values = trends_wide[brand].dropna()
        if len(values) < 8:
            continue
        q1 = values.iloc[:13].mean()
        q4 = values.iloc[-13:].mean()
        change_pct = ((q4 - q1) / q1 * 100) if q1 > 0 else 0
        records.append({
            "Marca": brand,
            "Q1_Promedio": round(q1, 2),
            "Q4_Promedio": round(q4, 2),
            "Cambio_Entre_Trimestres(%)": round(change_pct, 2),
        })
    return pd.DataFrame(records)


def compute_brand_correlation(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlacion entre marcas."""
    return trends_wide.corr().round(4)


def compute_market_share(trends_wide: pd.DataFrame) -> pd.DataFrame:
    """Calcula share de busqueda relativo por marca sobre el total."""
    totals = trends_wide.sum()
    total_general = totals.sum()
    share = (totals / total_general * 100).round(2)
    return share.sort_values(ascending=False).reset_index().rename(
        columns={"index": "Marca", 0: "Share_Busqueda(%)"}
    )
