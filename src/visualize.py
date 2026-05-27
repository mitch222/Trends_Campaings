import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import BRANDS, DATA_DIR

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def _ensure_images_dir():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def _save_plt(filename: str):
    _ensure_images_dir()
    path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Grafico guardado: {path}")


def _save_plotly(fig, filename: str):
    _ensure_images_dir()
    path = os.path.join(IMAGES_DIR, filename)
    fig.write_html(path)
    print(f"    Grafico guardado: {path}")


# ============================================================================
# 1. TENDENCIA TEMPORAL
# ============================================================================

def plot_trends_over_time(trends_wide: pd.DataFrame) -> None:
    """Grafico de lineas con todas las marcas y media movil."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Panel 1: Datos originales
    ax1 = axes[0]
    for brand in BRANDS:
        if brand in trends_wide.columns:
            ax1.plot(trends_wide.index, trends_wide[brand], label=brand, linewidth=1.5)
    ax1.set_title("Interes en Google Trends - Todas las Marcas", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Interes Relativo")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Media movil (suavizado)
    ax2 = axes[1]
    for brand in BRANDS:
        if brand in trends_wide.columns:
            ma = trends_wide[brand].rolling(window=4, min_periods=1).mean()
            ax2.plot(trends_wide.index, ma, label=f"{brand} (MA4)", linewidth=2)
    ax2.set_title("Media Movil (4 semanas) - Tendencia Suavizada", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Interes Relativo (Suavizado)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plt("01_trends_over_time.png")


def plot_individual_brands(trends_wide: pd.DataFrame) -> None:
    """Grafico individual por marca con volatilidad."""
    n_brands = len([b for b in BRANDS if b in trends_wide.columns])
    fig, axes = plt.subplots(n_brands, 1, figsize=(16, 4 * n_brands), sharex=True)
    if n_brands == 1:
        axes = [axes]

    for i, brand in enumerate(BRANDS):
        if brand not in trends_wide.columns:
            continue
        ax = axes[i]
        values = trends_wide[brand]

        # Datos originales
        ax.plot(trends_wide.index, values, color="steelblue", alpha=0.6, linewidth=1)

        # Media movil
        ma = values.rolling(window=4, min_periods=1).mean()
        ax.plot(trends_wide.index, ma, color="darkblue", linewidth=2, label="MA4")

        # Banda de volatilidad
        std = values.rolling(window=4, min_periods=1).std()
        upper = ma + std
        lower = ma - std
        ax.fill_between(trends_wide.index, lower, upper, alpha=0.2, color="steelblue",
                         label="Volatilidad (+-1 std)")

        ax.set_title(f"{brand}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Interes")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Fecha")
    plt.tight_layout()
    _save_plt("02_individual_brands.png")


# ============================================================================
# 2. ANALISIS DE MERCADO
# ============================================================================

def plot_market_share(market_share: pd.DataFrame) -> None:
    """Grafico de pie con share de busqueda."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(market_share)))
    wedges, texts, autotexts = ax.pie(
        market_share.iloc[:, 1],
        labels=market_share.iloc[:, 0],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    ax.set_title("Share de Busqueda por Marca", fontsize=14, fontweight="bold")
    _save_plt("03_market_share.png")


def plot_growth_comparison(growth: pd.DataFrame) -> None:
    """Grafico de barras con crecimiento por marca."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in growth["Crecimiento(%)"]]
    bars = ax.bar(growth["Marca"], growth["Crecimiento(%)"], color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Crecimiento: Primer vs Ultimo Valor", fontsize=14, fontweight="bold")
    ax.set_ylabel("Crecimiento (%)")
    ax.set_xlabel("Marca")
    for bar, val in zip(bars, growth["Crecimiento(%)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    _save_plt("04_growth_comparison.png")


def plot_dominance(dominance: pd.DataFrame) -> None:
    """Grafico de dominancia de mercado."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(dominance)))
    bars = ax.barh(dominance["Marca"], dominance["Dominancia(%)"], color=colors)
    ax.set_title("Dominancia de Mercado (% semanas como lider)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dominancia (%)")
    for bar, val in zip(bars, dominance["Dominancia(%)"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    _save_plt("05_dominance.png")


# ============================================================================
# 3. TENDENCIA Y DIRECCION
# ============================================================================

def plot_trend_arrows(trend_dir: pd.DataFrame) -> None:
    """Grafico de flechas indicando direccion de tendencia."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Creciente": "#2ecc71", "Decreciente": "#e74c3c", "Estable": "#f39c12"}
    icons = {"Creciente": "^", "Decreciente": "v", "Estable": "o"}

    for i, row in trend_dir.iterrows():
        color = colors.get(row["Direccion"], "gray")
        ax.scatter(row["Pendiente"], row["Marca"], s=200, c=color, zorder=5,
                   marker="o", edgecolors="black", linewidth=1)
        ax.annotate(
            f"{row['Direccion']}\n(R²={row['R_Cuadrado']:.2f})",
            (row["Pendiente"], row["Marca"]),
            textcoords="offset points", xytext=(10, 0),
            fontsize=9, fontweight="bold",
        )

    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Direccion de Tendencia por Marca (Regresion Lineal)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pendiente (positiva = creciente)")
    ax.grid(True, alpha=0.3, axis="x")
    _save_plt("06_trend_direction.png")


def plot_momentum(momentum: pd.DataFrame) -> None:
    """Grafico de momentum con indicadores de estado."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Acelerando": "#2ecc71", "Desacelerando": "#e74c3c", "Estable": "#f39c12"}
    bars = ax.barh(momentum["Marca"], momentum["Momentum(%)"],
                   color=[colors.get(s, "gray") for s in momentum["Estado"]])
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_title("Momentum: Reciente vs Historico", fontsize=14, fontweight="bold")
    ax.set_xlabel("Momentum (%)")
    for bar, val, state in zip(bars, momentum["Momentum(%)"], momentum["Estado"]):
        ax.text(bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 3,
                bar.get_y() + bar.get_height()/2,
                f"{val}%\n({state})", va="center", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    _save_plt("07_momentum.png")


# ============================================================================
# 4. ESTACIONALIDAD
# ============================================================================

def plot_seasonality(seasonal: pd.DataFrame, trends_wide: pd.DataFrame) -> None:
    """Grafico de estacionalidad: patron mensual promedio."""
    df = trends_wide.copy()
    df.index = pd.to_datetime(df.index)
    df["Mes"] = df.index.month
    monthly_avg = df.groupby("Mes")[BRANDS].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    for brand in BRANDS:
        if brand in monthly_avg.columns:
            ax.plot(monthly_avg.index, monthly_avg[brand], marker="o", linewidth=2, label=brand)

    ax.set_title("Patron Estacional: Interes Promedio por Mes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Interes Promedio")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_plt("08_seasonality.png")


def plot_monthly_heatmap(trends_wide: pd.DataFrame) -> None:
    """Heatmap de interes por marca y mes."""
    df = trends_wide.copy()
    df.index = pd.to_datetime(df.index)
    df["Mes"] = df.index.month
    monthly_avg = df.groupby("Mes")[BRANDS].mean()
    monthly_avg.index = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                          "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(monthly_avg.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(BRANDS)))
    ax.set_xticklabels(BRANDS, rotation=45, ha="right")
    ax.set_yticks(range(len(monthly_avg)))
    ax.set_yticklabels(monthly_avg.index)
    for i in range(len(monthly_avg)):
        for j in range(len(BRANDS)):
            val = monthly_avg.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if val > 50 else "black")
    plt.colorbar(im, ax=ax, label="Interes Promedio")
    ax.set_title("Heatmap: Interes por Marca y Mes", fontsize=14, fontweight="bold")
    _save_plt("09_monthly_heatmap.png")


# ============================================================================
# 5. CORRELACION
# ============================================================================

def plot_correlation(corr: pd.DataFrame) -> None:
    """Matriz de correlacion entre marcas."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(val) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Correlacion")
    ax.set_title("Correlacion entre Marcas", fontsize=14, fontweight="bold")
    _save_plt("10_correlation.png")


# ============================================================================
# 6. REGIONES
# ============================================================================

def plot_top_regions(regions: pd.DataFrame, top_n: int = 10) -> None:
    """Top regiones por marca."""
    brands_in_data = regions["Marca"].unique()
    n_brands = len(brands_in_data)
    fig, axes = plt.subplots(n_brands, 1, figsize=(14, 5 * n_brands))
    if n_brands == 1:
        axes = [axes]

    for i, brand in enumerate(brands_in_data):
        ax = axes[i]
        brand_data = regions[regions["Marca"] == brand].nlargest(top_n, "Interes")
        ax.barh(brand_data["Region"], brand_data["Interes"], color="steelblue")
        ax.set_title(f"Top {top_n} Regiones - {brand}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Interes")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    _save_plt("11_top_regions.png")


# ============================================================================
# 7. KPIs Y METRICAS
# ============================================================================

def plot_kpis_dashboard(kpi_df: pd.DataFrame) -> None:
    """Dashboard de KPIs simulados de marketing."""
    brands = kpi_df["Marca"].unique()
    n = len(brands)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = [axes]

    for i, brand in enumerate(brands):
        brand_kpi = kpi_df[kpi_df["Marca"] == brand]

        # CTR
        axes[i][0].plot(brand_kpi["Semana"], brand_kpi["CTR"], marker="o", color="steelblue")
        axes[i][0].set_title(f"{brand} - CTR (%)")
        axes[i][0].set_ylabel("CTR")
        axes[i][0].grid(True, alpha=0.3)

        # CPC
        axes[i][1].plot(brand_kpi["Semana"], brand_kpi["CPC"], marker="o", color="coral")
        axes[i][1].set_title(f"{brand} - CPC ($)")
        axes[i][1].set_ylabel("CPC")
        axes[i][1].grid(True, alpha=0.3)

        # ROI
        axes[i][2].plot(brand_kpi["Semana"], brand_kpi["ROI"], marker="o", color="green")
        axes[i][2].set_title(f"{brand} - ROI (%)")
        axes[i][2].set_ylabel("ROI")
        axes[i][2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plt("12_kpis_dashboard.png")


# ============================================================================
# 8. RESUMEN EJECUTIVO
# ============================================================================

def plot_executive_summary(
    growth: pd.DataFrame,
    dominance: pd.DataFrame,
    momentum: pd.DataFrame,
    trend_dir: pd.DataFrame,
) -> None:
    """Resumen ejecutivo en un solo grafico."""
    fig = plt.figure(figsize=(18, 12))

    # 1. Crecimiento
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in growth["Crecimiento(%)"]]
    ax1.barh(growth["Marca"], growth["Crecimiento(%)"], color=colors)
    ax1.set_title("Crecimiento (%)", fontweight="bold")
    ax1.axvline(x=0, color="black", linewidth=0.5)

    # 2. Dominancia
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.barh(dominance["Marca"], dominance["Dominancia(%)"], color="steelblue")
    ax2.set_title("Dominancia de Mercado (%)", fontweight="bold")

    # 3. Momentum
    ax3 = fig.add_subplot(2, 2, 3)
    colors_m = {"Acelerando": "#2ecc71", "Desacelerando": "#e74c3c", "Estable": "#f39c12"}
    ax3.barh(momentum["Marca"], momentum["Momentum(%)"],
             color=[colors_m.get(s, "gray") for s in momentum["Estado"]])
    ax3.set_title("Momentum Reciente (%)", fontweight="bold")
    ax3.axvline(x=0, color="black", linewidth=0.5)

    # 4. Tendencia
    ax4 = fig.add_subplot(2, 2, 4)
    colors_t = {"Creciente": "#2ecc71", "Decreciente": "#e74c3c", "Estable": "#f39c12"}
    ax4.scatter(trend_dir["Pendiente"], trend_dir["Marca"], s=200,
                c=[colors_t.get(d, "gray") for d in trend_dir["Direccion"]],
                edgecolors="black", zorder=5)
    ax4.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
    ax4.set_title("Direccion de Tendencia", fontweight="bold")
    ax4.set_xlabel("Pendiente")

    plt.suptitle("RESUMEN EJECUTIVO - Fintech Colombia", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_plt("13_executive_summary.png")
