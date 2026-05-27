import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError

from .config import BRANDS, HL, TZ, GEO, TIMEFRAME


def create_client() -> TrendReq:
    return TrendReq(hl=HL, tz=TZ)


def fetch_interest_over_time(
    client: TrendReq | None = None,
) -> pd.DataFrame:
    if client is None:
        client = create_client()

    try:
        client.build_payload(
            kw_list=BRANDS,
            timeframe=TIMEFRAME,
            geo=GEO,
        )
        df = client.interest_over_time()
    except ResponseError as e:
        raise RuntimeError(f"Error al obtener datos de Google Trends: {e}") from e

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    if df.empty:
        raise ValueError("No se obtuvieron datos de Google Trends")

    return df


def fetch_interest_by_region(
    brands: list[str] | None = None,
    client: TrendReq | None = None,
) -> pd.DataFrame:
    if brands is None:
        brands = BRANDS
    if client is None:
        client = create_client()

    frames = []
    for brand in brands:
        try:
            client.build_payload(
                kw_list=[brand],
                timeframe=TIMEFRAME,
                geo=GEO,
            )
            region_df = client.interest_by_region()
            region_df = region_df.reset_index()
            region_df = region_df[["geoName", brand]].copy()
            region_df.rename(columns={brand: "Interes", "geoName": "Region"}, inplace=True)
            region_df["Marca"] = brand
            frames.append(region_df)
        except ResponseError as e:
            print(f"  Error al obtener región para {brand}: {e}")
            continue

    if not frames:
        raise ValueError("No se pudo obtener datos regionales para ninguna marca")

    result = pd.concat(frames, ignore_index=True)
    return result
