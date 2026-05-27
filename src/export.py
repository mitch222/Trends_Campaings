import os

from .config import DATA_DIR


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_csv(df, filename: str, index: bool = False) -> str:
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=index)
    print(f"  Archivo guardado: {path}")
    return path
