import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TABLE = "ml_dataset"


def get_engine():
    load_dotenv(PROJECT_ROOT / ".env")
    url = (
        f"mysql+pymysql://{os.environ['MYSQL_USER']}:{os.environ['MYSQL_PASSWORD']}"
        f"@{os.environ['MYSQL_HOST']}:{os.environ['MYSQL_PORT']}/{os.environ['MYSQL_DATABASE']}"
        f"?charset=utf8mb4"
    )
    return create_engine(url)


def main():
    engine = get_engine()

    with engine.connect() as conn:
        tables = [r[0] for r in conn.execute(text("SHOW TABLES"))]
        print("Tabele u bazi:", tables)

        print(f"\nŠema tabele {TABLE}:")
        schema = pd.read_sql(text(f"DESCRIBE {TABLE}"), conn)
        print(schema.to_string(index=False))

        df = pd.read_sql(text(f"SELECT * FROM {TABLE}"), conn)

    print(f"\nBroj redova: {len(df)}")
    print(f"Kolone: {list(df.columns)}")
    print("\nPrvih 10 redova:")
    print(df.head(10).to_string())

    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= 50:
            print(f"\nRaspodela vrednosti za kolonu '{col}' ({n_unique} jedinstvenih):")
            print(df[col].value_counts().to_string())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{TABLE}.parquet"
    df.to_parquet(out_path, index=False)
    df.to_csv(RAW_DIR / f"{TABLE}.csv", index=False)
    print(f"\nSačuvano: {out_path}")


if __name__ == "__main__":
    main()