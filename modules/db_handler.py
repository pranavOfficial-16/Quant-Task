import os
from typing import Optional

import pandas as pd
import sqlite3


def get_connection(db_path: str = "db/analytics.db") -> sqlite3.Connection:
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def save_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    index_label: Optional[str] = None,
) -> None:
    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=True,
        index_label=index_label,
    )
