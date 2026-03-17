import os
from glob import glob

import pandas as pd


LDAP_DIR = os.path.join("..", "data", "raw", "r4.2", "LDAP")
OUTPUT_PATH = os.path.join("..", "data", "processed", "user_categorical_metadata.csv")

CAT_COLS = ["user_id", "role", "functional_unit", "department", "team"]


def load_ldap_files(ldap_dir: str, columns: list[str]) -> pd.DataFrame:
    """Load and concatenate all LDAP CSV files from the given directory."""
    files = sorted(glob(os.path.join(ldap_dir, "*.csv")))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {ldap_dir}")

    frames = []
    for file in files:
        try:
            df = pd.read_csv(file, usecols=columns)
            df["source_file"] = os.path.basename(file)
            frames.append(df)
            print("Loaded: %s (%d rows)" % (os.path.basename(file), len(df)))
        except Exception as e:
            print("Skipping %s: %s" % (file, e))

    if not frames:
        raise ValueError("No files could be loaded successfully.")

    return pd.concat(frames, ignore_index=True)


def deduplicate_by_user(df: pd.DataFrame, user_col: str = "user_id") -> pd.DataFrame:
    """Drop nulls and keep the last record per user."""
    df = df.dropna(subset=[user_col])
    df = df.drop_duplicates(subset=[user_col], keep="last")
    return df


def to_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert specified columns to categorical dtype for memory efficiency."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category")
    return df


def save_output(df: pd.DataFrame, path: str) -> None:
    """Save the DataFrame to a CSV file."""
    df.to_csv(path, index=False)
    print("Saved %d records to: %s" % (len(df), path))


def main() -> None:
    raw_df = load_ldap_files(LDAP_DIR, columns=CAT_COLS)

    deduped_df = deduplicate_by_user(raw_df)

    categorical_cols = CAT_COLS[1:]  # All except user_id
    typed_df = to_categorical(deduped_df, columns=categorical_cols)

    output_df = typed_df[CAT_COLS]
    save_output(output_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
