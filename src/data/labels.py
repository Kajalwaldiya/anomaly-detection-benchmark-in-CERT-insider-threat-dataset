import os
import pandas as pd


PATH = os.path.join("..", "data", "raw", "answers")


malicious_folders = [
    os.path.join(PATH, "r4.2-1"),
    os.path.join(PATH, "r4.2-2"),
    os.path.join(PATH, "r4.2-3"),
]


def extract_malicious_labels(folder_paths):
    """Extract (user, date) pairs from all CSV files in the given folders and return a set of malicious pairs.
    parameter: folder_paths: list of folder paths to search for CSV files
    returns: A set of (user, date) tuples representing malicious activity.
    """
    malicious_set = set()

    for folder in folder_paths:
        print('Inside: ', folder)
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder, filename)
                print("reading...... ", file_path)
                try:
                    df = pd.read_csv(file_path, sep=',', header=None, names=[
                        'field1', 'id', 'timestamp', 'user', 'pc', 'event_end', 'event_data', '8','9','10','11','12'
                    ])
                except Exception as e:
                    print(f"Skipping {file_path} due to read error: {e}")
                    continue

                df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                df = df.dropna(subset=['date', 'user'])

                # Add (user, date) to malicious set
                malicious_set.update(set(zip(df['user'], df['date'])))

    return malicious_set

malicious_pairs = extract_malicious_labels(malicious_folders)

def unlabel_full_df():
    df = pd.read_csv(os.path.join("..", "data", "processed", "final_merged_dataset.csv"), parse_dates=['date_only'])

    return df

def label_full_df(malicious_set):
    unlabel_df = unlabel_full_df()
    unlabel_df['is_malicious'] = unlabel_df.apply(
        lambda row: 1 if (row['user'], row['date_only'].date()) in malicious_set else 0,
        axis=1
    )
    return unlabel_df


dataset_df = label_full_df(malicious_pairs)

dataset_df.to_csv(os.path.join("..", "data", "processed", "data_final_label.csv"), index=False)