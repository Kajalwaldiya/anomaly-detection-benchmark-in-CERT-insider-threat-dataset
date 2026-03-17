import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder



data_dir = os.path.join("..", "data", "processed")

feature_path = os.path.join(data_dir, "merged_extracted_featuer.csv")
ldap_path = os.path.join(data_dir, "user_categorical_metadata.csv")

merged_df = pd.read_csv(feature_path)
ldap_df = pd.read_csv(ldap_path)

# Load previously saved categorical metadata
ldap_df_copy = ldap_df.copy()

# Columns to label encode
categorical_cols = ['role', 'functional_unit', 'department', 'team']

# Store encoders (optional: for inverse_transform later)
label_encoders = {}

# Apply label encoding to each column
for col in categorical_cols:
    le = LabelEncoder()
    ldap_df_copy[col + '_label'] = le.fit_transform(ldap_df_copy[col].astype(str))
    label_encoders[col] = le

# Optionally drop original string columns
ldap_encoded = ldap_df_copy[['user_id'] + [col + '_label' for col in categorical_cols]]

# Save encoded version
ldap_encoded.to_csv("user_categorical_metadata_encoded.csv", index=False)


final_df = merged_df.merge(ldap_encoded, left_on='user', right_on='user_id', how='left')

final_df.to_csv(os.path.join(data_dir, "final_merged_dataset.csv"), index=False)