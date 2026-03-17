from datetime import datetime
import os

import pandas as pd


PATH = "..\\data\\raw\\r4.2"

def get_own_pc_map(df):
    """Helper to determine the most frequent PC for each user."""
    return df.groupby('user')['pc'].agg(lambda x: x.mode()[0])

def process_logon(path):
    """
    process logon data to extract features such as:
    - logon_on_own_pc_normal
    - logon_on_other_pc_normal and aggregate by user + date

    parameters: path (str): the path to the directory containing the logon.csv file
    returns: DataFrame with aggregated logon features by user and date
    """
    print("processing logon.............")
    # Load the logon.csv file
    logon_path = os.path.join(path, "logon.csv")
    logon_df = pd.read_csv(logon_path, parse_dates=['date'])

    # Convert 'date' to datetime format and extract hour
    logon_df['date'] = pd.to_datetime(logon_df['date'])
    logon_df['hour'] = logon_df['date'].dt.hour
    logon_df['day_of_week'] = logon_df['date'].dt.dayofweek  # 0=Monday

    # Define working hours
    def is_off_hour(hour):
        return hour < 8 or hour > 19

    logon_df['off_hour'] = logon_df['hour'].apply(is_off_hour)

    # Assume 'user' is assigned to a specific 'pc' by default, and if not, it is considered "other pc"
    # Since we do not have a mapping of user to assigned_pc, we can simulate by assuming each user's most common PC is "own"    
    user_pc_mode = get_own_pc_map(logon_df)
    logon_df['own_pc'] = logon_df.apply(lambda r: r['pc'] == user_pc_mode[r['user']], axis=1)

    # Create more features based on the dataset description
    is_logon = logon_df['activity'] == 'Logon'
    logon_df['logon_on_own_pc_normal'] = (is_logon & logon_df['own_pc'] & ~logon_df['off_hour']).astype(int)
    logon_df['logon_on_other_pc_normal'] = (is_logon & ~logon_df['own_pc'] & ~logon_df['off_hour']).astype(int)
    logon_df['logon_on_own_pc_off_hour'] = (is_logon & logon_df['own_pc'] & logon_df['off_hour']).astype(int)
    logon_df['logon_on_other_pc_off_hour'] = (is_logon & ~logon_df['own_pc'] & logon_df['off_hour']).astype(int)

    logon_df['num_logons_off_hour'] = (is_logon & logon_df['off_hour']).astype(int)

    # Aggregate features per user per day
    logon_df['date_only'] = logon_df['date'].dt.date
    grouped = logon_df.groupby(['user', 'date_only'])

    agg_features = grouped.agg(
        num_logons=('activity', lambda x: (x == 'Logon').sum()),
        num_logoffs=('activity', lambda x: (x == 'Logoff').sum()),
        logon_on_own_pc_normal=('logon_on_own_pc_normal', 'sum'),
        logon_on_other_pc_normal=('logon_on_other_pc_normal', 'sum'),
        logon_on_own_pc_off_hour=('logon_on_own_pc_off_hour', 'sum'),
        logon_on_other_pc_off_hour=('logon_on_other_pc_off_hour', 'sum'),
        num_logons_off_hour=('num_logons_off_hour', 'sum'),
        avg_logon_hour=('hour', lambda x: x.mean() if not x.empty else 0),
        num_distinct_pcs=('pc', pd.Series.nunique),
        day_of_week=('day_of_week', 'first')
    ).reset_index()

    return agg_features


def process_device(path):
    """Process device data to extract quantity of device connects on own pc vs other pc during normal hours and off hours, and aggregate by user + date

    parameters: path (str): the path to the directory containing the device.csv file
    returns: DataFrame with aggregated device features by user and date
    """
    print("processing device.............")
    # Load the uploaded device.csv file
    device_df = pd.read_csv(os.path.join(path, "device.csv"))

    # Convert 'date' to datetime and extract hour and day of the week
    device_df['date'] = pd.to_datetime(device_df['date'])
    device_df['hour'] = device_df['date'].dt.hour

    def is_off_hour(hour):
        return hour < 8 or hour > 19

    device_df['off_hour'] = device_df['hour'].apply(is_off_hour)

    # Determine each user's "own" PC as most frequently used PC
    user_pc_mode_device = device_df.groupby('user')['pc'].agg(lambda x: x.mode()[0])
    device_df['own_pc'] = device_df.apply(lambda row: row['pc'] == user_pc_mode_device[row['user']], axis=1)

    # Filter only 'connect' events for device usage
    device_df['is_connect'] = (device_df['activity'] == 'connect')

    # Create categorized flags
    device_df['device_connects_on_own_pc_normal_hour'] = (device_df['is_connect'] & device_df['own_pc'] & (~device_df['off_hour'])).astype(int)
    device_df['device_connects_on_other_pc_normal_hour'] = (device_df['is_connect'] & (~device_df['own_pc']) & (~device_df['off_hour'])).astype(int)
    device_df['device_connects_on_own_pc_off_hour'] = (device_df['is_connect'] & device_df['own_pc'] & device_df['off_hour']).astype(int)
    device_df['device_connects_on_other_pc_off_hour'] = (device_df['is_connect'] & (~device_df['own_pc']) & device_df['off_hour']).astype(int)

    # Extract date for grouping
    device_df['date_only'] = device_df['date'].dt.date

    # Aggregate by user and date
    device_grouped = device_df.groupby(['user', 'date_only']).agg(
        device_connects_on_own_pc_normal_hour=('device_connects_on_own_pc_normal_hour', 'sum'),
        device_connects_on_other_pc_normal_hour=('device_connects_on_other_pc_normal_hour', 'sum'),
        device_connects_on_own_pc_off_hour=('device_connects_on_own_pc_off_hour', 'sum'),
        device_connects_on_other_pc_off_hour=('device_connects_on_other_pc_off_hour', 'sum')
    ).reset_index()

    return device_grouped

def process_file(path):
    """
    Process file data to extract features such as destination (own pc vs other pc), file type (document vs program), and time of access (normal hours vs off hours), and aggregate by user + date

    parameters: path (str): the path to the directory containing the file.csv file
    returns: DataFrame with aggregated file features by user and date     
    """
    print("processing file.............")

    # Load in chunks if file is large
    file_df = pd.read_csv(os.path.join(path,"file.csv"), parse_dates=['date'])

    # Extract hour and date
    file_df['hour'] = file_df['date'].dt.hour
    file_df['date_only'] = file_df['date'].dt.date

    # Define working hour flag
    file_df['off_hour'] = file_df['hour'].apply(lambda h: h < 8 or h > 19)

    # Determine "own PC" as most frequently used PC per user
    user_pc_mode = file_df.groupby('user')['pc'].agg(lambda x: x.mode()[0])
    file_df['own_pc'] = file_df.apply(lambda row: row['pc'] == user_pc_mode[row['user']], axis=1)

    # Classify file type using filename extension
    def get_file_type(filename):
        filename = str(filename).lower()
        if any(ext in filename for ext in ['.doc', '.docx', '.pdf', '.xls', '.xlsx', '.ppt']):
            return 'document'
        elif any(ext in filename for ext in ['.exe', '.dll', '.bat', '.msi']):
            return 'program'
        else:
            return 'other'

    file_df['file_type'] = file_df['filename'].apply(get_file_type)

    # Create 8 binary flag columns
    file_df['documents_copy_own_pc'] = ((file_df['file_type'] == 'document') & file_df['own_pc'] & ~file_df['off_hour']).astype(int)
    file_df['documents_copy_other_pc'] = ((file_df['file_type'] == 'document') & ~file_df['own_pc'] & ~file_df['off_hour']).astype(int)
    file_df['program_files_copy_own_pc'] = ((file_df['file_type'] == 'program') & file_df['own_pc'] & ~file_df['off_hour']).astype(int)
    file_df['program_files_copy_other_pc'] = ((file_df['file_type'] == 'program') & ~file_df['own_pc'] & ~file_df['off_hour']).astype(int)

    file_df['documents_copy_own_pc_off_hour'] = ((file_df['file_type'] == 'document') & file_df['own_pc'] & file_df['off_hour']).astype(int)
    file_df['documents_copy_other_pc_off_hour'] = ((file_df['file_type'] == 'document') & ~file_df['own_pc'] & file_df['off_hour']).astype(int)
    file_df['program_files_copy_own_pc_off_hour'] = ((file_df['file_type'] == 'program') & file_df['own_pc'] & file_df['off_hour']).astype(int)
    file_df['program_files_copy_other_pc_off_hour'] = ((file_df['file_type'] == 'program') & ~file_df['own_pc'] & file_df['off_hour']).astype(int)

    # Aggregate features by user + date
    file_agg = file_df.groupby(['user', 'date_only']).agg({
        'documents_copy_own_pc': 'sum',
        'documents_copy_other_pc': 'sum',
        'program_files_copy_own_pc': 'sum',
        'program_files_copy_other_pc': 'sum',
        'documents_copy_own_pc_off_hour': 'sum',
        'documents_copy_other_pc_off_hour': 'sum',
        'program_files_copy_own_pc_off_hour': 'sum',
        'program_files_copy_other_pc_off_hour': 'sum'
    }).reset_index()

    return file_agg

def process_http(path):
    """"Process http data to extract features such as site type (job search, hacking, neutral) and time of access (normal hours vs off hours), and aggregate by user + date
    
    parameters: path (str): the path to the directory containing the http.csv file
    returns: DataFrame with aggregated http features by user and date
    """
    print("processing http.............")


    # Load necessary columns
    df = pd.read_csv(os.path.join(path, "http.csv"), usecols=['user', 'date', 'url', 'content'], parse_dates=['date'])

    # Step 1: Add basic time columns
    df['hour'] = df['date'].dt.hour
    df['date_only'] = df['date'].dt.date
    df['off_hour'] = df['hour'].apply(lambda h: h < 8 or h > 19)

    # Step 2: Combine text fields
    df['text'] = (df['url'].fillna('') + ' ' + df['content'].fillna('')).str.lower()

    # Step 3: Define curated keyword lists

    job_keywords = [
        'job', 'career', 'resume', 'cv', 'recruit', 'linkedin', 'indeed',
        'apply', 'hiring', 'headhunter', 'interview', 'jobs', 'glassdoor', 'hire'
    ]

    hacking_keywords = [
        'hack', 'exploit', 'ransomware', 'malware', 'backdoor', 'trojan', 'keylogger',
        'cve', 'ddos', 'nmap', 'wireshark', 'darkweb', 'metasploit', 'bruteforce', 'shell', 'wikileaks'
    ]

    # Step 4: Tag each row as 'job', 'hacking', or 'neutral'
    def classify_text(text):
        if any(k in text for k in hacking_keywords):
            return 'hacking'
        elif any(k in text for k in job_keywords):
            return 'job'
        return 'neutral'

    df['site_type'] = df['text'].apply(classify_text)

    # Step 5: Flag each type by hour
    df['job_search'] = ((df['site_type'] == 'job') & ~df['off_hour']).astype(int)
    df['hacking_sites'] = ((df['site_type'] == 'hacking') & ~df['off_hour']).astype(int)
    df['neutral_sites'] = ((df['site_type'] == 'neutral') & ~df['off_hour']).astype(int)

    df['job_search_off_hour'] = ((df['site_type'] == 'job') & df['off_hour']).astype(int)
    df['hacking_sites_off_hour'] = ((df['site_type'] == 'hacking') & df['off_hour']).astype(int)
    df['neutral_sites_off_hour'] = ((df['site_type'] == 'neutral') & df['off_hour']).astype(int)

    # Step 6: Aggregate by user and date
    agg_http = df.groupby(['user', 'date_only']).agg({
        'job_search': 'sum',
        'hacking_sites': 'sum',
        'neutral_sites': 'sum',
        'job_search_off_hour': 'sum',
        'hacking_sites_off_hour': 'sum',
        'neutral_sites_off_hour': 'sum'
    }).reset_index()

    return agg_http

def process_email(path):
    """Process email data to extract features such as sender/recipient type (internal vs external), number of recipients, presence of attachments, time of sending (normal hours vs off hours), and aggregate by user + date

    parameters: path (str): the path to the directory containing the email.csv file
    returns: DataFrame with aggregated email features by user and date
    """
    print("processing email.............")


    # Load essential fields
    df = pd.read_csv(os.path.join(path, "email.csv"), parse_dates=['date'])

    # Step 1: Time features
    df['hour'] = df['date'].dt.hour
    df['off_hour'] = df['hour'].apply(lambda h: h < 8 or h > 19)
    df['date_only'] = df['date'].dt.date

    # Step 2: Define helpers for internal/external
    def is_internal(email):
        return isinstance(email, str) and '@dtaa.com' in email.lower()

    def split_emails(col):
        if pd.isna(col):
            return []
        return [e.strip().lower() for e in col.split(';') if e.strip()]

    # Process sender and recipients
    df['from_internal'] = df['from'].apply(is_internal)
    df['to_list'] = df['to'].apply(split_emails)
    df['cc_list'] = df['cc'].apply(split_emails)
    df['bcc_list'] = df['bcc'].apply(split_emails)

    # Combine all recipients
    df['all_recipients'] = df['to_list'] + df['cc_list'] + df['bcc_list']

    # Count internal and external recipients
    df['internal_recipients'] = df['all_recipients'].apply(lambda lst: sum(is_internal(e) for e in lst))
    df['external_recipients'] = df['all_recipients'].apply(lambda lst: sum(not is_internal(e) for e in lst))

    # Communication type flags
    df['int_to_int_mails'] = ((df['from_internal']) & (df['external_recipients'] == 0)).astype(int)
    df['int_to_out_mails'] = ((df['from_internal']) & (df['external_recipients'] > 0)).astype(int)
    df['out_to_int_mails'] = ((~df['from_internal']) & (df['internal_recipients'] > 0)).astype(int)
    df['out_to_out_mails'] = ((~df['from_internal']) & (df['internal_recipients'] == 0)).astype(int)

    # Other features
    df['total_emails'] = 1
    df['mails_with_attachments'] = (df['attachments'] > 0).astype(int)
    df['after_hour_mails'] = df['off_hour'].astype(int)
    df['distinct_bcc'] = df['bcc_list'].apply(lambda x: len(set(x)))

    # Group by user + date
    agg_email = df.groupby(['user', 'date_only']).agg({
        'total_emails': 'sum',
        'int_to_int_mails': 'sum',
        'int_to_out_mails': 'sum',
        'out_to_int_mails': 'sum',
        'out_to_out_mails': 'sum',
        'internal_recipients': 'sum',
        'external_recipients': 'sum',
        'distinct_bcc': 'sum',
        'mails_with_attachments': 'sum',
        'after_hour_mails': 'sum'
    }).reset_index()

    return agg_email

if __name__ == "__main__":
    logon_agg = process_logon(PATH)
    device_agg = process_device(PATH)
    files_agg = process_file(PATH)
    http_agg = process_http(PATH)
    email_agg = process_email(PATH)
    merged_df = logon_agg.merge(device_agg, on=['user', 'date_only'], how='outer') \
                    .merge(files_agg, on=['user', 'date_only'], how='outer') \
                    .merge(http_agg, on=['user', 'date_only'], how='outer') \
                    .merge(email_agg, on=['user', 'date_only'], how='outer')

    # Fill NaNs with zeros for all numerical features (safe for count-type features)
    merged_df.fillna(0, inplace=True)

    out_dir = os.path.join("..", "data", "processed")
    merged_df.to_csv(os.path.join(out_dir, "merged_extracted_feature.csv"), index=False)

    print("saved.............")



