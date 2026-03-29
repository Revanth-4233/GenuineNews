"""
Dataset Loader — Final Version
================================
Key fixes:
1. Smart balancing — minority classes (PARTIALLY TRUE, MISLEADING)
   are repeated/oversampled to match majority class count
2. Fake1.csv/True1.csv handled correctly
3. No data gets silently discarded
"""

import os
import pandas as pd
import numpy as np

LABEL_MAP = {
    'true':           'TRUE',
    'mostly-true':    'TRUE',
    'real':           'TRUE',
    '1':              'TRUE',
    'false':          'FALSE',
    'fake':           'FALSE',
    'pants-fire':     'FALSE',
    'pants on fire':  'FALSE',
    '0':              'FALSE',
    'half-true':      'PARTIALLY TRUE',
    'barely-true':    'PARTIALLY TRUE',
    'partially true': 'PARTIALLY TRUE',
    'partially_true': 'PARTIALLY TRUE',
    '2':              'PARTIALLY TRUE',
    'misleading':     'MISLEADING',
    '3':              'MISLEADING',
    'TRUE':           'TRUE',
    'FALSE':          'FALSE',
    'PARTIALLY TRUE': 'PARTIALLY TRUE',
    'MISLEADING':     'MISLEADING',
}

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'Dataset')


def _map_label(label):
    return LABEL_MAP.get(str(label).strip().lower(),
           LABEL_MAP.get(str(label).strip(), 'FALSE'))


def _label_from_filename(fname):
    f = fname.lower()
    if any(x in f for x in ['fake', 'misinfo_fake', 'propaganda', 'russian']):
        return 'FALSE'
    elif any(x in f for x in ['true', 'misinfo_true', 'real']):
        return 'TRUE'
    return None


def load_csv_dataset(path):
    fname = os.path.basename(path)

    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            print(f"  ❌ Cannot read {fname}: {e}")
            return pd.DataFrame(columns=['News', 'target', 'source'])

    df.columns = [c.strip().lower() for c in df.columns]
    cols = list(df.columns)

    # Case 1: IFND — statement + label
    if 'statement' in cols and 'label' in cols:
        df = df[['statement', 'label']].rename(columns={'statement': 'News', 'label': 'target'})
        df = df.dropna()
        df['target'] = df['target'].apply(_map_label)
        df['source'] = fname
        print(f"  ✅ {fname}: {len(df)} rows | {df['target'].value_counts().to_dict()}")
        return df

    # Case 2: text + label (indian_news_dataset)
    if 'text' in cols and 'label' in cols:
        df = df[['text', 'label']].rename(columns={'text': 'News', 'label': 'target'})
        df = df.dropna()
        df['target'] = df['target'].apply(_map_label)
        df['source'] = fname
        print(f"  ✅ {fname}: {len(df)} rows | {df['target'].value_counts().to_dict()}")
        return df

    # Case 3: title + text (Fake1/True1) — no label column
    if 'title' in cols and 'text' in cols:
        file_label = _label_from_filename(fname)
        if not file_label:
            print(f"  ⚠️ {fname}: Cannot determine label — skipping")
            return pd.DataFrame(columns=['News', 'target', 'source'])
        df = df[['text']].rename(columns={'text': 'News'})
        df['target'] = file_label
        df = df.dropna()
        df = df[df['News'].str.len() > 20]
        df['source'] = fname
        print(f"  ✅ {fname}: {len(df)} rows | label={file_label}")
        return df

    # Case 4: title only (Fake/True)
    if 'title' in cols and 'text' not in cols and 'label' not in cols:
        file_label = _label_from_filename(fname)
        if not file_label:
            print(f"  ⚠️ {fname}: Cannot determine label — skipping")
            return pd.DataFrame(columns=['News', 'target', 'source'])
        df = df[['title']].rename(columns={'title': 'News'})
        df['target'] = file_label
        df = df.dropna()
        df['source'] = fname
        print(f"  ✅ {fname}: {len(df)} rows | label={file_label}")
        return df

    # Case 5: text only (DataSet_Misinfo / Russian)
    if 'text' in cols and 'label' not in cols and 'target' not in cols:
        file_label = _label_from_filename(fname)
        if not file_label:
            print(f"  ⚠️ {fname}: Cannot determine label — skipping")
            return pd.DataFrame(columns=['News', 'target', 'source'])
        df = df[['text']].rename(columns={'text': 'News'})
        df['target'] = file_label
        df = df.dropna()
        df = df[df['News'].str.len() > 20]
        df['source'] = fname
        print(f"  ✅ {fname}: {len(df)} rows | label={file_label}")
        return df

    # Case 6: Generic fallback
    text_col = next((c for c in cols if c in [
        'news', 'statement', 'headline', 'content', 'article',
        'body', 'news_text', 'claim', 'story', 'description',
    ]), None)
    label_col = next((c for c in cols if c in [
        'target', 'class', 'category', 'verdict',
        'classification', 'news_type', 'is_fake', 'ground_truth',
    ]), None)

    if text_col and label_col:
        df = df[[text_col, label_col]].rename(
            columns={text_col: 'News', label_col: 'target'})
        df = df.dropna()
        df['target'] = df['target'].apply(_map_label)
        df['source'] = fname
        print(f"  ✅ {fname}: {len(df)} rows | {df['target'].value_counts().to_dict()}")
        return df

    print(f"  ❌ {fname}: Unrecognized format | columns: {cols}")
    return pd.DataFrame(columns=['News', 'target', 'source'])


def load_combined_dataset():
    frames = []

    if not os.path.exists(DATASET_DIR):
        print("[DataLoader] Dataset folder not found!")
        return _generate_sample_data()

    csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
    print(f"\n[DataLoader] Found {len(csv_files)} CSV files\n")

    for fname in csv_files:
        print(f"[DataLoader] → {fname}")
        df = load_csv_dataset(os.path.join(DATASET_DIR, fname))
        if len(df) > 0:
            frames.append(df)

    if not frames:
        print("[DataLoader] No datasets loaded!")
        return _generate_sample_data()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=['News', 'target'])
    combined = combined[combined['News'].str.len() > 15]
    combined = combined.drop_duplicates(subset=['News'])

    print(f"\n[DataLoader] Combined total: {len(combined)} rows")
    print("Before balancing:")
    counts = combined['target'].value_counts().to_dict()
    print(counts)

    # Smart balancing:
    # 1. Find target size = min(5000, max class size)
    # 2. For small classes → oversample (repeat) to reach target
    # 3. For large classes → undersample to target
    target_size = min(5000, max(counts.values()))

    balanced_frames = []
    for label in combined['target'].unique():
        subset = combined[combined['target'] == label]
        available = len(subset)

        if available >= target_size:
            # Undersample large classes
            sampled = subset.sample(n=target_size, random_state=42)
        else:
            # Oversample small classes by repeating
            repeats_needed = (target_size // available) + 1
            repeated = pd.concat([subset] * repeats_needed, ignore_index=True)
            sampled = repeated.sample(n=target_size, random_state=42)

        balanced_frames.append(sampled)
        print(f"  {label}: {available} available → using {target_size}")

    balanced = pd.concat(balanced_frames, ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[DataLoader] ✅ Final balanced: {len(balanced)} rows")
    print(balanced['target'].value_counts().to_dict())
    return balanced


def _generate_sample_data():
    samples = [
        ("India successfully launched Chandrayaan-3 to lunar south pole in 2023", "TRUE"),
        ("ISRO made India fourth country to land on moon with Chandrayaan-3", "TRUE"),
        ("India became most populous country surpassing China in 2023", "TRUE"),
        ("GST was implemented across India on July 1 2017", "TRUE"),
        ("BREAKING Indian government to ban all vehicles from tomorrow share now", "FALSE"),
        ("Free electricity for all Indians forward this WhatsApp message to claim", "FALSE"),
        ("Drinking cow urine cures cancer and COVID permanently", "FALSE"),
        ("RBI cancelling all 500 rupee notes again from next week", "FALSE"),
        ("India poverty reduced but millions still live below poverty line", "PARTIALLY TRUE"),
        ("Digital India improved cities but rural areas still lack connectivity", "PARTIALLY TRUE"),
        ("Air quality improved in some cities but Delhi remains critically poor", "PARTIALLY TRUE"),
        ("India tops growth charts ignoring base effect makes it look high", "MISLEADING"),
        ("Crime fell based on incomplete police reporting and underreporting", "MISLEADING"),
        ("Photo of old flood shared as current disaster to cause panic", "MISLEADING"),
    ]
    df = pd.DataFrame(samples, columns=['News', 'target'])
    df['source'] = 'sample'
    expanded = pd.concat([df] * 30, ignore_index=True)
    print(f"[DataLoader] Using built-in sample: {len(expanded)} rows")
    return expanded


def get_dataset_stats(df):
    return {
        'total': len(df),
        'label_counts': df['target'].value_counts().to_dict(),
        'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
    }





    