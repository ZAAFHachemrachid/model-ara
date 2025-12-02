import pandas as pd
import re
import emoji
from scipy import stats
import numpy as np
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings('ignore')

# ------------------------------
# 1️⃣ Load data from fake.csv and true.csv
# ------------------------------
print("="*70)
print("📊 NEWS DATA CLEANING AND BALANCING SYSTEM")
print("="*70)

try:
    # Load data
    fake_df = pd.read_csv("./DataSet/fake.csv")
    true_df = pd.read_csv("./DataSet/true.csv")
    
    print("✅ Files loaded successfully:")
    print(f"   - fake.csv: {fake_df.shape} rows × {fake_df.shape[1]} cols (Fake News)")
    print(f"   - true.csv: {true_df.shape} rows × {true_df.shape[1]} cols (True News)")
    
    # Add label column if not exists
    if 'label' not in fake_df.columns:
        fake_df['label'] = 0  # 0 for Fake News
    if 'label' not in true_df.columns:
        true_df['label'] = 1  # 1 for True News
    
    # Save original copies
    fake_original = fake_df.copy()
    true_original = true_df.copy()
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please ensure these files exist in DataSet folder:")
    print("   - ./DataSet/fake.csv")
    print("   - ./DataSet/true.csv")
    exit()

# ------------------------------
# 2️⃣ Clean Data (for each file separately)
# ------------------------------
def clean_data(df, dataset_name):
    print(f"\n{'='*70}")
    print(f"🧹 CLEANING: {dataset_name}")
    print(f"{'='*70}")
    
    df_original = df.copy()
    original_rows = len(df)
    
    # 2.1 Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    duplicates_removed = before_dup - after_dup
    print(f"✅ Removed {duplicates_removed} duplicate rows")
    print(f"   Rows: {before_dup} → {after_dup}")
    
    # 2.2 Handle missing values
    print(f"\n📋 Missing values before cleaning:")
    missing_before = df.isnull().sum()
    if any(missing_before > 0):
        print(missing_before[missing_before > 0])
    else:
        print("No missing values found")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    # 2.3 Clean text columns
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            text_columns.append(col)
    
    if text_columns:
        print(f"\n🧽 Cleaning text columns: {text_columns}")
        for col in text_columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.lower().str.strip()
            df[col] = df[col].apply(lambda x: re.sub(r"http\S+|www\S+", "", x))
            df[col] = df[col].apply(lambda x: emoji.replace_emoji(x, replace=""))
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
            df[col] = df[col].str.strip()
    
    return df, df_original

# Clean fake.csv
fake_cleaned, fake_original = clean_data(fake_df, "fake.csv (Fake News)")

# Clean true.csv
true_cleaned, true_original = clean_data(true_df, "true.csv (True News)")

# ------------------------------
# 3️⃣ Balance the Data
# ------------------------------
print(f"\n{'='*70}")
print("⚖️  BALANCING DATA BETWEEN CLASSES")
print(f"{'='*70}")

# Combine data to check balance
combined_df = pd.concat([fake_cleaned, true_cleaned], ignore_index=True)
target_col = 'label'

print("📊 Class distribution before balancing:")
class_dist = combined_df[target_col].value_counts()
print(f"   - Fake News (0): {class_dist.get(0, 0)} rows")
print(f"   - True News (1): {class_dist.get(1, 0)} rows")
print(f"   - Balance ratio: {class_dist.get(0, 0)/class_dist.get(1, 0):.2f}:1")

# Calculate target count for balancing
max_class_count = max(class_dist)
target_count = max_class_count  # Make all classes have same count

# Balance the data
fake_balanced = fake_cleaned.copy()
true_balanced = true_cleaned.copy()

# If fake has fewer rows than target, add more samples
if len(fake_balanced) < target_count:
    needed = target_count - len(fake_balanced)
    print(f"\n➕ Adding {needed} samples to Fake News...")
    
    # Duplicate existing samples (simple alternative to SMOTE)
    additional_samples = fake_balanced.sample(n=needed, replace=True, random_state=42)
    fake_balanced = pd.concat([fake_balanced, additional_samples], ignore_index=True)

# If true has fewer rows than target, add more samples
if len(true_balanced) < target_count:
    needed = target_count - len(true_balanced)
    print(f"➕ Adding {needed} samples to True News...")
    
    # Duplicate existing samples (simple alternative to SMOTE)
    additional_samples = true_balanced.sample(n=needed, replace=True, random_state=42)
    true_balanced = pd.concat([true_balanced, additional_samples], ignore_index=True)

print(f"\n✅ Balancing completed:")
print(f"   - Fake News: {len(fake_balanced)} rows")
print(f"   - True News: {len(true_balanced)} rows")

# Create combined balanced dataset
combined_balanced = pd.concat([fake_balanced, true_balanced], ignore_index=True)

# ------------------------------
# 4️⃣ Save Results
# ------------------------------
print(f"\n{'='*70}")
print("💾 SAVING RESULTS")
print(f"{'='*70}")

# Save cleaned data
fake_cleaned.to_csv("./DataSet/fake_cleaned.csv", index=False, encoding='utf-8')
true_cleaned.to_csv("./DataSet/true_cleaned.csv", index=False, encoding='utf-8')

# Save balanced data
fake_balanced.to_csv("./DataSet/fake_balanced.csv", index=False, encoding='utf-8')
true_balanced.to_csv("./DataSet/true_balanced.csv", index=False, encoding='utf-8')

# Save combined dataset
combined_balanced.to_csv("./DataSet/combined_balanced.csv", index=False, encoding='utf-8')

print("✅ Files saved successfully:")
print(f"\n📁 CLEANED FILES:")
print(f"   1. fake_cleaned.csv      - {len(fake_cleaned):6d} rows × {fake_cleaned.shape[1]} cols")
print(f"   2. true_cleaned.csv      - {len(true_cleaned):6d} rows × {true_cleaned.shape[1]} cols")

print(f"\n📁 BALANCED FILES:")
print(f"   3. fake_balanced.csv     - {len(fake_balanced):6d} rows × {fake_balanced.shape[1]} cols")
print(f"   4. true_balanced.csv     - {len(true_balanced):6d} rows × {true_balanced.shape[1]} cols")

print(f"\n📁 COMBINED FILE:")
print(f"   5. combined_balanced.csv - {len(combined_balanced):6d} rows × {combined_balanced.shape[1]} cols")

# ------------------------------
# 5️⃣ Generate Detailed Report
# ------------------------------
print(f"\n{'='*70}")
print("📈 DETAILED REPORT")
print(f"{'='*70}")

# Create report DataFrame
report_data = {
    "Dataset": ["Fake News", "True News", "TOTAL"],
    "Original Rows": [
        len(fake_original),
        len(true_original),
        len(fake_original) + len(true_original)
    ],
    "Cleaned Rows": [
        len(fake_cleaned),
        len(true_cleaned),
        len(fake_cleaned) + len(true_cleaned)
    ],
    "Balanced Rows": [
        len(fake_balanced),
        len(true_balanced),
        len(fake_balanced) + len(true_balanced)
    ],
    "Duplicates Removed": [
        len(fake_original) - len(fake_cleaned),
        len(true_original) - len(true_cleaned),
        (len(fake_original) - len(fake_cleaned)) + (len(true_original) - len(true_cleaned))
    ]
}

report_df = pd.DataFrame(report_data)
print("\n" + report_df.to_string(index=False))

# Generate text report
report_text = f"""
{'='*70}
DATA CLEANING AND BALANCING REPORT
{'='*70}

📊 CLEANING STATISTICS:
{'='*30}
• Fake News Dataset:
  - Original rows: {len(fake_original):,}
  - Cleaned rows:  {len(fake_cleaned):,}
  - Duplicates removed: {len(fake_original) - len(fake_cleaned):,}
  - Rows kept: {len(fake_cleaned)/len(fake_original)*100:.1f}%

• True News Dataset:
  - Original rows: {len(true_original):,}
  - Cleaned rows:  {len(true_cleaned):,}
  - Duplicates removed: {len(true_original) - len(true_cleaned):,}
  - Rows kept: {len(true_cleaned)/len(true_original)*100:.1f}%⚖️ BALANCING STATISTICS:
{'='*30}
• Before Balancing:
  - Fake News: {len(fake_cleaned):,} rows
  - True News: {len(true_cleaned):,} rows
  - Balance ratio: {len(fake_cleaned)/len(true_cleaned) if len(true_cleaned) > 0 else 'N/A'}:1

• After Balancing:
  - Fake News: {len(fake_balanced):,} rows
  - True News: {len(true_balanced):,} rows
  - Balance ratio: {len(fake_balanced)/len(true_balanced) if len(true_balanced) > 0 else 'N/A'}:1
  - Total increase: {((len(fake_balanced)+len(true_balanced))/(len(fake_cleaned)+len(true_cleaned))*100-100):.1f}%

💾 OUTPUT FILES SUMMARY:
{'='*30}
1. fake_cleaned.csv      - {len(fake_cleaned):,} rows × {fake_cleaned.shape[1]} cols
2. true_cleaned.csv      - {len(true_cleaned):,} rows × {true_cleaned.shape[1]} cols
3. fake_balanced.csv     - {len(fake_balanced):,} rows × {fake_balanced.shape[1]} cols
4. true_balanced.csv     - {len(true_balanced):,} rows × {true_balanced.shape[1]} cols
5. combined_balanced.csv - {len(combined_balanced):,} rows × {combined_balanced.shape[1]} cols

📈 OVERALL STATISTICS:
{'='*30}
• Total original data: {len(fake_original) + len(true_original):,} rows
• Total cleaned data:  {len(fake_cleaned) + len(true_cleaned):,} rows
• Total balanced data: {len(fake_balanced) + len(true_balanced):,} rows
• Overall duplicates removed: {(len(fake_original) - len(fake_cleaned)) + (len(true_original) - len(true_cleaned)):,} rows
• Data preservation rate: {(len(fake_cleaned) + len(true_cleaned))/(len(fake_original) + len(true_original))*100:.1f}%

{'='*70}
✅ PROCESS COMPLETED SUCCESSFULLY!
{'='*70}
"""

print(report_text)

# Save report to file
with open("./DataSet/cleaning_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("📄 Report saved to: ./DataSet/cleaning_report.txt")
print(f"\n{'='*70}")
print("🎉 DATA CLEANING AND BALANCING COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")

# Display final summary
print(f"\n📋 FINAL SUMMARY:")
print(f"{'─'*40}")
print(f"Original Files:")
print(f"  • fake.csv: {len(fake_original):,} rows")
print(f"  • true.csv: {len(true_original):,} rows")
print(f"  • Total:    {len(fake_original) + len(true_original):,} rows")

print(f"\nCleaned Files:")
print(f"  • fake_cleaned.csv: {len(fake_cleaned):,} rows")
print(f"  • true_cleaned.csv: {len(true_cleaned):,} rows")
print(f"  • Total cleaned:    {len(fake_cleaned) + len(true_cleaned):,} rows")

print(f"\nBalanced Files:")
print(f"  • fake_balanced.csv: {len(fake_balanced):,} rows")
print(f"  • true_balanced.csv: {len(true_balanced):,} rows")
print(f"  • Combined:          {len(combined_balanced):,} rows")

print(f"\n📁 All files saved in: ./DataSet/")