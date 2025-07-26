import pandas as pd
import os
import re
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

# 📁 Include all CAP rounds
DATA_DIRS = ["CAP1", "CAP2", "CAP3"]
MODEL_PATH = "trained_model.joblib"

def extract_year(filename):
    match = re.search(r'\d{4}', filename)
    return int(match.group(0)) if match else None

def load_past_data():
    df_list = []
    print("\n📊 Loading data...")

    for folder in DATA_DIRS:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"⚠️ Created folder '{folder}' — Please add cutoff CSV files.")
            continue

        all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        if not all_files:
            print(f"⚠️ No CSV files found in '{folder}'")
            continue

        for file in all_files:
            year = extract_year(file)
            if year is None:
                continue

            file_path = os.path.join(folder, file)
            try:
                df = pd.read_csv(file_path, dtype={
                    "College Name": "category",
                    "Branch Code": "category",
                    "Branch Name": "category",
                    "Category": "category",
                    "Percentile": "float32"
                })
                df["Year"] = year
                df_list.append(df)
            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e}")

    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        print(f"✅ Loaded {len(final_df)} rows.")
        return final_df
    else:
        print("❌ No valid data found across CAP folders.")
        return None

def preprocess_data(df):
    df['Composite_Key'] = (
        df['College Name'].astype(str) + "|" +
        df['Branch Name'].astype(str) + "|" +
        df['Category'].astype(str)
    )
    return df

def train_models(df):
    models = {}
    df = preprocess_data(df)

    valid_groups = df['Composite_Key'].value_counts()
    valid_groups = valid_groups[valid_groups >= 2].index

    print(f"\n⚙️ Training models...")
    for key in tqdm(valid_groups, desc="Training"):
        group = df[df['Composite_Key'] == key]
        X = group[["Year"]].values
        y = group["Percentile"].values

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        models[key] = model

    return models

def main():
    df = load_past_data()
    if df is None:
        return

    models = train_models(df)

    print("\n💾 Saving model to", MODEL_PATH, "...")
    joblib.dump(models, MODEL_PATH)
    print("✅ Training complete and model saved.")

if __name__ == "__main__":
    main()
