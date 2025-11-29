import pandas as pd
import numpy as np

# load Data
df = pd.read_csv("combined_data.csv")  
print("Shape before cleaning:", df.shape)

# budget & revenue: 0 means "Unknown"
for col in ["budget", "revenue", "status","keywords"]:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna("Unknown")

# runtime: replace 0 with median
if "runtime" in df.columns:
    runtime_median = df.loc[df["runtime"] > 0, "runtime"].median()
    df.loc[df["runtime"] == 0, "runtime"] = np.nan
    df["runtime"] = df["runtime"].fillna(runtime_median)


# Remove [] and '' from text list columns
list_cols = [
    "genres",
    "director",     
    "cast",
    "crew",
    "production_companies"
]

for col in list_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace("[", "", regex=False)
        df[col] = df[col].str.replace("]", "", regex=False)
        df[col] = df[col].str.replace("'", "", regex=False)
        df[col] = df[col].str.strip()

# language code → full language name
language_map = {
    "en": "English",
    "fr": "French",
    "ko": "Korean",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "nl": "Dutch",
    "fi": "Finnish",
    "da": "Danish",
    "tl": "Tagalog",
    "hi": "Hindi",
    "kn": "Kannada",
    "te": "Telugu",
    "lv": "Latvian",
    "ta": "Tamil",
    "cs": "Czech",
    "ro": "Romanian",
    "pt": "Portuguese",
}


df["original_language"] = df["original_language"].map(language_map).fillna(df["original_language"])


# empty original_title fix them with title 
df["original_title"] = df["original_title"].replace("", np.nan)
df["original_title"] = df["original_title"].fillna(df["title"])


# check duplicate
if "id" in df.columns:
    duplicate_ids = df[df.duplicated("id", keep=False)].sort_values("id")
    print("Number of duplicated movie IDs:",
          duplicate_ids["id"].nunique())

    df = df.drop_duplicates(subset="id", keep="first")

# sanity check
print("\nMissing values per column after cleaning:")
print(df.isna().sum())

print("\nData types after cleaning:")
print(df.dtypes)

print("Final shape:", df.shape)

# save data
df.to_csv("FinalCleaned_data.csv", index=False)
