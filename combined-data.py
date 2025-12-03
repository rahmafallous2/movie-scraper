import pandas as pd

# Read both CSV files
df1 = pd.read_csv('cleaned_data.csv')
df2 = pd.read_csv('cleaned_data2.csv')

# Rename 'category_name' in df2 to 'category' for consistency
df2 = df2.rename(columns={'category_name': 'category'})

# Combine the two dataframes, keeping all columns
combined_df = pd.concat([df1, df2], ignore_index=True)

# Remove any duplicate rows based on all columns
combined_df = combined_df.drop_duplicates()

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)

print("CSV files combined successfully!")
print(f"Total rows in combined file: {len(combined_df)}")
print(f"Columns in combined file: {list(combined_df.columns)}")