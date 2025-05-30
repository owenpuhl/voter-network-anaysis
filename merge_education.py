import pandas as pd

# List to store dataframes
dfs = []

# Process each Excel file for Congress 108-118
for congress in range(108, 119):
    filename = f"congressbio/education{congress}.xlsx"

    # Read the Excel file
    df = pd.read_excel(filename)

    # Add a column for Congress number
    df['congress'] = congress

    # Append to our list
    dfs.append(df)

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Save as CSV
merged_df.to_csv("congressbio/merged_congress_education.csv", index=False)

print(f"Successfully merged {len(dfs)} files into merged_congress_education.csv")