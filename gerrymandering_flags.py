import pandas as pd

# Load the data
df = pd.read_csv("StatesAndCyclesData_production-20240301a.csv")

# Create dummy variables for Institution type
institution_dummies = pd.get_dummies(df['Institution'], prefix='is')
df = pd.concat([df, institution_dummies], axis=1)

print("\nInstitution dummy counts:")
for col in institution_dummies.columns:
    print(f"{col}: {institution_dummies[col].sum()}")


def categorize_control_status(row):
    if pd.isna(row['Party Control']) or row['Party Control'] == 'n/a':
        return 'Unknown'
    elif row['Party Control'] == 'Split':
        return 'Split'
    elif row['Party Control'] in ['Democratic', 'Republican']:
        return 'Partisan'
    else:
        return 'Other'


def analyze_gerrymandering_data(df):
    # Ensure boolean columns are properly set
    institution_columns = [col for col in df.columns if col.startswith('is_')]

    for col in institution_columns:
        df[col] = df[col].fillna(0).astype(int)

    # Print institution counts
    print("\nInstitution dummy counts:")
    for col in institution_columns:
        print(f"{col}: {df[col].sum()}")

    # Properly categorize control status
    df['control_status_cat'] = df.apply(categorize_control_status, axis=1)

    # Print control status distribution
    print("\nControl Status Distribution:")
    status_counts = df['control_status_cat'].value_counts()
    print(status_counts)

    # Calculate split vs unified control
    df['split_control'] = (df['control_status_cat'] == 'Split').astype(int)
    df['partisan_control'] = (df['control_status_cat'] == 'Partisan').astype(int)

    print("\nSplit vs Unified Control:")
    print(f"Split control: {df['split_control'].sum()}")
    print(f"Partisan control: {df['partisan_control'].sum()}")

    # Create additional indicators for gerrymandering risk
    df['legislature_partisan'] = ((df['Institution'] == 'Legislature') &
                                  (df['control_status_cat'] == 'Partisan')).astype(int)

    df['court_drawn'] = df['Drawn by'].isin(['State court', 'Federal court']).astype(int)

    df['independent_commission'] = (df['Institution'] == 'Independent commission').astype(int)

    # Print counts of high-risk and low-risk indicators
    print("\nGerrymandering Risk Indicators:")
    print(f"Legislature with partisan control: {df['legislature_partisan'].sum()}")
    print(f"Court-drawn maps: {df['court_drawn'].sum()}")
    print(f"Independent commission: {df['independent_commission'].sum()}")

    # Create a composite risk score
    # Higher score = higher gerrymandering risk
    df['gerrymandering_risk'] = (
            (df['legislature_partisan'] * 2) +  # High risk factor
            (df['partisan_control'] * 1) -  # Medium risk factor
            (df['court_drawn'] * 1.5) -  # Protective factor
            (df['independent_commission'] * 2)  # Protective factor
    )

    # Categorize risk
    def risk_category(score):
        if score >= 2:
            return 'High Risk'
        elif score >= 0:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    df['risk_category'] = df['gerrymandering_risk'].apply(risk_category)

    # Print risk distribution
    print("\nGerrymandering Risk Distribution:")
    risk_counts = df['risk_category'].value_counts()
    print(risk_counts)

    return df

result_df = analyze_gerrymandering_data(df)

# Print a summary at the end
print("\nAnalysis Complete!")
print(f"Total maps analyzed: {len(result_df)}")
print(f"Maps at high risk for gerrymandering: {len(result_df[result_df['risk_category'] == 'High Risk'])}")
