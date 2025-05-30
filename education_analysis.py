import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


# Function to load and examine the dataset
def load_and_examine_dataset(file_path):
    """
    Load the dataset and perform initial examination
    """
    print(f"Attempting to read file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few columns:")
    for i, col in enumerate(df.columns[:10]):
        print(f"{i}: {col}")

    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing_values}")

    # Examine the first few rows
    print("\nDetailed first 5 rows sample (transposed for readability):")
    print(df.head().T)

    return df


# Function to restructure the dataset using the real headers
def restructure_education_data(df):
    """
    Restructure the dataset using the actual column headers from row 1
    """
    print("\n=== Restructuring Congressional Dataset ===")

    # Based on our exploration, we know row 1 (index 1) contains the actual headers
    print("Using header row at index 1")

    # Get the headers from row 1
    headers = df.iloc[1].values

    # Create clean column names
    clean_headers = []
    for i, header in enumerate(headers):
        if pd.isna(header) or header == '':
            # Keep original unnamed column names
            if i < len(df.columns):
                clean_headers.append(df.columns[i])
            else:
                clean_headers.append(f"Column_{i}")
        else:
            # Clean and standardize header name
            clean_header = str(header).strip().replace(" ", "_").lower()
            clean_headers.append(clean_header)

    # Create a new dataframe with the proper data rows
    new_df = df.iloc[2:].copy()
    new_df.columns = clean_headers

    # Reset index for the new dataframe
    new_df = new_df.reset_index(drop=True)

    # Print information about the restructured dataset
    print(f"\nRestructured dataset shape: {new_df.shape}")
    print("\nNew column names:")
    for i, col in enumerate(new_df.columns[:25]):
        print(f"{i}: {col}")

    # Print first few rows of restructured data
    print("\nFirst 3 rows of restructured data:")
    print(new_df.head(3))

    return new_df


# Function to categorize education levels
def categorize_education(df):
    """
    Categorize education levels in the dataset
    """
    print("\n=== Categorizing Education Levels ===")

    # First, check if the educational_attainment column exists
    if 'educational_attainment' in df.columns:
        education_col = 'educational_attainment'
    else:
        # Try to find the education column by examining column contents
        for col in df.columns:
            if col.lower().startswith('unnamed'):
                sample_values = df[col].dropna().unique()[:10]
                if any('degree' in str(val).lower() for val in sample_values):
                    education_col = col
                    print(f"Found education column: {col}")
                    print(f"Sample values: {sample_values}")
                    break
        else:
            print("Could not find education column. Please check the dataset structure.")
            return df

    print(f"Using '{education_col}' as the primary education column")

    # Print a sample of education values
    print("\nSample education values:")
    sample_values = df[education_col].dropna().unique()[:10]
    for val in sample_values:
        print(f"  - {val}")

    # Define education categorization function
    def education_category(edu_str):
        if pd.isna(edu_str) or edu_str == '':
            return 'Unknown'

        edu_str = str(edu_str).lower()

        # Map education levels to standardized categories
        if any(term in edu_str for term in ['doctorate', 'ph.d', 'phd']):
            return 'PhD'
        elif any(term in edu_str for term in ['professional', 'jd', 'j.d.', 'law degree', 'juris']):
            return 'JD (Law Degree)'
        elif any(term in edu_str for term in ['md', 'm.d.', 'medical degree']):
            return 'MD (Medical Degree)'
        elif any(term in edu_str for term in ['master', 'ma ', 'm.a.', 'ms ', 'm.s.', 'mba']):
            return 'Master\'s Degree'
        elif any(term in edu_str for term in ['bachelor', 'ba ', 'b.a.', 'bs ', 'b.s.']):
            return 'Bachelor\'s Degree'
        elif any(term in edu_str for term in ['associate', 'aa ', 'a.a.', 'as ', 'a.s.']):
            return 'Associate\'s Degree'
        elif any(term in edu_str for term in ['high school', 'secondary']):
            return 'High School'
        elif any(term in edu_str for term in ['some college', 'college', 'university', 'institution']):
            return 'Some College'
        else:
            return 'Other/Unknown'

    # Apply categorization
    df['education_category'] = df[education_col].apply(education_category)

    # Count occurrences of each category
    category_counts = df['education_category'].value_counts()
    print("\nEducation Category Counts:")
    print(category_counts)

    return df


# Function to prepare additional demographic data
def prepare_demographic_data(df):
    """
    Process demographic information like party and gender
    """
    print("\n=== Preparing Demographic Data ===")

    # Process party information
    if 'party' in df.columns:
        print("Found party column")

        # Simplify party affiliations
        def simplify_party(party_str):
            if pd.isna(party_str) or party_str == '':
                return 'Unknown'

            party_str = str(party_str).lower()

            if 'democrat' in party_str:
                return 'Democrat'
            elif 'republican' in party_str:
                return 'Republican'
            else:
                return 'Other'

        df['party_simplified'] = df['party'].apply(simplify_party)

        # Show party distribution
        party_counts = df['party_simplified'].value_counts()
        print("\nParty Distribution:")
        print(party_counts)

    # Process gender information
    if 'sex' in df.columns:
        print("Found gender/sex column")

        # Simplify gender
        def simplify_gender(gender_str):
            if pd.isna(gender_str) or gender_str == '':
                return 'Unknown'

            gender_str = str(gender_str)

            # Based on our data exploration, it looks like:
            # 1 = Female, 2 = Male
            if gender_str == '1':
                return 'Female'
            elif gender_str == '2':
                return 'Male'
            else:
                return 'Other/Unknown'

        df['gender_simplified'] = df['sex'].apply(simplify_gender)

        # Show gender distribution
        gender_counts = df['gender_simplified'].value_counts()
        print("\nGender Distribution:")
        print(gender_counts)

    # Process race/ethnicity information
    if 'race' in df.columns:
        print("Found race/ethnicity column")

        # Show race distribution
        race_counts = df['race'].value_counts()
        print("\nRace/Ethnicity Distribution:")
        print(race_counts)

    return df


# Function to analyze education by Congress
def analyze_education_by_congress(df):
    """
    Analyze trends in education levels across different Congresses
    """
    print("\n=== Analyzing Education by Congress ===")

    # Check if required columns exist
    if 'education_category' not in df.columns or 'congress' not in df.columns:
        print("Required columns not found for Congress analysis")
        return None

    # Create a pivot table of education categories by Congress
    edu_by_congress = pd.crosstab(df['education_category'], df['congress'])
    print("\nEducation Categories by Congress (sample):")
    print(edu_by_congress.head())

    # Calculate percentages
    edu_by_congress_pct = edu_by_congress.div(edu_by_congress.sum(axis=0), axis=1) * 100

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Plot percentage with Bachelor's or higher over time
    higher_edu = ['Bachelor\'s Degree', 'Master\'s Degree', 'JD (Law Degree)', 'MD (Medical Degree)', 'PhD']
    higher_edu_data = edu_by_congress_pct.loc[edu_by_congress_pct.index.isin(higher_edu)].sum(axis=0)

    plt.plot(higher_edu_data.index, higher_edu_data.values, marker='o', linewidth=2,
             label='Bachelor\'s or higher', color='blue')

    # Plot advanced degrees (Master's, JD, MD, PhD)
    advanced_degrees = ['Master\'s Degree', 'JD (Law Degree)', 'MD (Medical Degree)', 'PhD']
    advanced_edu_data = edu_by_congress_pct.loc[edu_by_congress_pct.index.isin(advanced_degrees)].sum(axis=0)

    plt.plot(advanced_edu_data.index, advanced_edu_data.values, marker='s', linewidth=2,
             label='Advanced degrees', color='green')

    # Add labels and formatting
    plt.title('Trends in Higher Education Levels in Congress')
    plt.xlabel('Congress')
    plt.ylabel('Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(sorted(df['congress'].unique()))

    plt.tight_layout()
    plt.savefig('education_trends_by_congress.png')
    plt.close()

    return edu_by_congress_pct


# Function to analyze education by party
def analyze_education_by_party(df):
    """
    Analyze education levels by political party
    """
    print("\n=== Analyzing Education by Party ===")

    # Check if required columns exist
    if 'education_category' not in df.columns or 'party_simplified' not in df.columns:
        print("Required columns not found for party analysis")
        return None

    # Create a crosstab of education levels by party
    edu_by_party = pd.crosstab(df['education_category'], df['party_simplified'])
    print("\nEducation levels by party:")
    print(edu_by_party)

    # Calculate percentages
    edu_by_party_pct = edu_by_party.div(edu_by_party.sum(axis=0), axis=1) * 100

    # Create visualization
    plt.figure(figsize=(10, 6))
    edu_by_party_pct.plot(kind='bar', stacked=False)
    plt.title('Education Levels by Political Party')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(title='Party')
    plt.tight_layout()
    plt.savefig('education_by_party.png')
    plt.close()

    # Analyze party education trends over time
    if 'congress' in df.columns:
        plt.figure(figsize=(12, 8))

        # Create separate plots for Democrats and Republicans
        parties = ['Democrat', 'Republican']
        colors = ['blue', 'red']
        markers = ['o', 's']

        for i, party in enumerate(parties):
            if party in df['party_simplified'].unique():
                party_df = df[df['party_simplified'] == party]

                # Calculate percentage with at least Bachelor's by Congress for this party
                higher_edu_by_congress = {}
                for congress in sorted(party_df['congress'].unique()):
                    congress_party_df = party_df[party_df['congress'] == congress]
                    higher_edu_count = congress_party_df[congress_party_df['education_category'].isin(
                        ['Bachelor\'s Degree', 'Master\'s Degree', 'JD (Law Degree)', 'MD (Medical Degree)',
                         'PhD'])].shape[0]
                    total_count = congress_party_df.shape[0]
                    if total_count > 0:
                        higher_edu_by_congress[congress] = higher_edu_count / total_count * 100

                # Plot the trend for this party
                congresses = list(higher_edu_by_congress.keys())
                percentages = list(higher_edu_by_congress.values())
                plt.plot(congresses, percentages, marker=markers[i], linewidth=2,
                         label=f'{party} (Bachelor\'s or higher)', color=colors[i])

        plt.title('Higher Education Trends by Party Over Time')
        plt.xlabel('Congress')
        plt.ylabel('Percentage with Bachelor\'s or higher')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(sorted(df['congress'].unique()))

        plt.tight_layout()
        plt.savefig('education_trends_by_party.png')
        plt.close()

    return edu_by_party_pct


# Function to analyze education by gender
def analyze_education_by_gender(df):
    """
    Analyze education levels by gender
    """
    print("\n=== Analyzing Education by Gender ===")

    # Check if required columns exist
    if 'education_category' not in df.columns or 'gender_simplified' not in df.columns:
        print("Required columns not found for gender analysis")
        return None

    # Create a crosstab of education levels by gender
    edu_by_gender = pd.crosstab(df['education_category'], df['gender_simplified'])
    print("\nEducation levels by gender:")
    print(edu_by_gender)

    # Calculate percentages
    edu_by_gender_pct = edu_by_gender.div(edu_by_gender.sum(axis=0), axis=1) * 100

    # Create visualization
    plt.figure(figsize=(10, 6))
    edu_by_gender_pct.plot(kind='bar', stacked=False)
    plt.title('Education Levels by Gender')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig('education_by_gender.png')
    plt.close()

    # Analyze gender distribution trends over time
    if 'congress' in df.columns:
        gender_by_congress = pd.crosstab(df['gender_simplified'], df['congress'])
        gender_by_congress_pct = gender_by_congress.div(gender_by_congress.sum(axis=0), axis=1) * 100

        plt.figure(figsize=(12, 6))
        for gender in ['Female', 'Male']:
            if gender in gender_by_congress_pct.index:
                plt.plot(gender_by_congress_pct.columns, gender_by_congress_pct.loc[gender],
                         marker='o', linewidth=2, label=gender)

        plt.title('Gender Distribution in Congress Over Time')
        plt.xlabel('Congress')
        plt.ylabel('Percentage')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(sorted(df['congress'].unique()))

        plt.tight_layout()
        plt.savefig('gender_trends.png')
        plt.close()

    return edu_by_gender_pct


# Function to analyze education by race
def analyze_education_by_race(df):
    """
    Analyze education levels by race/ethnicity
    """
    print("\n=== Analyzing Education by Race/Ethnicity ===")

    # Check if required columns exist
    if 'education_category' not in df.columns or 'race' not in df.columns:
        print("Required columns not found for race/ethnicity analysis")
        return None

    # Create a crosstab of education levels by race
    edu_by_race = pd.crosstab(df['education_category'], df['race'])
    print("\nEducation levels by race/ethnicity:")
    print(edu_by_race)

    # Calculate percentages
    edu_by_race_pct = edu_by_race.div(edu_by_race.sum(axis=0), axis=1) * 100

    # Create visualization - filter to most common races for readability
    major_races = edu_by_race.sum(axis=0).nlargest(5).index
    edu_by_race_filtered = edu_by_race_pct[major_races]

    plt.figure(figsize=(12, 8))
    edu_by_race_filtered.plot(kind='bar', stacked=False)
    plt.title('Education Levels by Race/Ethnicity')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(title='Race/Ethnicity')
    plt.tight_layout()
    plt.savefig('education_by_race.png')
    plt.close()

    return edu_by_race_pct


# Function to analyze education by state
def analyze_education_by_state(df):
    """
    Analyze education levels by state
    """
    print("\n=== Analyzing Education by State ===")

    # Check if required columns exist
    if 'education_category' not in df.columns or 'state' not in df.columns:
        print("Required columns not found for state analysis")
        return None

    # Get the top 10 states by representative count
    top_states = df['state'].value_counts().head(10).index

    # Filter to these states
    state_df = df[df['state'].isin(top_states)]

    # Create a crosstab
    edu_by_state = pd.crosstab(state_df['education_category'], state_df['state'])

    # Calculate percentages
    edu_by_state_pct = edu_by_state.div(edu_by_state.sum(axis=0), axis=1) * 100

    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(edu_by_state_pct, cmap='viridis', annot=True, fmt='.1f', cbar_kws={'label': 'Percentage'})
    plt.title('Education Levels by Top 10 States')
    plt.ylabel('Education Level')
    plt.xlabel('State')
    plt.tight_layout()
    plt.savefig('education_by_state.png')
    plt.close()

    return edu_by_state_pct


# Function to analyze candidate quality metrics
def analyze_quality_metrics(df):
    """
    Analyze education-based quality metrics for House representatives
    """
    print("\n=== Analyzing Candidate Quality Metrics ===")

    # Check if required columns exist
    if 'education_category' not in df.columns or 'congress' not in df.columns:
        print("Required columns not found for quality metrics analysis")
        return None

    # Define education quality metrics
    advanced_degrees = ['PhD', 'MD (Medical Degree)', 'JD (Law Degree)', 'Master\'s Degree']
    law_degrees = ['JD (Law Degree)']

    # Calculate metrics by congress
    quality_metrics = {}
    for congress in sorted(df['congress'].unique()):
        congress_df = df[df['congress'] == congress]
        total = len(congress_df)

        if total == 0:
            continue

        # Calculate percentages
        adv_degree_count = len(congress_df[congress_df['education_category'].isin(advanced_degrees)])
        law_degree_count = len(congress_df[congress_df['education_category'].isin(law_degrees)])

        quality_metrics[congress] = {
            'total_members': total,
            'advanced_degree_pct': adv_degree_count / total * 100,
            'law_degree_pct': law_degree_count / total * 100
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(quality_metrics, orient='index')
    print("\nQuality Metrics by Congress:")
    print(metrics_df)

    # Visualize trends
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df.index, metrics_df['advanced_degree_pct'],
             marker='o', linewidth=2, label='Advanced Degrees')
    plt.plot(metrics_df.index, metrics_df['law_degree_pct'],
             marker='s', linewidth=2, label='Law Degrees')

    plt.title('Trends in Educational Quality Metrics')
    plt.xlabel('Congress')
    plt.ylabel('Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(metrics_df.index)

    plt.tight_layout()
    plt.savefig('quality_metrics_trends.png')
    plt.close()

    return metrics_df


# Function to generate summary statistics
def generate_summary(df):
    """
    Generate a summary of the education analysis
    """
    print("\n===== CONGRESSIONAL EDUCATION SUMMARY =====")

    # Overall education distribution
    edu_counts = df['education_category'].value_counts()
    total = len(df)
    print("\nOverall Education Distribution:")
    for edu, count in edu_counts.items():
        print(f"{edu}: {count} ({count / total * 100:.1f}%)")

    # Education by party
    if 'party_simplified' in df.columns:
        print("\nAdvanced Degrees by Party:")
        advanced_degrees = ['PhD', 'MD (Medical Degree)', 'JD (Law Degree)', 'Master\'s Degree']

        for party in sorted(df['party_simplified'].unique()):
            party_df = df[df['party_simplified'] == party]
            party_total = len(party_df)

            if party_total == 0:
                continue

            adv_count = len(party_df[party_df['education_category'].isin(advanced_degrees)])
            print(f"{party}: {adv_count / party_total * 100:.1f}% have advanced degrees")

    # Education by gender
    if 'gender_simplified' in df.columns:
        print("\nAdvanced Degrees by Gender:")
        for gender in sorted(df['gender_simplified'].unique()):
            gender_df = df[df['gender_simplified'] == gender]
            gender_total = len(gender_df)

            if gender_total == 0:
                continue

            adv_count = len(gender_df[gender_df['education_category'].isin(advanced_degrees)])
            print(f"{gender}: {adv_count / gender_total * 100:.1f}% have advanced degrees")

    # Education over time
    if 'congress' in df.columns:
        print("\nEducation Trends Over Time:")
        print(f"{'Congress':<10}{'Advanced (%)':<15}{'Law Degrees (%)':15}")
        print("-" * 40)

        for congress in sorted(df['congress'].unique()):
            congress_df = df[df['congress'] == congress]
            congress_total = len(congress_df)

            if congress_total == 0:
                continue

            adv_count = len(congress_df[congress_df['education_category'].isin(advanced_degrees)])
            law_count = len(congress_df[congress_df['education_category'] == 'JD (Law Degree)'])

            adv_pct = adv_count / congress_total * 100
            law_pct = law_count / congress_total * 100

            print(f"{congress:<10}{adv_pct:<15.1f}{law_pct:<15.1f}")


# Main function to run the analysis
def main():
    """
    Main function to run all analyses
    """
    print("===== CONGRESSIONAL EDUCATION DATA ANALYSIS =====\n")

    # File path to the merged dataset
    file_path = "merged_congress_education.csv"

    # 1. Load and examine the dataset
    df = load_and_examine_dataset(file_path)

    # 2. Restructure the dataset using the actual headers
    restructured_df = restructure_education_data(df)

    # 3. Categorize education levels
    processed_df = categorize_education(restructured_df)

    # 4. Prepare demographic data
    processed_df = prepare_demographic_data(processed_df)

    # 5. Run analyses
    print("\n===== RUNNING ANALYSES =====")

    # Education by Congress
    edu_by_congress = analyze_education_by_congress(processed_df)

    # Education by Party
    edu_by_party = analyze_education_by_party(processed_df)

    # Education by Gender
    edu_by_gender = analyze_education_by_gender(processed_df)

    # Education by Race
    edu_by_race = analyze_education_by_race(processed_df)

    # Education by State
    edu_by_state = analyze_education_by_state(processed_df)

    # Education-based Quality Metrics
    quality_metrics = analyze_quality_metrics(processed_df)

    # Generate summary
    generate_summary(processed_df)

    # Export processed data
    print("\n===== ANALYSIS COMPLETE =====")
    print("Visualizations saved to:")
    print("- education_trends_by_congress.png")
    print("- education_by_party.png")
    print("- education_trends_by_party.png")
    print("- education_by_gender.png")
    print("- gender_trends.png")
    print("- education_by_race.png")
    print("- education_by_state.png")
    print("- quality_metrics_trends.png")

    # Save processed data
    processed_df.to_csv("processed_education_data.csv", index=False)
    print("\nProcessed data saved to: processed_education_data.csv")


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()