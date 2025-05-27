import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_missing_values(df):
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

    if len(missing_data) > 0:
        plt.figure(figsize=(12, 6))

        # Missing values bar plot
        plt.subplot(1, 2, 1)
        sns.barplot(data=missing_data, y='Column', x='Missing_Percentage', palette='viridis')
        plt.title('Missing Values by Column (%)')
        plt.xlabel('Missing Percentage')

        # Missing values heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')

        plt.tight_layout()
        plt.show()

        return missing_data
    else:
        print("No missing values found!")
        return pd.DataFrame()
