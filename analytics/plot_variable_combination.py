import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

from sdv.metadata import Metadata

DATA_IN_DIRECTORY = "processed_data/synthetic_population_generation/"

def plot_variable_combinations_comparison(df_a, df_b, variable_names, max_cols=2, num_bins=10):

    num_combinations = len(list(combinations(variable_names, 2)))
    num_rows = (num_combinations + max_cols - 1) // max_cols

    fig, axes = plt.subplots(num_rows, min(max_cols, num_combinations), figsize=(12 * min(max_cols, num_combinations), 10 * num_rows), squeeze=False)

    combination_index = 0
    for var1, var2 in combinations(variable_names, 2):
        row = combination_index // max_cols
        col = combination_index % max_cols
        ax = axes[row, col]

        # Check if variables are continuous and discretize if needed
        if pd.api.types.is_numeric_dtype(df_a[var1]):  # Check if var1 is continuous
            bins_col1_a = pd.cut(df_a[var1], bins=num_bins, include_lowest=True, duplicates='drop') # Added include_lowest and duplicates
            bins_col1_b = pd.cut(df_b[var1], bins=bins_col1_a.cat.categories, include_lowest=True) # Use same bins for df_b

            col1_label = f'{var1} (binned)'
        else:
          bins_col1_a = df_a[var1]
          bins_col1_b = df_b[var1]
          col1_label = var1

        if pd.api.types.is_numeric_dtype(df_a[var2]):  # Check if var2 is continuous
            bins_col2_a = pd.cut(df_a[var2], bins=num_bins, include_lowest=True, duplicates='drop') # Added include_lowest and duplicates
            bins_col2_b = pd.cut(df_b[var2], bins=bins_col2_a.cat.categories, include_lowest=True) # Use same bins for df_b

            col2_label = f'{var2} (binned)'
        else:
          bins_col2_a = df_a[var2]
          bins_col2_b = df_b[var2]
          col2_label = var2

        # Create contingency tables
        contingency_a = pd.crosstab(bins_col1_a, bins_col2_a, normalize='index').fillna(0)
        contingency_b = pd.crosstab(bins_col1_b, bins_col2_b, normalize='index').fillna(0)

        # Ensure both tables have the same indices and columns by reindexing
        all_categories_col1 = sorted(list(set(contingency_a.index) | set(contingency_b.index)))
        all_categories_col2 = sorted(list(set(contingency_a.columns) | set(contingency_b.columns)))

        contingency_a = contingency_a.reindex(all_categories_col1).reindex(columns=all_categories_col2).fillna(0)
        contingency_b = contingency_b.reindex(all_categories_col1).reindex(columns=all_categories_col2).fillna(0)


        # Calculate the difference as a percentage
        difference = (contingency_a - contingency_b) * 100

        # Calculate the absolute difference for coloring
        abs_difference = np.abs(difference)

        # Create a combined dataframe for display
        combined = pd.DataFrame(index=contingency_a.index, columns=contingency_a.columns)
        for c in combined.columns:
            combined[c] = [f"{a:.2f}%\n{b:.2f}%\n{diff:.2f}%" for a, b, diff in zip(contingency_a[c] * 100, contingency_b[c] * 100, difference[c])]

        # Plot the combined table as a heatmap
        sns.heatmap(abs_difference, annot=combined, fmt="s", cmap="RdYlGn_r", ax=ax)
        ax.set_xlabel(col2_label)
        ax.set_ylabel(col1_label)
        ax.set_title(f'{var1} vs {var2} Distribution')

        combination_index += 1

    # Hide any unused subplots
    for i in range(row + 1, num_rows):
        for j in range(0, max_cols):
          fig.delaxes(axes[i,j])
    for j in range(col + 1, max_cols):
        fig.delaxes(axes[row,j])

    plt.tight_layout()
    plt.savefig("variable_combination.png")

if __name__ == "__main__":
    source_df = pd.read_csv(DATA_IN_DIRECTORY + 'source_ACS/acs_people.csv')
    gen_df = pd.read_csv("output/people.csv")
    ref_df = pd.read_csv(DATA_IN_DIRECTORY + "../agents/people.csv")
    common_columns = list(set(source_df) & set(gen_df))

    course_df = source_df[common_columns]
    gen_df = gen_df[common_columns]

    metadata = Metadata.detect_from_dataframe(
        data=gen_df,
        table_name='gen_df'
)
    variable_names = [col for col in source_df.columns if col in gen_df.columns]
    variable_names = [col for col in variable_names if len(source_df[col].unique()) > 1 and len(gen_df[col].unique()) > 1]
    variable_names = [col for col in variable_names if col != "hh_id" and col != "weight_cross_sectional"]
    print(variable_names)
    plot_variable_combinations_comparison(source_df, gen_df, variable_names, num_bins=10)
