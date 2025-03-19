import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from itertools import combinations
from sdv.metadata import Metadata

DATA_IN_DIRECTORY = "processed_data/synthetic_population_generation/"

def plot_categorical_combinations(df_a, df_b, dataset_a_name, dataset_b_name, categorical_cols, max_cols=2):

    num_combinations = len(list(combinations(categorical_cols, 2)))
    num_rows = (num_combinations + max_cols - 1) // max_cols

    fig, axes = plt.subplots(num_rows, min(max_cols, num_combinations), figsize=(10 * min(max_cols, num_combinations), 8 * num_rows), squeeze=False)

    combination_index = 0
    for col1, col2 in combinations(categorical_cols, 2):
        row = combination_index // max_cols
        col = combination_index % max_cols
        ax = axes[row, col]

        # Create a contingency table (cross-tabulation)
        contingency_a = pd.crosstab(df_a[col1], df_a[col2], normalize='index')
        contingency_b = pd.crosstab(df_b[col1], df_b[col2], normalize='index')
        all_categories_col1 = sorted(list(set(contingency_a.index) | set(contingency_b.index)))
        all_categories_col2 = sorted(list(set(contingency_a.columns) | set(contingency_b.columns)))
        contingency_a = contingency_a.reindex(all_categories_col1).reindex(columns=all_categories_col2).fillna(0)
        contingency_b = contingency_b.reindex(all_categories_col1).reindex(columns=all_categories_col2).fillna(0)

        difference = (contingency_b - contingency_a) / ((contingency_a + contingency_b) + 1e-10) * 100
        abs_difference = np.abs(difference)
        combined = pd.DataFrame(index=contingency_a.index, columns=contingency_a.columns)
        for c in combined.columns:
            combined[c] = [f"{dataset_a_name}: {a:.2f}\n{dataset_b_name}: {b:.2f}\nRD: {diff:.1f}%" for a, b, diff in zip(contingency_a[c], contingency_b[c], difference[c])]

        # Plot the contingency table as a heatmap
        sns.heatmap(abs_difference, annot=combined, fmt="s", cmap="RdYlGn_r", ax=ax)
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        ax.set_title(f'{col1} vs {col2} Distribution')

        combination_index += 1

    # Hide any unused subplots
    for i in range(row + 1, num_rows):
        for j in range(0, max_cols):
          fig.delaxes(axes[i,j])
    for j in range(col + 1, max_cols):
        fig.delaxes(axes[row,j])



    plt.tight_layout()
    plt.savefig("categorical_combinations.png")


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


    categorical_columns = [column for column in metadata.tables["gen_df"].columns if metadata.tables["gen_df"].columns[column]['sdtype'] == "categorical"] 
    categorical_columns = [col for col in categorical_columns if col in source_df.columns and col in gen_df.columns]
    categorical_columns = [col for col in categorical_columns if len(source_df[col].unique()) > 1 and len(gen_df[col].unique()) > 1]
    print(categorical_columns)
    plot_categorical_combinations(source_df, gen_df, "S", "G", categorical_columns)
# S: Source data
# G: Generated data
# RD: Relative difference (between -100% and 100%)