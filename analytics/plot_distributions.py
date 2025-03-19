import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata
import pandas as pd
from sdv.metadata import Metadata

DATA_IN_DIRECTORY = "processed_data/synthetic_population_generation/"


def calculate_accuracies(tests, source_data_dir):
    """
    Calculate the accuracies of synthetic data compared to source data.

    Args:
        tests (list of tuples): List of (test_name, test_file_dir) pairs.
        source_data_dir (str): Path to the source data file.

    Returns:
        list: A list of tuples containing (test_name, individuals_score, pairs_score).
    """
    results = []

    source_df = pd.read_csv(source_data_dir)

    for test_name, test_file_dir in tests:
        gen_df = pd.read_csv(test_file_dir)

        common_columns = list(set(source_df.columns) & set(gen_df.columns))

        source_df_filtered = source_df[common_columns]
        gen_df_filtered = gen_df[common_columns]

        metadata = Metadata.detect_from_dataframe(
            data=gen_df_filtered,
            table_name='gen_df'
        )

        quality_report = evaluate_quality(
            real_data=source_df_filtered,
            synthetic_data=gen_df_filtered,
            metadata=metadata,
            verbose=False
        )

        scores = quality_report.get_properties()
        individuals_score = scores[scores["Property"] == "Column Shapes"]["Score"].values[0]
        pairs_score = scores[scores["Property"] == "Column Pair Trends"]["Score"].values[0]

        results.append((test_name, individuals_score, pairs_score))

    return results


def plot_accuracies(results, title):
    """
    Plots the accuracies of synthetic data evaluation.

    Args:
        results (list of tuples): List containing (test_name, individual_score, pair_score).
        title (str): Title for the plot.

    Returns:
        None
    """
    test_names = [item[0] for item in results]
    individual_scores = [item[1] for item in results]
    pair_scores = [item[2] for item in results]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].grid(zorder=0)
    axs[0].bar(test_names, individual_scores, color='blue', zorder=3)
    axs[0].set_ylabel("Individual Score")
    axs[0].set_title("Column Shapes Accuracy")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_xticklabels(test_names, rotation=45, ha='right', rotation_mode="anchor")
    axs[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axs[1].grid(zorder=0)
    axs[1].bar(test_names, pair_scores, color='green', zorder=3)
    axs[1].set_ylabel("Pair Score")
    axs[1].set_title("Column Pair Trends Accuracy")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].set_xticklabels(test_names, rotation=45, ha='right', rotation_mode="anchor")
    axs[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("analytics.png")

def plot_distributions(source_df, gen_df, ref_df, columns, max_cols=2):
    import seaborn as sns
    import matplotlib.pyplot as plt

    """Plots distributions for specified columns in subplots.

    Args:
        source_df: DataFrame containing source data.
        gen_df: DataFrame containing generated data.
        ref_df: DataFrame containing reference data.
        columns: List of column names to plot.
        max_cols: Maximum number of subplots per row.
    """

    num_plots = len(columns)
    num_rows = (num_plots + max_cols - 1) // max_cols

    fig, axes = plt.subplots(num_rows, min(max_cols, num_plots), figsize=(5 * min(max_cols, num_plots) , 4 * num_rows), squeeze=False)
    row, col = 0, 0
    for column in columns:
        lower_bound = min(source_df[column].quantile([0.02]).values[0], gen_df[column].quantile([0.02]).values[0], ref_df[column].quantile([0.02]).values[0])
        upper_bound = max(source_df[column].quantile([0.98]).values[0], gen_df[column].quantile([0.98]).values[0], ref_df[column].quantile([0.98]).values[0])
        
        ax = axes[row, col]  # Get the correct subplot axes

        sns.kdeplot(gen_df[column], ax=ax, label="Generated", bw_adjust=1)
        sns.kdeplot(source_df[column], ax=ax, label="ACS", bw_adjust=1)
        sns.kdeplot(ref_df[column], ax=ax, label="Reference", bw_adjust=1)
        ax.set_xlim(lower_bound, upper_bound)
        ax.legend()
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.set_title(f'{column} Distribution')

        col += 1
        if col == max_cols:
            col = 0
            row += 1

    # Hide any unused subplots
    for i in range(row, num_rows):
      for j in range(col, max_cols):
        fig.delaxes(axes[i,j])

        plt.tight_layout()  # Adjust subplot parameters for a tight layout
        plt.savefig("distributions.png")  # Save the plot locally

if __name__ == "__main__":
    source_df = pd.read_csv(DATA_IN_DIRECTORY + 'source_ACS/acs_people.csv')
    gen_df = pd.read_csv("output/people.csv")
    ref_df = pd.read_csv(DATA_IN_DIRECTORY + "../agents/people.csv")
    common_columns = list(set(source_df) & set(gen_df))

    course_df = source_df[common_columns]
    gen_df = gen_df[common_columns]

    metadata = Metadata.detect_from_dataframe(
        data=gen_df,
        table_name='gen_df')
    numerical_columns = [column for column in metadata.tables["gen_df"].columns if metadata.tables["gen_df"].columns[column]['sdtype'] == "numerical" and "weight" not in column and "hh_id" not in column] 
    plot_distributions(source_df, gen_df, ref_df, numerical_columns)