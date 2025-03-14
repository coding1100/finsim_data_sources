import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata

DATA_IN_DIRECTORY = "../../processed_data/synthetic_population_generation/"


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
    plt.show()
