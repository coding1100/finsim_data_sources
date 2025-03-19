import pandas as pd
from rdt.transformers.categorical import LabelEncoder
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

COLUMNS_IN_ACS_PEOPLE_DF = [
    "person_id", "employment_status", "relationshipToR", "age", "yearOfBirth",
    "gender", "weight_cross_sectional", "adm1", "adm2", "puma", "race", "educ_years",
    "educ_level", "partnership_status", "elderly_or_disabled", "is_kid", "is_adult",
    "school_type", "degree_field", "occupation_soc", "occupation_naics", "veteran", "citizen",
    "divorce_lstyear", "marriage_lstyear", "fertility_lstyear", "widow_lstyear", "income_wage",
    "income_business", "income_ss", "income_welfare", "income_investment", "income_retirement",
    "income_supss", "income_other", "income_totalearned", "cur_income",
]


def read_acs_people(file_path: str) -> pd.DataFrame:
    """
    Reads and preprocesses the ACS people dataset.
    """
    acs_people_df = pd.read_csv(file_path)
    acs_people_df = acs_people_df[COLUMNS_IN_ACS_PEOPLE_DF]
    return acs_people_df


def train_acs_generator(acs_dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:
    """
    Trains a Gaussian Copula Synthesizer for ACS data.
    """
    acs_metadata = Metadata.detect_from_dataframe(data=acs_dataframe, table_name='acs_all')

    categorical_variables = ["adm1", "adm2", "puma"]
    for categorical_variable in categorical_variables:
        if categorical_variable in acs_dataframe.columns:
            acs_metadata.update_column(column_name=categorical_variable, sdtype='categorical')

    acs_people_generator = GaussianCopulaSynthesizer(metadata=acs_metadata, enforce_min_max_values=False)
    acs_people_generator.auto_assign_transformers(acs_dataframe)

    for categorical_variable in categorical_variables:
        if categorical_variable in acs_dataframe.columns:
            acs_people_generator.update_transformers({categorical_variable: LabelEncoder(add_noise=False)})

    acs_people_generator.fit(acs_dataframe)

    return acs_people_generator


def impute_missing_values_acs_people(acs_people_df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values iteratively using synthetic data generation.
    """
    all_columns_with_nans = acs_people_df.columns[acs_people_df.isna().any()].tolist()

    while len(all_columns_with_nans) > 0:

        all_columns_without_nans = acs_people_df.columns[acs_people_df.notna().all()].tolist()
        count_of_nans_per_column = acs_people_df.isna().sum()[all_columns_with_nans].sort_values()
        column_to_impute = count_of_nans_per_column.index[0]
        indices_to_impute = acs_people_df[acs_people_df[column_to_impute].isna()].index

        print(f"Imputing {len(indices_to_impute)} values for column: {column_to_impute}")

        columns_for_imputation_basis = all_columns_without_nans
        columns_in_imputation_model = columns_for_imputation_basis + [column_to_impute]
        training_data_for_imputation_model = acs_people_df[columns_in_imputation_model].dropna()

        acs_imputation_generator = train_acs_generator(training_data_for_imputation_model)

        imputed_rows = acs_imputation_generator.sample_remaining_columns(
            known_columns=acs_people_df.loc[indices_to_impute][columns_for_imputation_basis],
            max_tries_per_batch=20
        )

        indices_failed_to_impute = indices_to_impute.difference(imputed_rows.index)
        if len(indices_failed_to_impute) > 0:
            print(f"Failed to impute {len(indices_failed_to_impute)} rows.")
            print(indices_failed_to_impute)
            print("Consider using a default value instead of dropping the data.")
            acs_people_df = acs_people_df.drop(index=indices_failed_to_impute)

        acs_people_df.update(imputed_rows[column_to_impute])

        print(f"Successfully imputed {len(imputed_rows)} values for column: {column_to_impute}")

        all_columns_with_nans = acs_people_df.columns[acs_people_df.isna().any()].tolist()

    return acs_people_df


if __name__ == "__main__":
    acs_people_df = read_acs_people("processed_data/synthetic_population_generation/source_ACS/acs_people.csv")
    columns_to_drop = ["school_type", "citizen", "divorce_lstyear", "marriage_lstyear", "fertility_lstyear",
                    "widow_lstyear"]
    acs_people_df = acs_people_df.drop(columns=columns_to_drop, errors='ignore')
    acs_people_df = acs_people_df[acs_people_df["age"] > 0]
    acs_people_imputed_df = impute_missing_values_acs_people(acs_people_df)
    acs_people_imputed_df.to_csv("acs_people_imputed.csv", index=False)

    print("Data processing complete. Imputed dataset saved as 'acs_people_imputed.csv'.")
