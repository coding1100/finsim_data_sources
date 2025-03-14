import numpy as np
import pandas as pd
from rdt.transformers.categorical import LabelEncoder
from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer

DATA_IN_DIRECTORY = "../../processed_data/synthetic_population_generation/"

columns_in_acs_people_df = list(set([
    "acs_hh_id", "person_id", "employment_status", "age", "gender", "weight_cross_sectional",
    "adm1", "race", "educ_years", "educ_level", "partnership_status", "elderly_or_disabled",
    "is_kid", "is_adult", "cur_income"
]) - set(["relationshipToR", "age_of_retirement", "adm2", "citizen", "degree_field", "divorce_lstyear",
          "fertility_lstyear", "income_business", "income_investment", "income_other", "income_retirement",
          "income_ss", "income_supss", "income_totalearned", "income_wage", "income_welfare",
          "marriage_lstyear", "occupation_naics", "occupation_soc", "puma", "relationshipToR", "school_type",
          "veteran", "widow_lstyear", "yearOfBirth"]))

columns_in_acs_households_df = [
    "acs_hh_id", "adm1", "puma", "acs_weight_cross_sectional", "num_elderly_or_disabled",
    "num_kids", "num_adults", "household_income", "foodstamp", "couple_type"
]

columns_in_psid_households_df = [
    "psid_hh_id", "partnership_status", "psid_weight_cross_sectional", "active_savings_rate",
    "large_gift", "small_gift", "num_elderly_or_disabled", "num_kids", "num_adults"
]


def read_acs_households(acs_households_filename: str):
    acs_households_df = pd.read_csv(acs_households_filename)
    acs_households_df["acs_hh_id"] = acs_households_df["hh_id"]
    acs_households_df["acs_weight_cross_sectional"] = acs_households_df["weight_cross_sectional"]
    acs_households_df = acs_households_df[columns_in_acs_households_df]
    acs_households_df = acs_households_df.dropna()
    acs_households_df["size"] = acs_households_df["num_kids"] + acs_households_df["num_adults"]
    acs_households_df = acs_households_df.drop(columns=["num_kids", "num_adults", "num_elderly_or_disabled"])
    return acs_households_df


def read_acs_people(household_size: int, acs_people_filename: str):
    acs_people_df = pd.read_csv(acs_people_filename)
    acs_people_df["acs_hh_id"] = acs_people_df["hh_id"]
    acs_people_df = acs_people_df[columns_in_acs_people_df].drop(columns=["is_kid", "is_adult", "person_id"])

    acs_hh_ids_of_hh_size = acs_people_df.groupby("acs_hh_id").size()[lambda x: x == household_size].index
    acs_people_size_df = acs_people_df[acs_people_df["acs_hh_id"].isin(acs_hh_ids_of_hh_size)]

    dict_hh_to_people = {hh_id: [] for hh_id in acs_people_size_df["acs_hh_id"].unique()}
    for _, row in acs_people_size_df.iterrows():
        dict_hh_to_people[row["acs_hh_id"]].append(row)

    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1).T
    wide_people_df.columns = np.concatenate(
        [[f"{col}_{i}" for col in acs_people_df.columns] for i in range(household_size)])
    wide_people_df["acs_hh_id"] = wide_people_df["acs_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"acs_hh_id_{i}" for i in range(household_size)])

    return wide_people_df.dropna()


def read_acs_data_of_household_size(household_size: int, acs_households_filename: str, acs_people_filename: str):
    acs_households_df = read_acs_households(acs_households_filename)
    acs_people_with_size_df = read_acs_people(household_size, acs_people_filename)
    acs_all_size_df = pd.merge(acs_people_with_size_df, acs_households_df, on="acs_hh_id", how="inner")
    return acs_all_size_df.dropna()


def read_psid_households_csv(psid_households_filename: str):
    psid_households_df = pd.read_csv(psid_households_filename)
    psid_households_df = psid_households_df.rename(columns={"hh_id": "psid_hh_id"})
    psid_households_df["psid_weight_cross_sectional"] = psid_households_df["weight_cross_sectional"]
    psid_households_df = psid_households_df[columns_in_psid_households_df].dropna()
    psid_households_df["size"] = psid_households_df["num_kids"] + psid_households_df["num_adults"]
    return psid_households_df.drop(columns=["num_elderly_or_disabled", "num_kids", "num_adults"])


def read_psid_holdings_csv(psid_holdings_filename: str):
    assets_df = pd.read_csv(psid_holdings_filename)[["hh_id", "asset_type", "curr_value"]]
    asset_types = assets_df["asset_type"].unique()
    assets_wide_df = pd.DataFrame(columns=asset_types)

    for hh_id in assets_df["hh_id"].unique():
        assets_for_hh_df = assets_df[assets_df["hh_id"] == hh_id]
        assets_wide_df.loc[hh_id] = [assets_for_hh_df[assets_for_hh_df[
                                                          "asset_type"] == asset].curr_value.max() if asset in assets_for_hh_df.asset_type.values else 0
                                     for asset in asset_types]

    return assets_wide_df.reset_index().rename(columns={"index": "psid_hh_id"}).fillna(0)


def read_psid_people_csv(household_size: int, psid_people_filename: str):
    psid_people_df = pd.read_csv(psid_people_filename).rename(columns={"hh_id": "psid_hh_id"})
    psid_people_df = psid_people_df.drop(
        columns=["yearOfBirthI", "adm1_int", "is_kid", "is_adult", "employment_status"]).dropna()

    psid_hh_ids_of_hh_size = psid_people_df.groupby("psid_hh_id").size()[lambda x: x == household_size].index
    psid_people_size_df = psid_people_df[psid_people_df["psid_hh_id"].isin(psid_hh_ids_of_hh_size)]

    dict_hh_to_people = {hh_id: [] for hh_id in psid_people_size_df["psid_hh_id"].unique()}
    for _, row in psid_people_size_df.iterrows():
        dict_hh_to_people[row["psid_hh_id"]].append(row)

    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1).T
    wide_people_df.columns = np.concatenate(
        [[f"{col}_{i}" for col in psid_people_df.columns] for i in range(household_size)])
    wide_people_df["psid_hh_id"] = wide_people_df["psid_hh_id_0"]

    return wide_people_df.dropna()


if __name__ == "__main__":
    acs_households = read_acs_households("acs_households.csv")
    acs_people = read_acs_people(3, "acs_people.csv")
    psid_households = read_psid_households_csv("psid_households.csv")
    psid_people = read_psid_people_csv(3, "psid_people.csv")

    print("ACS Households:", acs_households.shape)
    print("ACS People:", acs_people.shape)
    print("PSID Households:", psid_households.shape)
    print("PSID People:", psid_people.shape)


def get_household_sizes_to_generate(sample_size: int, acs_households_filename: str) -> pd.DataFrame:
    expected_counts = pd.read_csv(acs_households_filename)
    expected_counts["size"] = expected_counts["num_kids"] + expected_counts["num_adults"]
    expected_counts = expected_counts[["weight_cross_sectional", "size"]].groupby("size").sum()
    expected_counts = (expected_counts / expected_counts.sum()) * sample_size

    rounded_counts = expected_counts.round()

    total_difference = int(sample_size - rounded_counts.sum().values[0])
    if total_difference != 0:
        expected_counts["fraction"] = expected_counts["weight_cross_sectional"] - rounded_counts[
            "weight_cross_sectional"]
        expected_counts = expected_counts.sort_values("fraction", ascending=False)
        indices = expected_counts.index.tolist()
        for i in range(abs(total_difference)):
            idx = indices[i % len(indices)]
            rounded_counts.at[idx, "weight_cross_sectional"] += np.sign(total_difference)

    return rounded_counts.drop(columns=["fraction"], errors="ignore").astype(int)


def train_acs_generator(acs_dataframe: pd.DataFrame) -> TVAESynthesizer:
    acs_metadata = Metadata.detect_from_dataframe(acs_dataframe, table_name='acs_all')
    acs_metadata.update_column("adm1", sdtype="categorical")
    acs_metadata.update_column("puma", sdtype="categorical")

    acs_generator = TVAESynthesizer(metadata=acs_metadata, epochs=40, verbose=True)
    acs_generator.auto_assign_transformers(acs_dataframe)
    acs_generator.update_transformers({"puma": LabelEncoder(add_noise=False), "adm1": LabelEncoder(add_noise=False)})
    acs_generator.fit(acs_dataframe)

    return acs_generator


def train_psid_generator(psid_dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:
    psid_metadata = Metadata.detect_from_dataframe(psid_dataframe, table_name="psid_all")
    psid_metadata.update_column("adm1", sdtype="categorical")

    for column in [col for col in psid_dataframe.columns if "years_partnered" in col]:
        psid_metadata.update_column(column, sdtype="numerical")

    psid_generator = GaussianCopulaSynthesizer(metadata=psid_metadata, enforce_min_max_values=False)
    psid_generator.auto_assign_transformers(psid_dataframe)
    psid_generator.update_transformers({"adm1": LabelEncoder(add_noise=False)})
    psid_generator.fit(psid_dataframe)

    return psid_generator


def train_sipp_generator(sipp_dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:
    sipp_metadata = Metadata.detect_from_dataframe(sipp_dataframe, table_name="sipp_all")

    for col in [col for col in sipp_dataframe.columns if "benefits_" in col or "educ_years" in col or "age" in col]:
        sipp_metadata.update_column(col, sdtype="numerical")

    sipp_generator = GaussianCopulaSynthesizer(metadata=sipp_metadata, enforce_min_max_values=False)
    sipp_generator.auto_assign_transformers(sipp_dataframe)
    sipp_generator.fit(sipp_dataframe)

    return sipp_generator


def generate_acs_one_size(generator: TVAESynthesizer, sample_size: int) -> pd.DataFrame:
    print(f"Generating ACS data with TVAE Synthesizer for {sample_size} samples")
    return generator.sample(sample_size)


def generate_psid_on_acs_one_size(psid_columns, generator: GaussianCopulaSynthesizer,
                                  generated_acs_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_match = list(
        set(generated_acs_df.columns) & set(psid_columns) - {col for col in generated_acs_df.columns if
                                                             "weight" in col})

    print(f"Generating PSID data based on ACS using columns: {columns_to_match}")
    generated_psid_df = generator.sample_remaining_columns(known_columns=generated_acs_df[columns_to_match],
                                                           max_tries_per_batch=20)

    return pd.concat(
        [generated_acs_df, generated_psid_df[generated_psid_df.columns.difference(generated_acs_df.columns)]], axis=1)


def generate_sipp_on_acs_psid_one_size(sipp_columns, generator: GaussianCopulaSynthesizer,
                                       generated_acs_psid_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_match = list(
        set(generated_acs_psid_df.columns) & set(sipp_columns) - {col for col in generated_acs_psid_df.columns if
                                                                  "weight" in col})

    print(f"Generating SIPP data based on ACS & PSID using columns: {columns_to_match}")
    generated_sipp_df = generator.sample_remaining_columns(known_columns=generated_acs_psid_df[columns_to_match],
                                                           max_tries_per_batch=50)

    return pd.concat(
        [generated_acs_psid_df, generated_sipp_df[generated_sipp_df.columns.difference(generated_acs_psid_df.columns)]],
        axis=1)


def generate_for_size(household_size: int, n_samples: int):
    print(f"Generating data for household size {household_size}")

    acs_df = pd.read_csv(DATA_IN_DIRECTORY + "source_ACS/acs_household.csv")
    psid_df = pd.read_csv(DATA_IN_DIRECTORY + "source_PSID/psid_households.csv")
    sipp_df = pd.read_csv(DATA_IN_DIRECTORY + "source_SIPP/sipp_households.csv")

    acs_generator = train_acs_generator(acs_df)
    psid_generator = train_psid_generator(psid_df)
    sipp_generator = train_sipp_generator(sipp_df)

    generated_acs_df = generate_acs_one_size(acs_generator, n_samples)
    generated_psid_df = generate_psid_on_acs_one_size(psid_df.columns, psid_generator, generated_acs_df)
    generated_sipp_df = generate_sipp_on_acs_psid_one_size(sipp_df.columns, sipp_generator, generated_psid_df)

    return generated_sipp_df


def generate_ACS_PSID_SIPP_all():
    sizes_to_generate = get_household_sizes_to_generate(1000, DATA_IN_DIRECTORY + "source_ACS/acs_household.csv")

    n_sample_size_1 = int(sizes_to_generate.loc[1].values[0])
    n_sample_size_2 = int(sizes_to_generate.loc[2].values[0])
    n_sample_size_3plus = int(sizes_to_generate.loc[3:].sum()[0])

    generated_size_1 = generate_for_size(1, n_sample_size_1)
    generated_size_2 = generate_for_size(2, n_sample_size_2)
    generated_size_3plus = generate_for_size(3, n_sample_size_3plus)

    generated_size_2.index += generated_size_1.index.max() + 1
    generated_size_3plus.index += generated_size_2.index.max() + 1

    people_df = pd.concat([generated_size_1, generated_size_2, generated_size_3plus])
    people_df.to_csv("output/people.csv", index=False)


if __name__ == "__main__":
    generate_ACS_PSID_SIPP_all()
    print("Synthetic population generation completed!")
