import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer

DATA_IN_DIRECTORY = "processed_data/synthetic_population_generation/"

columns_in_acs_people_df = list(set([
    "acs_hh_id", "person_id", "employment_status", "relationshipToR", "age", "yearOfBirth",
    "gender", "weight_cross_sectional", "adm1", "adm2", "puma", "race", "educ_years",
    "educ_level", "partnership_status", "elderly_or_disabled", "is_kid", "is_adult",
    "school_type", "degree_field", "occupation_soc", "occupation_naics", "veteran", "citizen",
    "divorce_lstyear", "marriage_lstyear", "fertility_lstyear", "widow_lstyear", "income_wage",
    "income_business", "income_ss", "income_welfare", "income_investment", "income_retirement",
    "income_supss", "income_other", "income_totalearned", "cur_income"
]) - set([
    "relationshipToR", "age_of_retirement"
]) - set(['adm2', 'citizen', 'degree_field', 'divorce_lstyear',
       'fertility_lstyear', 'income_business', 'income_investment',
       'income_other', 'income_retirement', 'income_ss', 'income_supss',
       'income_totalearned', 'income_wage', 'income_welfare',
       'marriage_lstyear', 'occupation_naics', 'occupation_soc', 'puma',
       'relationshipToR', 'relationshipToR ', 'school_type', 'veteran',
       'widow_lstyear', 'yearOfBirth'
]))
columns_in_acs_households_df = [
    "acs_hh_id", "adm1", "puma", "acs_weight_cross_sectional", "num_elderly_or_disabled",
    "num_kids", "num_adults", "household_income", "foodstamp", "couple_type"
]
columns_in_psid_households_df = [
    "psid_hh_id", "partnership_status", "psid_weight_cross_sectional", "active_savings_rate",
    "large_gift", "small_gift", "num_elderly_or_disabled", "num_kids", "num_adults"
]
# Q: relationshipToR and age_of_retirement only contained NaN values.
# Q: filter out columns not in people.csv (check this with Steve)


def read_acs_households(acs_households_filename: str):

    import pandas as pd

    # Read and preprocess acs_household data
    acs_households_df = pd.read_csv(acs_households_filename)
    acs_households_df["acs_hh_id"] = acs_households_df["hh_id"]
    acs_households_df["acs_weight_cross_sectional"] = acs_households_df["weight_cross_sectional"]
    acs_households_df = acs_households_df[columns_in_acs_households_df]
    acs_households_df = acs_households_df[acs_households_df.apply(lambda row: row["household_income"] == row["household_income"], axis=1)]

    acs_households_df["size"] = acs_households_df["num_kids"] + acs_households_df["num_adults"]
    acs_households_df = acs_households_df.drop(columns=[
        "num_kids", "num_adults", "num_elderly_or_disabled"
    ])
    return acs_households_df.dropna() # here, all data should aleady be imputed

def read_acs_people(household_size: int, acs_people_filename: str):

    import pandas as pd
    import numpy as np

    # Read and preprocess acs people data
    acs_people_df = pd.read_csv(acs_people_filename)
    acs_people_df["acs_hh_id"] = acs_people_df["hh_id"]
    acs_people_df = acs_people_df[columns_in_acs_people_df]

    # Q: employment_status, educ_years and educ_level contain nan values.
    # What to do with these? Currently removed from data but we cloud fill in during preprocess.
    # Check with: (acs_people_df.isna()).sum()
    # acs_people_df = acs_people_df.dropna()

    # Drop aggregate columns:
    acs_people_df = acs_people_df.drop(columns=[
        "is_kid", "is_adult", "person_id"
    ])

    # Filter to household size
    acs_hh_ids_of_hh_size = acs_people_df.groupby(acs_people_df["acs_hh_id"]).count().index[acs_people_df.groupby(acs_people_df["acs_hh_id"]).count().max(axis=1) == household_size]
    acs_people_size_df = acs_people_df[acs_people_df["acs_hh_id"].isin(acs_hh_ids_of_hh_size)]

    dict_hh_to_people = {}
    for hh_id in acs_people_size_df["acs_hh_id"].unique():
        dict_hh_to_people[hh_id] = []
    for _, row in acs_people_size_df.iterrows():
        dict_hh_to_people[row["acs_hh_id"]].append(row)
    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1)
    wide_people_df = wide_people_df.transpose()
    wide_people_df.columns = np.concatenate([[f"{column}_{i}" for column in acs_people_df.columns] for i in range(household_size)])
    wide_people_df = wide_people_df.drop(columns=[f"adm1_{i}" for i in range(household_size)])
    wide_people_df["acs_hh_id"] = wide_people_df["acs_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"acs_hh_id_{i}" for i in range(household_size)])

    return wide_people_df.dropna() # here, all data should aleady be imputed

def read_acs_data_of_household_size(household_size: int, acs_households_filename: str, acs_people_filename: str):

    import pandas as pd
    import numpy as np

    # Read and preprocess acs_household data
    acs_households_df = read_acs_households(acs_households_filename)
    acs_people_with_size_df = read_acs_people(household_size, acs_people_filename)

    acs_all_size_df = pd.merge(acs_people_with_size_df, acs_households_df, on="acs_hh_id", how="inner")#.drop(columns=["acs_hh_id"])
    return acs_all_size_df.dropna() # here, all data should aleady be imputed

def read_acs_people_groups(acs_people_filename: str):

    import pandas as pd
    import numpy as np

    # Read and preprocess acs people data
    acs_people_df = pd.read_csv(acs_people_filename)
    acs_people_df["acs_hh_id"] = acs_people_df["hh_id"]
    acs_people_df = acs_people_df[columns_in_acs_people_df]

    # Q: employment_status, acs_people_filename and educ_level contain nan values.
    # What to do with these? Currently removed from data but we cloud fill in during preprocess.
    # Check with: (acs_people_df.isna()).sum()
    # acs_people_df = acs_people_df.dropna()

    for column_to_float in ["age", "educ_years"]:
        acs_people_df[column_to_float] = acs_people_df[column_to_float].astype("float")


    # Drop aggregate columns:
    acs_people_df = acs_people_df.drop(columns=[
        "is_kid", "is_adult", "person_id"
    ])

    # Filter to household size
    acs_hh_ids_of_hh_size = acs_people_df.groupby(acs_people_df["acs_hh_id"]).count().index[acs_people_df.groupby(acs_people_df["acs_hh_id"]).count().max(axis=1) >= 3]
    acs_people_size_df = acs_people_df[acs_people_df["acs_hh_id"].isin(acs_hh_ids_of_hh_size)]

    # Get oldest two
    df_oldest_two = acs_people_size_df.sort_values(['acs_hh_id', 'age'], ascending=[True, False]).groupby('acs_hh_id').head(2)

    dict_hh_to_people = {}
    for hh_id in df_oldest_two["acs_hh_id"].unique():
        dict_hh_to_people[hh_id] = []
    for _, row in df_oldest_two.iterrows():
        dict_hh_to_people[row["acs_hh_id"]].append(row)
    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1)
    wide_people_df = wide_people_df.transpose()
    wide_people_df.columns = np.concatenate([[f"{column}_{i}" for column in acs_people_df.columns] for i in range(2)])
    wide_people_df = wide_people_df.drop(columns=[f"adm1_{i}" for i in range(2)])
    wide_people_df["acs_hh_id"] = wide_people_df["acs_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"acs_hh_id_{i}" for i in range(2)])
    df_younger = acs_people_size_df[~acs_people_size_df.index.isin(df_oldest_two.index)]
    df_younger = df_younger.rename(columns = {column: f"{column}_2" for column in df_younger.columns if column != 'acs_hh_id'})
    df_younger = df_younger.drop(columns=["adm1_2"])
    two_oldest_plus_extra_df = pd.merge(wide_people_df, df_younger, on='acs_hh_id', how='outer')

    return two_oldest_plus_extra_df.dropna() # here, all data should aleady be imputed

def read_acs_data_of_groups(acs_households_filename: str, acs_people_filename: str):

    import pandas as pd
    import numpy as np

    # Read and preprocess acs_household data
    acs_households_df = read_acs_households(acs_households_filename)
    acs_people_with_size_df = read_acs_people_groups(acs_people_filename)

    # Drop aggregate columns but add size column
    acs_all_size_df = pd.merge(acs_people_with_size_df, acs_households_df, on="acs_hh_id", how="inner")
    acs_all_size_df = acs_all_size_df.dropna() # here, all data should aleady be imputed

    acs_all_size_first_two = acs_all_size_df[[col for col in acs_all_size_df.columns if col[-2:] != "_2"]]
    acs_all_size_first_two = acs_all_size_first_two[~acs_all_size_first_two.index.duplicated()]

    acs_all_size_df = acs_all_size_df.drop(columns=["acs_hh_id"])
    acs_all_size_first_two = acs_all_size_first_two.drop(columns=["acs_hh_id"])
    return acs_all_size_df, acs_all_size_first_two

def read_psid_holdings_csv(psid_holdings_filename: str):
    import pandas as pd

    assets_df = pd.read_csv(psid_holdings_filename)
    assets_df = assets_df[["hh_id", "asset_type", "curr_value"]]
    asset_types = assets_df["asset_type"].unique()
    assets_wide_df = pd.DataFrame(columns = asset_types)

    # Get all asset types and convert to seperate columns for each asset.
    for hh_id in assets_df["hh_id"].unique():
        assets_for_hh_df = assets_df[assets_df["hh_id"] == hh_id]
        assets_for_hh_array = [assets_for_hh_df[assets_for_hh_df["asset_type"] == asset_type].values[-1][2] if len(assets_for_hh_df[assets_for_hh_df["asset_type"] == asset_type]) > 0 else 0 for asset_type in asset_types]
        assets_wide_df.loc[hh_id] = assets_for_hh_array
    assets_wide_df = assets_wide_df.fillna(0)

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    assets_wide_df = assets_wide_df.reset_index().rename(columns={"index": "psid_hh_id"})
    return assets_wide_df.dropna() # here, all data should aleady be imputed

def read_psid_households_csv(psid_households_filename: str):
    import pandas as pd
    import numpy as np

    psid_households_df = pd.read_csv(psid_households_filename)

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    psid_households_df = psid_households_df.rename(columns={"hh_id": "psid_hh_id"})
    psid_households_df["psid_weight_cross_sectional"] = psid_households_df["weight_cross_sectional"]
    psid_households_df = psid_households_df[columns_in_psid_households_df]

    # Fill missing values with 0
    psid_households_df["small_gift"] = psid_households_df["small_gift"].fillna(0)
    psid_households_df["large_gift"] = psid_households_df["large_gift"].fillna(0)

    # Remove #NAME? from active savings rate
    psid_households_df = psid_households_df.replace("#NAME?", np.nan).dropna()
    psid_households_df = psid_households_df.replace("inf", np.nan).dropna()
    psid_households_df["active_savings_rate"] = psid_households_df["active_savings_rate"].apply(lambda x: float(x))

    # Remove calculatable columns
    psid_households_df["size"] = psid_households_df["num_kids"] + psid_households_df["num_adults"]
    psid_households_df = psid_households_df.drop(columns=["num_elderly_or_disabled", "num_kids", "num_adults"])
    return psid_households_df.dropna() # here, all data should aleady be imputed

def read_psid_people_csv(household_size: int, psid_people_filename: str):
    import pandas as pd
    import numpy as np

    psid_people_df = pd.read_csv(psid_people_filename)
    psid_people_df["years_partnered"] = psid_people_df["years_partnered"].fillna(0)
    psid_people_df["age_of_retirement"] = psid_people_df["age_of_retirement"].fillna(0)
    psid_people_df["cur_income"] = psid_people_df["cur_income"].fillna(0)
    psid_people_df["startMostRecentMarriage"] = psid_people_df["startMostRecentMarriage"].fillna(0)
    psid_people_df["cur_income"] = psid_people_df["cur_income"].fillna(0)
    # Q: Default partnership status?

    # Remove semi-inferrable columns and duplicate columns: yearOfBirthI
    psid_people_df = psid_people_df.drop(columns=["yearOfBirthI", "adm1_int"])

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    psid_people_df = psid_people_df.rename(columns={"hh_id": "psid_hh_id"})

    # Partnership status currently removed, mapping between spid and acs needs to be created.
    psid_people_df = psid_people_df.drop(columns=["partnership_status"])

    # Rename to be able to merge with acs
    psid_people_df["gender"] = psid_people_df["gender"].replace("F", "Female").replace("M", "Male")

    # Drop duplicate and aggregate columns:
    psid_people_df = psid_people_df.drop(columns=[
        "is_kid", "is_adult"
    ])

    # Q: Keys are not comparable, what to do?
    psid_people_df = psid_people_df.drop(columns=[
        "employment_status"
    ])
    psid_people_df = psid_people_df.dropna()

    # Filter to household size
    psid_hh_ids_of_hh_size = psid_people_df.groupby(psid_people_df["psid_hh_id"]).count().index[psid_people_df.groupby(psid_people_df["psid_hh_id"]).count().max(axis=1) == household_size]
    psid_people_size_df = psid_people_df[psid_people_df["psid_hh_id"].isin(psid_hh_ids_of_hh_size)]

    dict_hh_to_people = {}
    for hh_id in psid_people_size_df["psid_hh_id"].unique():
        dict_hh_to_people[hh_id] = []
    for _, row in psid_people_size_df.iterrows():
        dict_hh_to_people[row["psid_hh_id"]].append(row)
    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1)
    wide_people_df = wide_people_df.transpose()
    wide_people_df.columns = np.concatenate([[f"{column}_{i}" for column in psid_people_df.columns] for i in range(household_size)])

    # remove unneeded (duplicate) rows
    wide_people_df = wide_people_df.drop(columns=[f"person_id_{i}" for i in range(household_size)])
    wide_people_df["psid_hh_id"] = wide_people_df["psid_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"psid_hh_id_{i}" for i in range(household_size)])
    wide_people_df["adm1"] = wide_people_df["adm1_0"]
    wide_people_df = wide_people_df.drop(columns=[f"adm1_{i}" for i in range(household_size)])

    return wide_people_df.dropna() # here, all data should aleady be imputed

def read_psid_people_groups(psid_people_filename: str):
    import pandas as pd
    import numpy as np
    psid_people_filename = DATA_IN_DIRECTORY + "source_PSID/psid_people.csv"
    psid_people_df = pd.read_csv(psid_people_filename)
    psid_people_df["years_partnered"] = psid_people_df["years_partnered"].fillna(0)
    psid_people_df["age_of_retirement"] = psid_people_df["age_of_retirement"].fillna(0)
    psid_people_df["cur_income"] = psid_people_df["cur_income"].fillna(0)
    psid_people_df["startMostRecentMarriage"] = psid_people_df["startMostRecentMarriage"].fillna(0)
    psid_people_df["cur_income"] = psid_people_df["cur_income"].fillna(0)
    # Q: Default partnership status?

    for column_to_float in ["age", "educ_years", "total_social_security_contributions", "years_of_social_security_contributions"]:
        psid_people_df[column_to_float] = psid_people_df[column_to_float].astype("float")


    # Remove semi-inferrable columns and duplicate columns: yearOfBirthI
    psid_people_df = psid_people_df.drop(columns=["yearOfBirthI", "adm1_int"])

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    psid_people_df = psid_people_df.rename(columns={"hh_id": "psid_hh_id"})

    # Partnership status currently removed, mapping between spid and acs needs to be created.
    psid_people_df = psid_people_df.drop(columns=["partnership_status"])

    # Rename to be able to merge with acs
    psid_people_df["gender"] = psid_people_df["gender"].replace("F", "Female").replace("M", "Male")

    # Drop duplicate and aggregate columns:
    psid_people_df = psid_people_df.drop(columns=[
        "is_kid", "is_adult"
    ])

    # Q: Keys are not comparable, what to do?
    psid_people_df = psid_people_df.drop(columns=[
        "employment_status"
    ])
    psid_people_df = psid_people_df.dropna()

    # Filter to household size
    psid_hh_ids_of_hh_size = psid_people_df.groupby(psid_people_df["psid_hh_id"]).count().index[psid_people_df.groupby(psid_people_df["psid_hh_id"]).count().max(axis=1) >= 3]
    psid_people_size_df = psid_people_df[psid_people_df["psid_hh_id"].isin(psid_hh_ids_of_hh_size)]

    # Get oldest two
    df_oldest_two = psid_people_size_df.sort_values(['psid_hh_id', 'age'], ascending=[True, False]).groupby('psid_hh_id').head(2)

    dict_hh_to_people = {}
    for hh_id in df_oldest_two["psid_hh_id"].unique():
        dict_hh_to_people[hh_id] = []
    for _, row in df_oldest_two.iterrows():
        dict_hh_to_people[row["psid_hh_id"]].append(row)
    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1)
    wide_people_df = wide_people_df.transpose()
    wide_people_df.columns = np.concatenate([[f"{column}_{i}" for column in psid_people_df.columns] for i in range(2)])
    wide_people_df = wide_people_df.drop(columns=[f"person_id_{i}" for i in range(2)])
    wide_people_df["psid_hh_id"] = wide_people_df["psid_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"psid_hh_id_{i}" for i in range(2)])
    wide_people_df["adm1"] = wide_people_df["adm1_0"]
    wide_people_df = wide_people_df.drop(columns=[f"adm1_{i}" for i in range(2)])

    df_younger = psid_people_size_df[~psid_people_size_df.index.isin(df_oldest_two.index)]
    df_younger = df_younger.rename(columns = {column: f"{column}_2" for column in df_younger.columns if column != 'psid_hh_id'})
    df_younger = df_younger.drop(columns=["adm1_2", "person_id_2"])

    two_oldest_plus_extra_df = pd.merge(wide_people_df, df_younger, on='psid_hh_id', how='outer')
    return two_oldest_plus_extra_df.dropna() # here, all data should aleady be imputed

def read_psid_data_of_groups(psid_households_filename: str, psid_people_filename: str, psid_holdings_filename: str):
    import pandas as pd
    import numpy as np

    # Read psid files
    psid_households_df = read_psid_households_csv(psid_households_filename)
    psid_people_with_size_df = read_psid_people_groups(psid_people_filename)
    psid_holdings_df = read_psid_holdings_csv(psid_holdings_filename)

    # Merge psid people and households
    psid_all_size_df = pd.merge(psid_people_with_size_df, psid_households_df, on="psid_hh_id", how="inner")

    # Also merge with holdings
    psid_all_size_df = pd.merge(psid_all_size_df, psid_holdings_df, on="psid_hh_id", how="inner")
    psid_all_size_df = psid_all_size_df.dropna() # here, all data should aleady be imputed

    psid_all_size_first_two = psid_all_size_df[[col for col in psid_all_size_df.columns if col[-2:] != "_2"]]
    psid_all_size_first_two = psid_all_size_first_two[~psid_all_size_first_two.index.duplicated()]

    psid_all_size_df = psid_all_size_df.drop(columns=["psid_hh_id"])
    psid_all_size_first_two = psid_all_size_first_two.drop(columns=["psid_hh_id"])
    return psid_all_size_df, psid_all_size_first_two

def read_psid_data_of_household_size(household_size: int, psid_households_filename: str, psid_people_filename: str, psid_holdings_filename: str):
    import pandas as pd
    import numpy as np

    # Read psid files
    psid_households_df = read_psid_households_csv(psid_households_filename)
    psid_people_with_size_df = read_psid_people_csv(household_size, psid_people_filename)
    psid_holdings_df = read_psid_holdings_csv(psid_holdings_filename)

    # Merge psid people and households
    psid_all_size_df = pd.merge(psid_people_with_size_df, psid_households_df, on="psid_hh_id", how="inner")

    # Also merge with holdings
    psid_all_size_df = pd.merge(psid_all_size_df, psid_holdings_df, on="psid_hh_id", how="inner").drop(columns=["psid_hh_id"])

    return psid_all_size_df.dropna() # here, all data should aleady be imputed

def read_sipp_households(sipp_households_filename: str):

    import pandas as pd
    sipp_households_df = pd.read_csv(sipp_households_filename)

    # drop columns wiht only nan values
    sipp_households_df = sipp_households_df.drop(columns=["active_savings_rate", "large_gift", "small_gift"])

    # drop rows with nan values for foodstamps
    sipp_households_df = sipp_households_df.dropna()

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    sipp_households_df = sipp_households_df.rename(columns={"hh_id": "sipp_hh_id"})

    # Replaceing True and False by 2 and 1 (I guess this are what the foodstamps in acs_households.csv mean)
    # Q: Is this correct?
    sipp_households_df = sipp_households_df.replace(True, 2).replace(False, 1)

    # Remove calculateable fields
    sipp_households_df["size"] = sipp_households_df["num_kids"] + sipp_households_df["num_adults"]
    sipp_households_df = sipp_households_df.drop(columns=[
        "num_kids", "num_adults", "num_elderly_or_disabled"
    ])

    # Drop adm1, as it only contains zeroes
    sipp_households_df = sipp_households_df.drop(columns=["adm1"])

    return sipp_households_df.dropna() # here, all data should aleady be imputed

def read_sipp_holdings(sipp_holdings_filename: str):

    import pandas as pd
    sipp_holdings_df = pd.read_csv(sipp_holdings_filename)
    sipp_holdings_df = sipp_holdings_df[["hh_id", "asset_type", "curr_value"]]
    asset_types = sipp_holdings_df["asset_type"].unique()
    assets_wide_df = pd.DataFrame(columns = asset_types)

    # Get all asset types and convert to seperate columns for each asset.
    for hh_id in sipp_holdings_df["hh_id"].unique():
        assets_for_hh_df = sipp_holdings_df[sipp_holdings_df["hh_id"] == hh_id]
        assets_for_hh_array = [assets_for_hh_df[assets_for_hh_df["asset_type"] == asset_type].values[-1][2] if len(assets_for_hh_df[assets_for_hh_df["asset_type"] == asset_type]) > 0 else 0 for asset_type in asset_types]
        assets_wide_df.loc[hh_id] = assets_for_hh_array
    assets_wide_df = assets_wide_df.fillna(0)

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    assets_wide_df = assets_wide_df.reset_index().rename(columns={"index": "sipp_hh_id"})
    return assets_wide_df.dropna() # here, all data should aleady be imputed

def read_sipp_people(household_size: int, sipp_people_filename: str):

    import pandas as pd
    import numpy as np

    sipp_people_df = pd.read_csv(sipp_people_filename)

    # fill nan values with zero, or no school (for which I suspect tha- set([column for column in generated_acs_psid_one_size_df.columns if "weight" in column])t missing values mean zero)
    # Q: is this correct?
    sipp_people_df[[col for col in sipp_people_df if "benefits_" in col]] = sipp_people_df[[col for col in sipp_people_df if "benefits_" in col]].fillna(0)
    sipp_people_df["cur_income"] = sipp_people_df["cur_income"].fillna(0)
    sipp_people_df["school_type"] = sipp_people_df["school_type"].fillna("no school")

    # For these, I'm not confident to guess a default value, skipping
    # Q: which ones can be filled to what default values, and which one need to be filtered out?
    # taxes_filed, taxes_filing_status, retired, taxes_eitc, currently_student, partnership_status, employment_status
    sipp_people_df = sipp_people_df.fillna(0)

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    sipp_people_df = sipp_people_df.rename(columns={"hh_id": "sipp_hh_id"})

    # Filter to household size
    sipp_hh_ids_of_hh_size = sipp_people_df.groupby(sipp_people_df["sipp_hh_id"]).count().index[sipp_people_df.groupby(sipp_people_df["sipp_hh_id"]).count().max(axis=1) == household_size]
    sipp_people_size_df = sipp_people_df[sipp_people_df["sipp_hh_id"].isin(sipp_hh_ids_of_hh_size)]

    dict_hh_to_people = {}
    for hh_id in sipp_people_size_df["sipp_hh_id"].unique():
        dict_hh_to_people[hh_id] = []
    for _, row in sipp_people_size_df.iterrows():
        dict_hh_to_people[row["sipp_hh_id"]].append(row)
    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1)
    wide_people_df = wide_people_df.transpose()
    wide_people_df.columns = np.concatenate([[f"{column}_{i}" for column in sipp_people_df.columns] for i in range(household_size)])

    # remove unneeded (duplicate) rows
    wide_people_df = wide_people_df.drop(columns=[f"person_id_{i}" for i in range(household_size)])
    wide_people_df["sipp_hh_id"] = wide_people_df["sipp_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"sipp_hh_id_{i}" for i in range(household_size)])
    # Only Some High School? No other variables, thus skipping
    wide_people_df = wide_people_df.drop(columns=[f"educ_level_{i}" for i in range(household_size)])

    return wide_people_df.dropna() # here, all data should aleady be imputed

def read_sipp_people_groups(sipp_people_filename: str):
    import pandas as pd
    import numpy as np

    sipp_people_df = pd.read_csv(sipp_people_filename)

    # fill nan values with zero, or no school (for which I suspect tha- set([column for column in generated_acs_psid_one_size_df.columns if "weight" in column])t missing values mean zero)
    # Q: is this correct?
    sipp_people_df[[col for col in sipp_people_df if "benefits_" in col]] = sipp_people_df[[col for col in sipp_people_df if "benefits_" in col]].fillna(0)
    sipp_people_df["cur_income"] = sipp_people_df["cur_income"].fillna(0)
    sipp_people_df["school_type"] = sipp_people_df["school_type"].fillna("no school")

    # For these, I'm not confident to guess a default value, skipping
    # Q: which ones can be filled to what default values, and which one need to be filtered out?
    # taxes_filed, taxes_filing_status, retired, taxes_eitc, currently_student, partnership_status, employment_status
    sipp_people_df = sipp_people_df.fillna(0)
    
    for column_to_float in ["yearOfBirth"]:
        sipp_people_df[column_to_float] = sipp_people_df[column_to_float].astype("float")

    # Rename hh_id to only match with psid and not mistakenly match with acs data
    sipp_people_df = sipp_people_df.rename(columns={"hh_id": "sipp_hh_id"})

    # Filter to household size
    sipp_hh_ids_of_hh_size = sipp_people_df.groupby(sipp_people_df["sipp_hh_id"]).count().index[sipp_people_df.groupby(sipp_people_df["sipp_hh_id"]).count().max(axis=1) >= 3]
    sipp_people_size_df = sipp_people_df[sipp_people_df["sipp_hh_id"].isin(sipp_hh_ids_of_hh_size)]

    # Get oldest two
    df_oldest_two = sipp_people_size_df.sort_values(['sipp_hh_id', 'age'], ascending=[True, False]).groupby('sipp_hh_id').head(2)

    dict_hh_to_people = {}
    for hh_id in df_oldest_two["sipp_hh_id"].unique():
        dict_hh_to_people[hh_id] = []
    for _, row in df_oldest_two.iterrows():
        dict_hh_to_people[row["sipp_hh_id"]].append(row)
    wide_people_df = pd.concat([pd.concat(dict_hh_to_people[hh_id]) for hh_id in dict_hh_to_people], axis=1)
    wide_people_df = wide_people_df.transpose()
    wide_people_df.columns = np.concatenate([[f"{column}_{i}" for column in sipp_people_df.columns] for i in range(2)])

    # remove unneeded (duplicate) rows
    wide_people_df = wide_people_df.drop(columns=[f"person_id_{i}" for i in range(2)])
    wide_people_df["sipp_hh_id"] = wide_people_df["sipp_hh_id_0"]
    wide_people_df = wide_people_df.drop(columns=[f"sipp_hh_id_{i}" for i in range(2)])
    # Only Some High School? No other variables, thus skipping
    wide_people_df = wide_people_df.drop(columns=[f"educ_level_{i}" for i in range(2)])
    # Need translation between different codings
    wide_people_df = wide_people_df.drop(columns=[f"employment_status_{i}" for i in range(2)])

    df_younger = sipp_people_size_df[~sipp_people_size_df.index.isin(df_oldest_two.index)]
    df_younger = df_younger.rename(columns = {column: f"{column}_2" for column in df_younger.columns if column != 'sipp_hh_id'})
    df_younger = df_younger.drop(columns=[f"person_id_2"])
    # Only Some High School? No other variables, thus skipping
    df_younger = df_younger.drop(columns=["educ_level_2"])
    # Need translation between different codings
    df_younger = df_younger.drop(columns=["employment_status_2"])

    two_oldest_plus_extra_df = pd.merge(wide_people_df, df_younger, on='sipp_hh_id', how='outer')
    return two_oldest_plus_extra_df.dropna() # here, all data should aleady be imputed
    
def read_sipp_data_of_household_size(household_size: int, sipp_households_filename: str, sipp_people_filename: str, sipp_holdings_filename: str):
    import pandas as pd

    sipp_households = read_sipp_households(sipp_households_filename)
    sipp_people_with_size_df = read_sipp_people(household_size, sipp_people_filename)
    sipp_holdings = read_sipp_holdings(sipp_holdings_filename)

    # Merge psid people and households
    sipp_all_size_df = pd.merge(sipp_people_with_size_df, sipp_households, on="sipp_hh_id", how="inner")

    # Also merge with holdings
    sipp_all_size_df = pd.merge(sipp_all_size_df, sipp_holdings, on="sipp_hh_id", how="inner").drop(columns=["sipp_hh_id"])

    return sipp_all_size_df.dropna() # here, all data should aleady be imputed

def read_sipp_data_of_groups(sipp_households_filename: str, sipp_people_filename: str, sipp_holdings_filename: str):
    import pandas as pd
    import numpy as np

    # Read psid files
    sipp_households = read_sipp_households(sipp_households_filename)
    sipp_people_with_size_df = read_sipp_people_groups(sipp_people_filename)
    sipp_holdings = read_sipp_holdings(sipp_holdings_filename)

    # Drop duplicate id's
    sipp_people_with_size_df.drop_duplicates(subset = ['sipp_hh_id'], keep = 'first', inplace = True) 
    sipp_households.drop_duplicates(subset = ['sipp_hh_id'], keep = 'first', inplace = True) 

    # Merge psid people and households
    sipp_all_size_df = pd.merge(sipp_people_with_size_df, sipp_households, on="sipp_hh_id", how="inner")

    # Also merge with holdings
    sipp_all_size_df = pd.merge(sipp_all_size_df, sipp_holdings, on="sipp_hh_id", how="inner")
    sipp_all_size_df = sipp_all_size_df.dropna() # here, all data should aleady be imputed

    sipp_all_size_first_two = sipp_all_size_df[[col for col in sipp_all_size_df.columns if col[-2:] != "_2"]]
    sipp_all_size_first_two = sipp_all_size_first_two[~sipp_all_size_first_two.index.duplicated()]

    sipp_all_size_df = sipp_all_size_df.drop(columns=["sipp_hh_id"])
    sipp_all_size_first_two = sipp_all_size_first_two.drop(columns=["sipp_hh_id"])
    return sipp_all_size_df, sipp_all_size_first_two

def get_household_sizes_to_generate(sample_size: int, acs_households_filename: str) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    expected_counts_per_household_size = pd.read_csv(acs_households_filename)
    expected_counts_per_household_size["size"] = expected_counts_per_household_size["num_kids"] + expected_counts_per_household_size["num_adults"]
    expected_counts_per_household_size = expected_counts_per_household_size[["weight_cross_sectional", "size"]]
    expected_counts_per_household_size = expected_counts_per_household_size.groupby(expected_counts_per_household_size["size"]).sum()
    expected_counts_per_household_size = expected_counts_per_household_size / expected_counts_per_household_size.sum() * sample_size

        # Round to integers
    rounded_counts = expected_counts_per_household_size.round()

    # Adjust to ensure sum matches sample_size
    total_difference = int(sample_size - rounded_counts.sum().values[0])  # Get rounding error
    if total_difference != 0:
        # Compute the fractional part and sort by largest errors
        expected_counts_per_household_size["fraction"] = expected_counts_per_household_size["weight_cross_sectional"] - rounded_counts["weight_cross_sectional"]
        expected_counts_per_household_size = expected_counts_per_household_size.sort_values("fraction", ascending=False)

        # Distribute the difference by adjusting the largest fractional errors first
        indices = expected_counts_per_household_size.index.tolist()
        for i in range(abs(total_difference)):
            idx = indices[i % len(indices)]
            rounded_counts.at[idx, "weight_cross_sectional"] += np.sign(total_difference)  # Adjust up/down

    return rounded_counts.drop(columns=["fraction"], errors="ignore").astype(int)

def train_acs_generator(acs_dataframe: pd.DataFrame) -> TVAESynthesizer:

    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import Metadata
    from rdt.transformers.categorical import LabelEncoder

    acs_metadata = Metadata.detect_from_dataframe(
        data=acs_dataframe,
        table_name='acs_all')

    acs_metadata.update_column(column_name='adm1', sdtype='categorical')
    acs_metadata.update_column(column_name='puma', sdtype='categorical')

    acs_generator_one_size = TVAESynthesizer(metadata=acs_metadata, epochs=40, verbose=True)
    acs_generator_one_size.auto_assign_transformers(acs_dataframe)
    acs_generator_one_size.update_transformers({
        'puma': LabelEncoder(add_noise=False),
        'adm1': LabelEncoder(add_noise=False)
    })
    acs_generator_one_size.fit(acs_dataframe)

    return acs_generator_one_size

def train_acs_generator_groups(acs_dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:

    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata
    from rdt.transformers.categorical import LabelEncoder

    acs_metadata = Metadata.detect_from_dataframe(
        data=acs_dataframe,
        table_name='acs_all')

    acs_metadata.update_column(column_name='adm1', sdtype='categorical')
    acs_metadata.update_column(column_name='puma', sdtype='categorical')

    acs_generator_groups = GaussianCopulaSynthesizer(metadata=acs_metadata, enforce_min_max_values=False)
    acs_generator_groups.auto_assign_transformers(acs_dataframe)
    acs_generator_groups.update_transformers({
        'puma': LabelEncoder(add_noise=False),
        'adm1': LabelEncoder(add_noise=False)
    })
    acs_generator_groups.fit(acs_dataframe)

    return acs_generator_groups

def train_psid_generator(psid_dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:

    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata
    from rdt.transformers.categorical import LabelEncoder

    psid_metadata = Metadata.detect_from_dataframe(
        data=psid_dataframe,
        table_name='psid_all')

    psid_metadata.update_column(column_name='adm1', sdtype='categorical')
    for columns_years_partnered in [column for column in psid_dataframe.columns if "years_parntered" in column]:
        psid_metadata.update_column(column_name=columns_years_partnered, sdtype='numerical')

    psid_generator_one_size = GaussianCopulaSynthesizer(metadata=psid_metadata, enforce_min_max_values=False)
    psid_generator_one_size.auto_assign_transformers(psid_dataframe)
    psid_generator_one_size.update_transformers({
        'adm1': LabelEncoder(add_noise=False)
    })
    psid_generator_one_size.fit(psid_dataframe)
    return psid_generator_one_size

def train_sipp_generator(sipp_dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:

    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata
    from rdt.transformers.categorical import LabelEncoder

    sipp_metadata = Metadata.detect_from_dataframe(
        data=sipp_dataframe,
        table_name='sipp_all')

    # Set correct value type
    for column_benefit in [column for column in sipp_dataframe.columns if "benefits_" in column]:
        sipp_metadata.update_column(column_name=column_benefit, sdtype='numerical')
    for column_educ_years in [column for column in sipp_dataframe.columns if "educ_years" in column]:
        sipp_metadata.update_column(column_name=column_educ_years, sdtype='numerical')
    for age_column in [column for column in sipp_dataframe.columns if "age" in column]:
        sipp_metadata.update_column(column_name=age_column, sdtype='numerical')
    for column_asset in ['CheckingAndSavings', 'DCRetirePlan', 'BrokerageStocks', 'OtherRealEstate', 'OtherAssets', 'Business', 'EducationSavings', 'House', 'OtherDebts']:
        sipp_metadata.update_column(column_name=column_asset, sdtype='numerical')
    #sipp_metadata.update_column(column_name="adm1", sdtype='categorical')
    #Add above line if adm1 is added back to sipp household files
    
    sipp_generator_one_size = GaussianCopulaSynthesizer(metadata=sipp_metadata, enforce_min_max_values=False)
    sipp_generator_one_size.auto_assign_transformers(sipp_dataframe)
    #sipp_generator_one_size.update_transformers({
    #    'adm1': LabelEncoder(add_noise=False)
    #})
    # Add above line if adm1 is added back
    sipp_generator_one_size.fit(sipp_dataframe)
    return sipp_generator_one_size

def generate_acs_one_size(acs_generator_one_size: TVAESynthesizer, sample_size_for_household_size: int) -> pd.DataFrame:
        
    print(f"generating acs with Variatonal Auto Encoder-Generator")

    generated_acs_one_size_df = acs_generator_one_size.sample(sample_size_for_household_size)

    return generated_acs_one_size_df

def generate_psid_on_acs_one_size(psid_columns, psid_generator_one_size: GaussianCopulaSynthesizer, generated_acs_one_size_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_match_psid_on_acs = list(set(generated_acs_one_size_df.columns) & set(psid_columns) # TODO these columns
                                    # Don't match on weigth (probably not needed and makes the generation significantly slower)
                                    - set([column for column in generated_acs_one_size_df.columns if "weight" in column]))
    print(f"generating psid based on acs columns: {columns_to_match_psid_on_acs} with Gaussian Copula-Generator")

    # Generation goes fast enough to include all columns!
    generated_psid_one_size_df = psid_generator_one_size.sample_remaining_columns(
        known_columns=generated_acs_one_size_df[columns_to_match_psid_on_acs],
        max_tries_per_batch=20
    )

    # Merge the new columns into the dataframe
    generated_acs_psid_one_size_df = pd.concat([
            generated_acs_one_size_df,
            generated_psid_one_size_df[[col for col in generated_psid_one_size_df.columns if col not in generated_acs_one_size_df.columns]]
        ], axis=1
    )
    return generated_acs_psid_one_size_df

def generate_sipp_on_acs_psid_one_size(sipp_columns, sipp_generator_one_size: GaussianCopulaSynthesizer, generated_acs_psid_one_size_df: pd.DataFrame) -> pd.DataFrame:
 
    columns_to_match_sipp_on_psid_and_acs = list(set(generated_acs_psid_one_size_df.columns) & set(sipp_columns)
                                    # Don't match on weigth (probably not needed and makes the generation significantly slower)
                                    - set([column for column in generated_acs_psid_one_size_df.columns if "weight" in column])
                                    # Need a good mapping from employment statusses of different datasets.
                                    # Replace to a uniform scheme during file reading, and uncomment the line to start using the employment statusses and education levels.
                                    - set(["employment_status_0", "employment_status_1"])
                                    - set(["educ_level_0", "educ_level_1"])
                                    #- set(["educ_years_0", "educ_years_1", "partnership_status"])
                                    
                                    #- set (["relationshipToR_0", "relationshipToR_1"])
                                    # Don't match on assets, these numerical fields have so much possibilities that matching becomes extremely slow.
                                    # This is something known in the field and extremely difficult to fix since combinatorical complexity of the challenge.
                                    # If we can assume (or show) independence of all variables in SIPP from, for example the "Business" asset,
                                    # we can indeed decide not to match on these variables.
                                    #- set(["CheckingAndSavings", "Business", "House"])
                                    - set(["DCRetirePlan", "BrokerageStocks", "OtherRealEstate", "OtherDebts", "OtherAssets"])
                                    - set(["household_income", "cur_income_0", "cur_income_1"])
                                    ) # Household income we might be able to skip if we would match on cur_income_0 and cur_income_1
    print(f"generating sipp based on acs and psid columns: {columns_to_match_sipp_on_psid_and_acs} with Gaussian Copula-Generator")

    # Q: some categorical values existing in psid&acs data, do not exist in the sipp data. What mapping should we use?
    # For example, in sipp data, there's no "Some College", should we rename to "College"? Removed for now, let me know.
    # For Race, NativeAmerican is unknown in the SIPP data. Also employment status I need to know how to translate
    # The most I can guess (for example EmploymentStatus.EMPLOYED should be Employed, but EmploymentStatus.COLLEGE_STUDENT, should it be Unemployed?)
    # Also, the SIPP data doesn't span all adm1 codes and relationshipToR_0.
    # If the SIPP data doesnt have the adm1 code, it selects a random adm1 code (slow) to be able to fill in the data.
    generated_sipp_one_size_df = sipp_generator_one_size.sample_remaining_columns(
        known_columns=generated_acs_psid_one_size_df[columns_to_match_sipp_on_psid_and_acs],
        max_tries_per_batch=50
    )

    # Note: Generation is quite slow, and starts for me after ~1 minute.
    # Q: I'm interested in how fast it runs for you and how  much time you are willing to let it compute.


    # Merge the new columns into the dataframe
    generated_acs_psid_sipp_one_size_df = pd.concat([
            generated_acs_psid_one_size_df,
            generated_sipp_one_size_df[[col for col in generated_sipp_one_size_df.columns if col not in generated_acs_psid_one_size_df.columns]]
        ], axis=1
    )
    return generated_acs_psid_sipp_one_size_df

def extract_people_one_size_csv(household_size: int, generated_acs_psid_one_size: pd.DataFrame):
    import pandas as pd
    columns_in_people_df = [
        "hh_id", "person_id", "sequenceNoI_2019", "employment_status", "relationshipToR_2019",
        "age", "yearOfBirthI_2019", "gender", "startMostRecentMarriage", "weight_cross_sectional",
        "weight_longitudinal", "adm1", "gender_hh", "age_fromhh", "race", "educ_years",
        "employment_status_fromhh", "partnership_status", "cur_income", "partner_income",
        "total_hh_income", "age_ofSpouse", "race_ofSpouse", "adm1_rp", "race_rp", "total_hh_income_rp",
        "cur_income_rp", "partner_income_rp", "adm1_from_hh", "adm1_int", "educ_level",
        "age_of_retirement", "total_social_security_contributions", "years_of_social_security_contributions",
        "clipped_income", "years_partnered", "elderly_or_disabled", "is_kid", "is_adult", "rp_percent_of_total",
        "spouse_percent_of_total", "rp_and_spouse_percent_of_total"
    ]
    columns_out = [column for column in generated_acs_psid_one_size.columns
                                            if any([column_in_people_df == column[:-2] for column_in_people_df in columns_in_people_df])]

    people_df = generated_acs_psid_one_size[columns_out]
    people_df["hh_id"] = people_df.index
    people_out_dfs = []
    for person_id in range(household_size): # TODO: fix different household sizes
        people_out_rows = []
        for _, row in people_df.iterrows():
            row_out = row[[col for col in columns_out if col[-2:] == f"_{person_id}"] + ["hh_id"]]
            people_out_rows.append(row_out)
        people_out_df = pd.DataFrame(people_out_rows)
        people_out_df = people_out_df.rename(columns={col: col[:-2] for col in people_out_df if col != "hh_id"})
        people_out_df["hh_id"] = people_out_df["hh_id"].apply(lambda x: int(x))
        people_out_dfs.append(people_out_df)
    people_out_df = pd.concat(people_out_dfs, axis=0)
    return people_out_df

def extract_households_csv(generated_acs_psid_one_size: pd.DataFrame):
    # Todo: calculate num_kids, num_adults and make more large_gift and small_gift zero by creating a categorical column has_large_gift and has_small_gift
    columns_in_households_df = [
        "hh_id", "adm1", "num_adults_orig", "num_kids_orig", "partnership_status",
        "weight_longitudinal", "weight_cross_sectional", "active_savings_rate",
        "large_gift", "small_gift", "adm1_int", "num_elderly_or_disabled", "num_kids", "num_adults"
    ]
    columns_out = [column for column in generated_acs_psid_one_size.columns if column in columns_in_households_df]

    household_df = generated_acs_psid_one_size[columns_out]
    household_df["num_kids"] = generated_acs_psid_one_size[[column for column in generated_acs_psid_one_size if column[:3] == "age" and len(column) == 5]].apply(lambda x: sum(x <= 17), axis=1)
    household_df["num_adults"] = generated_acs_psid_one_size[[column for column in generated_acs_psid_one_size if column[:3] == "age" and len(column) == 5]].apply(lambda x: sum(x >= 18), axis=1)
    household_df["large_gift"] = household_df["large_gift"].apply(lambda x: round(max(x, 0), -2))
    household_df["small_gift"] = household_df["small_gift"].apply(lambda x: round(max(x, 0), 2))
    return household_df

def extract_people_groups_csv(extra_persons_df):
    import pandas as pd
    columns_in_people_df = [
        "hh_id", "person_id", "sequenceNoI_2019", "employment_status", "relationshipToR_2019",
        "age", "yearOfBirthI_2019", "gender", "startMostRecentMarriage", "weight_cross_sectional",
        "weight_longitudinal", "adm1", "gender_hh", "age_fromhh", "race", "educ_years",
        "employment_status_fromhh", "partnership_status", "cur_income", "partner_income",
        "total_hh_income", "age_ofSpouse", "race_ofSpouse", "adm1_rp", "race_rp", "total_hh_income_rp",
        "cur_income_rp", "partner_income_rp", "adm1_from_hh", "adm1_int", "educ_level",
        "age_of_retirement", "total_social_security_contributions", "years_of_social_security_contributions",
        "clipped_income", "years_partnered", "elderly_or_disabled", "is_kid", "is_adult", "rp_percent_of_total",
        "spouse_percent_of_total", "rp_and_spouse_percent_of_total"
    ]
    columns_out = [column for column in extra_persons_df.columns
                                            if any([column_in_people_df == column[:-2] for column_in_people_df in columns_in_people_df])]
    people_df = extra_persons_df[columns_out]
    people_df = people_df.rename(columns={col: col[:-2] for col in people_df if col != "hh_id" and col[-2:] == "_2"})
    return people_df

def extract_holdings_csv(generated_acs_psid_one_size: pd.DataFrame):
    asset_types = [
        'Mortgage', 'House',
        'OtherRealEstate', 'Business', 'BrokerageStocks', 'CheckingAndSavings',
        'Vehicle', 'OtherAssets', 'OtherDebts', 'DBRetirePlan', 'DCRetirePlan'
    ]
    columns_out = [column for column in generated_acs_psid_one_size.columns if column in asset_types]
    holdings_wide_df = generated_acs_psid_one_size[columns_out]
    holdings_wide_df["hh_id"] = holdings_wide_df.index
    holdings_long_df = pd.melt(holdings_wide_df, id_vars=['hh_id'], value_vars=asset_types, var_name='asset_type', value_name='curr_value') 
    holdings_long_df["units"] = 1
    holdings_long_df["curr_value"] = holdings_long_df["curr_value"].apply(lambda x: round(max(x, 0), -3))
    holdings_long_df = holdings_long_df[holdings_long_df["curr_value"] > 0]
    holdings_long_df["holding_id"] = holdings_long_df.index
    holdings_long_df = holdings_long_df[["hh_id", "curr_value", "asset_type", "units", "holding_id"]]
    return holdings_long_df

#---------------------------------
def generate_for_size(household_size, n_samples):

    # Read all data and filter on household size
    print("reading files 0/3")
    acs_all_one_size_df = read_acs_data_of_household_size(
        household_size = household_size,
        acs_households_filename = DATA_IN_DIRECTORY + "source_ACS/acs_household.csv",
        acs_people_filename = DATA_IN_DIRECTORY + "source_ACS/acs_people.csv"
    ).dropna() # This can be removed if imputation is done
    print("reading files 1/3")
    psid_all_one_size_df = read_psid_data_of_household_size(
        household_size = household_size,
        psid_households_filename = DATA_IN_DIRECTORY + "source_PSID/psid_households.csv",
        psid_people_filename = DATA_IN_DIRECTORY + "source_PSID/psid_people.csv",
        psid_holdings_filename = DATA_IN_DIRECTORY + "source_PSID/psid_holdings.csv",
    ).dropna() # This can be removed if imputation is done
    print("reading files 2/3")
    sipp_all_one_size_df = read_sipp_data_of_household_size(
        household_size = household_size,
        sipp_households_filename = DATA_IN_DIRECTORY + "source_SIPP/sipp_households.csv",
        sipp_people_filename = DATA_IN_DIRECTORY + "source_SIPP/sipp_people.csv",
        sipp_holdings_filename = DATA_IN_DIRECTORY + "source_SIPP/sipp_holdings.csv",
    ).dropna() # This can be removed if imputation is done
    print("reading files 3/3")

    # Generate all samples for the household size
    print(acs_all_one_size_df.columns)
    acs_generator_one_size = train_acs_generator(acs_all_one_size_df)
    psid_generator_one_size = train_psid_generator(psid_all_one_size_df)
    sipp_generator_one_size = train_sipp_generator(sipp_all_one_size_df)
    generated_acs_one_size_df = generate_acs_one_size(acs_generator_one_size, n_samples)
    generated_acs_psid_one_size_df = generate_psid_on_acs_one_size(psid_all_one_size_df.columns, psid_generator_one_size, generated_acs_one_size_df)
    generated_acs_psid_sipp_one_size_df = generate_sipp_on_acs_psid_one_size(sipp_all_one_size_df.columns, sipp_generator_one_size, generated_acs_psid_one_size_df)
    return generated_acs_psid_sipp_one_size_df

def generate_groups(sample_size):
    
    # Read all data and filter on household size
    print("reading files 0/3")
    acs_all_groups_df, acs_all_first_two_df = read_acs_data_of_groups(
        acs_households_filename = DATA_IN_DIRECTORY + "source_ACS/acs_household.csv",
        acs_people_filename = DATA_IN_DIRECTORY + "source_ACS/acs_people.csv"
    )
    print("reading files 1/3")
    psid_all_groups_df, psid_all_first_two_df = read_psid_data_of_groups(
        psid_households_filename = DATA_IN_DIRECTORY + "source_PSID/psid_households.csv",
        psid_people_filename = DATA_IN_DIRECTORY + "source_PSID/psid_people.csv",
        psid_holdings_filename = DATA_IN_DIRECTORY + "source_PSID/psid_holdings.csv",
    )
    print("reading files 2/3")
    sipp_all_groups_df, sipp_all_first_two_df = read_sipp_data_of_groups(
        sipp_households_filename = DATA_IN_DIRECTORY + "source_SIPP/sipp_households.csv",
        sipp_people_filename = DATA_IN_DIRECTORY + "source_SIPP/sipp_people.csv",
        sipp_holdings_filename = DATA_IN_DIRECTORY + "source_SIPP/sipp_holdings.csv",
    )
    print("reading files 3/3")

    # generate two oldest persons in the household
    # (assuming they are the main incomes, this is a valid approach, but we should maybe select more cleverly the first two persons)
    acs_generator_first_two = train_acs_generator(acs_all_first_two_df.dropna())
    psid_generator_first_two = train_psid_generator(psid_all_first_two_df.dropna())
    sipp_generator_first_two = train_sipp_generator(sipp_all_first_two_df.dropna())
    generated_acs_first_two_df = generate_acs_one_size(acs_generator_first_two, sample_size)
    generated_acs_first_two_df = generated_acs_first_two_df.dropna() # SDV Still drops some generation
    generated_acs_psid_first_two_df = generate_psid_on_acs_one_size(psid_all_first_two_df.columns, psid_generator_first_two, generated_acs_first_two_df)
    generated_acs_psid_first_two_df = generated_acs_psid_first_two_df.dropna() # SDV Still drops some generation

    # Replace some categorical values from ACS and PSID unseen in the SIPP dataset.
    # The SDV isn't handling them correctly.
    # The default behavior of the SDV is to pick a random category for categorical values it hasn't seen yet.
    # In this way, it can still generate values for these items.
    # However, doing this before generation is way faster!
    # However, there might be a bug that let's the generation crash.
    # This is reported on: https://github.com/sdv-dev/SDV/issues/2376
    generated_acs_psid_first_two_df = generated_acs_psid_first_two_df.replace("NativeAmerican", "Black").replace("Grandparent", "Parent")
    generated_acs_psid_sipp_first_two_df = generate_sipp_on_acs_psid_one_size(sipp_all_first_two_df.columns, sipp_generator_first_two, generated_acs_psid_first_two_df)
    generated_acs_psid_sipp_first_two_df = generated_acs_psid_sipp_first_two_df.dropna() # SDV Still drops some generation

    # Generate ACS data for other persons in the house based on the main two persons
    acs_generator_groups = train_acs_generator_groups(acs_all_groups_df)
    first_two_acs = generated_acs_first_two_df[generated_acs_first_two_df["size"] >= 3][[col for col in generated_acs_first_two_df.columns if col[-2:] != "_2"]]
    extra_generated_persons_acs = acs_generator_groups.sample_remaining_columns(
        known_columns=first_two_acs.reindex(first_two_acs.index.repeat(first_two_acs["size"] - 2)),
        max_tries_per_batch=20
    )
    extra_generated_persons_acs = extra_generated_persons_acs[[col for col in extra_generated_persons_acs if col[-2:] == "_2"]]
    extra_generated_persons_acs = extra_generated_persons_acs.dropna() # SDV Still drops some generation
    extra_generated_persons_acs = extra_generated_persons_acs.reset_index().rename(columns={"index": "hh_id"})

    # Generate PSID data for other persons in the house based on the main two persons and ACS data for the other persons
    psid_generator_groups = train_psid_generator(psid_all_groups_df)
    extra_generated_persons_acs_index_hh_id = extra_generated_persons_acs.copy()
    extra_generated_persons_acs_index_hh_id.index = extra_generated_persons_acs_index_hh_id["hh_id"]
    acs_complete_psid_first_two = pd.merge(generated_acs_psid_first_two_df, extra_generated_persons_acs_index_hh_id, left_index=True, right_index=True)
    acs_complete_psid_first_two = acs_complete_psid_first_two[acs_complete_psid_first_two["size"] >= 3][[col for col in acs_complete_psid_first_two.columns if col in psid_all_groups_df.columns and col[:6] != "weight"]]
    acs_complete_psid_first_two = acs_complete_psid_first_two.reset_index().rename(columns={"index": "hh_id"})

    extra_generated_persons_psid = psid_generator_groups.sample_remaining_columns(
        known_columns=acs_complete_psid_first_two.drop(columns=["hh_id"]).dropna(),
        max_tries_per_batch=20
    )

    #extra_generated_persons_acs = extra_generated_persons_acs.reset_index().rename(columns={"index": "hh_id"})
    extra_generated_persons_psid = extra_generated_persons_psid[[col for col in extra_generated_persons_psid.columns if col[-2:] == "_2"]]
    extra_generated_persons_acs_psid = pd.merge(extra_generated_persons_acs, extra_generated_persons_psid[[col for col in extra_generated_persons_psid.columns if col not in extra_generated_persons_acs.columns]], left_index=True, right_index=True)
    extra_generated_persons_acs_psid = extra_generated_persons_acs_psid.dropna() # SDV Still drops some generation

    # Generate SIPP data for other persons in the house based on the main two persons and ACS&PSID data for the other persons
    sipp_generator_groups = train_sipp_generator(sipp_all_groups_df)
    acs_psid_complete_sipp_first_two = pd.merge(generated_acs_psid_sipp_first_two_df, extra_generated_persons_acs_psid, left_index=True, right_on="hh_id")

    acs_psid_complete_sipp_first_two_to_generate_next_sipp_person = acs_psid_complete_sipp_first_two[acs_psid_complete_sipp_first_two["size"] >= 3][[col for col in acs_psid_complete_sipp_first_two.columns if col in sipp_all_groups_df.columns and col[:6] != "weight"]]
    columns_to_match_extra_sipp = [
        'age_0', 'race_0', 'gender_0', 'cur_income_0',
        'age_1', 'race_1', 'gender_1', 'cur_income_1',
        'age_2', 'educ_years_2', 'race_2', 'gender_2', 'cur_income_2', 'partnership_status_2',
        'household_income', 'size', 'partnership_status', 'relationshipToR_2',
    ] # Adding more variables to the conditional generation is possible, but significantly slows down generation
    # For example, it goes from  .. to 28 minutes if adding educ_years_0, educ_years_1, House, OtherRealEstate, Business
    acs_psid_complete_sipp_first_two_to_generate_next_sipp_person = acs_psid_complete_sipp_first_two[columns_to_match_extra_sipp]

    # Replace some categorical values from ACS and PSID unseen in the SIPP dataset.
    # The SDV isn't handling them correctly.
    # The default behavior of the SDV is to pick a random category for categorical values it hasn't seen yet.
    # In this way, it can still generate values for these items.
    # However, doing this before generation is way faster!
    # However, there might be a bug that let's the generation crash.
    # This is reported on: https://github.com/sdv-dev/SDV/issues/2376
    acs_psid_complete_sipp_first_two_to_generate_next_sipp_person = acs_psid_complete_sipp_first_two_to_generate_next_sipp_person.replace("NativeAmerican", "Black").replace("Pacific", "Black").drop(columns=["relationshipToR_2"])
    extra_generated_persons_sipp = sipp_generator_groups.sample_remaining_columns(
        known_columns=acs_psid_complete_sipp_first_two_to_generate_next_sipp_person.dropna(),
        max_tries_per_batch=20
    )

    extra_generated_persons_sipp[[col for col in extra_generated_persons_sipp.columns if col[-2:] == "_2"]]
    extra_generated_persons_sipp.index = extra_generated_persons_sipp.index.to_series().apply(lambda x: int(x))
    extra_generated_persons_acs_psid_sipp = pd.merge(extra_generated_persons_acs_psid, extra_generated_persons_sipp[[col for col in extra_generated_persons_psid.columns if col not in extra_generated_persons_acs_psid.columns]], left_index=True, right_index=True)
    extra_generated_persons_acs_psid_sipp.index = extra_generated_persons_acs_psid_sipp["hh_id"]
    extra_generated_persons_acs_psid_sipp = extra_generated_persons_acs_psid_sipp.drop(columns=["hh_id"])
    extra_generated_persons_acs_psid_sipp = extra_generated_persons_acs_psid_sipp.dropna() # SDV Still drops some generation

    return generated_acs_psid_sipp_first_two_df, extra_generated_persons_acs_psid_sipp

def generate_ACS_PSID_SIPP_all():
    sizes_to_generate = get_household_sizes_to_generate(1000, DATA_IN_DIRECTORY + "source_ACS/acs_household.csv")

    # Generation, including reading in files and training
    n_sample_for_size_1 = int(sizes_to_generate.loc[1].values[0])
    generated_for_size_1 = generate_for_size(1, n_sample_for_size_1)
    n_sample_for_size_2 = int(sizes_to_generate.loc[2].values[0])
    generated_for_size_2 = generate_for_size(2, n_sample_for_size_2)
    n_sample_for_size_3plus = int(sizes_to_generate.loc[3:].sum()[0])
    generated_acs_psid_sipp_first_two_df, extra_generated_persons_acs_psid_sipp = generate_groups(n_sample_for_size_3plus)

    # Make household ids different over different datasets
    generated_for_size_2.index = generated_for_size_2.index + generated_for_size_1.index.max() + 1
    generated_acs_psid_sipp_first_two_df.index = generated_acs_psid_sipp_first_two_df.index + generated_for_size_2.index.max() + 1
    extra_generated_persons_acs_psid_sipp.index = extra_generated_persons_acs_psid_sipp.index + generated_for_size_2.index.max() + 1

    # Extract data
    people_1_df = extract_people_one_size_csv(1, generated_for_size_1)
    household_1_df = extract_households_csv(generated_for_size_1)
    holdings_1_long_df = extract_holdings_csv(generated_for_size_1)
    people_2_df = extract_people_one_size_csv(2, generated_for_size_2)
    household_2_df = extract_households_csv(generated_for_size_2)
    holdings_2_long_df = extract_holdings_csv(generated_for_size_2)
    people_3_df = extract_people_one_size_csv(2, generated_acs_psid_sipp_first_two_df)
    household_3_df = extract_households_csv(generated_acs_psid_sipp_first_two_df)
    holdings_3_long_df = extract_holdings_csv(generated_acs_psid_sipp_first_two_df)
    people_3_extra_df = extract_people_groups_csv(extra_generated_persons_acs_psid_sipp)

    # Gather and save
    people_df = pd.concat([people_1_df, people_2_df, people_3_df, people_3_extra_df])
    household_df = pd.concat([household_1_df, household_2_df, household_3_df])
    holdings_long_df = pd.concat([holdings_1_long_df, holdings_2_long_df, holdings_3_long_df])
    people_df.to_csv("output/people.csv")
    household_df.to_csv("output/household.csv")
    holdings_long_df.to_csv("output/holdings.csv", index=False)


if __name__ == "__main__":
    generate_ACS_PSID_SIPP_all()
