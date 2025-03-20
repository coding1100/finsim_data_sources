## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Analytics

### Run plot_categorical_combinations.py

Run by setting paths:
```
python analytics/plot_categorical_combincations.py --data_in_directory PATH_TO_DATASET --source_ACS_csv PATH_TO_SOURCE_ACS_CSV --ref_agents_csv PATH_TO_REF_AGENTS_CSV --output_file_path PATH_TO_OUTPUT_CSV

```
Example:
```
python analytics/plot_categorical_combincations.py --data_in_directory processed_data/synthetic_population_generation/ --source_ACS_csv processed_data/synthetic_population_generation/source_ACS/acs_people.csv --ref_agents_csv processed_data/agents/people.csv --output_file_path
```
