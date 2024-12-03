import os
import pandas as pd
import asyncio
import yaml
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import DataDriftPreset

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)

async def create_project_and_report():
    evidently_token = os.getenv("EVIDENTLY_TOKEN")
    #team_id = os.getenv("TEAM_ID")
    
    ws = CloudWorkspace(token=evidently_token, url="https://app.evidently.cloud")
    project = ws.get_project(default_config['evidently']['project_id'])
    


    # The data fetch and preprocessed each day by my data_pipeline
    current_data = pd.read_csv(default_config['data']['preprocessed_pipeline_features_data_path']) 
    # The test set using when evualting the performance of the model...
    reference_data = pd.read_csv(default_config['data']['preprocessed_test_data_path'])
    
    data_report = Report(
        metrics=[
            DataDriftPreset(stattest='psi', stattest_threshold='0.3'),
            DataQualityPreset(),
        ],
    )

    data_report.run(reference_data=reference_data, current_data=current_data)
    ws.add_report(project.id, data_report)


    project.save()
    

# Run the async function
asyncio.run(create_project_and_report())
