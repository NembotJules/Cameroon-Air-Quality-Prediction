import os
import yaml
import pandas as pd
from sklearn import datasets

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)

evidently_token = os.getenv("EVIDENTLY_TOKEN")
team_id = os.getenv("TEAM_ID")

ws = CloudWorkspace(token= "dG9rbgFak65gtmhETqpgvtj+ZhHh76ajru2sbt0UIZTtd8eNvwBQJ0M7zv6g7aRt5PE6XPtNgmX6iJ8XExsSfhNC8nCcBH4Uma8mWBLMenieNqxZvBJptedd4DnxN94/DrsawFCkeSitdfayx5rVBNazri1vWAyfxbzS", 
                    url="https://app.evidently.cloud")

project = ws.create_project("Cameroon Air Quality Prediction Project",
                             team_id="01938c28-7e77-7825-88fe-862af1d9b1fc")

project.description = "Cameroon Air Quality Prediction Project"
project.save()

## Dataset

train_data = pd.read_csv(default_config['data']['raw_train_data_path'])

print(train_data.shape)


data_report = Report(
       metrics=[
           DataQualityPreset(),
       ],
    )
data_report.run(reference_data=None, current_data=train_data)
ws.add_report(project.id, data_report)

# print(ws)