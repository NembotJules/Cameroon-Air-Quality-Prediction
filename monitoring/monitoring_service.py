import os
import pandas as pd
from prefect import flow
import asyncio
from prefect_github import GitHubCredentials
from prefect.runner.storage import GitRepository
from prefect.client.schemas.schedules import CronSchedule
import yaml
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import DataDriftPreset
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import DashboardPanelTestSuite
from evidently.ui.dashboards import PanelValue
from evidently import metrics
from evidently.metrics import RegressionQualityMetric
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import DataDriftTestPreset
from evidently.tests.base_test import TestResult, TestStatus
from evidently.ui.dashboards import TestFilter
from evidently.ui.dashboards import TestSuitePanelType
from evidently.renderers.html_widgets import WidgetSize

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)


@flow(name="Monitoring Flow", log_prints=True)
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


    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Daily inference Count",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
            	PanelValue(
                	metric_id="DatasetSummaryMetric",
                	field_path=metrics.DatasetSummaryMetric.fields.current.number_of_rows,
                	legend="count",
            	),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.FULL,
        ),
        tab="Summary"
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Share of drifting features (PSI > 0.3)",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                	metric_id="DatasetDriftMetric",
                	field_path="share_of_drifted_columns",
                	legend="share",
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.FULL,
        ),
        tab="Summary"
    )

    drift_tests = TestSuite(
        tests=[
            DataDriftTestPreset(stattest_threshold=0.5),
            TestShareOfMissingValues(lte=0.05),
            TestNumberOfConstantColumns(eq=0),
            TestNumberOfEmptyRows(eq=0),
            TestNumberOfEmptyColumns(eq=0),
            TestNumberOfDuplicatedColumns(eq=0)
        ],
       
    )

    drift_tests.run(reference_data=reference_data, current_data=current_data)

    project.dashboard.add_panel(
        DashboardPanelTestSuite(
            title="Data quality tests",
            test_filters=[
                TestFilter(test_id="TestNumberOfConstantColumns", test_args={}),
                TestFilter(test_id="TestShareOfMissingValues", test_args={}),
                TestFilter(test_id="TestNumberOfEmptyRows", test_args={}),
                TestFilter(test_id="TestNumberOfEmptyColumns", test_args={}),
                TestFilter(test_id="TestNumberOfDuplicatedColumns", test_args={}),
            ],
            filter=ReportFilter(metadata_values={}, tag_values=[], include_test_suites=True),
            size=WidgetSize.FULL,
            panel_type=TestSuitePanelType.DETAILED,
            time_agg="1D",
        ),
        tab="Data Tests"
    )
    project.dashboard.add_panel(
        DashboardPanelTestSuite(
            title="Data drift per column in time",
            test_filters=[
                TestFilter(test_id="TestColumnDrift", test_args={}),
            ],
            filter=ReportFilter(metadata_values={}, tag_values=[], include_test_suites=True),
            size=WidgetSize.FULL,
            panel_type=TestSuitePanelType.DETAILED,
            time_agg="1D",
        ),
        tab="Data Tests"
    )

    # Define Column Mapping
    


    project.save()
    



if __name__=="__main__": 
    # Run the async function
    # asyncio.run(create_project_and_report())
    create_project_and_report.from_source(
        
         source=GitRepository(
            url="https://github.com/NembotJules/Cameroon-Air-Quality-Prediction.git",
            branch="main",
            credentials=GitHubCredentials.load("git-credentials")
            ),
        entrypoint = "monitoring/monitoring_service.py:create_project_and_report"
    ).deploy(
        name="Monitoring Service Pipeline", 
        work_pool_name="Managed-Pool", 
        schedules = [
            CronSchedule(
                cron = "5 1 * * *", 
                timezone = "Africa/Douala"
            )
        ]

     )


