from prefect import flow, task

@flow(name="Test flow")
def greatings(): 
    print("Hello World")

if __name__ == "__main__": 
    greatings.from_source(
        source="https://github.com/NembotJules/Cameroon-Air-Quality-Prediction.git", 
        entrypoint="src/data/test.py:greatings"
    ).deploy(
        name="my-first-deployment", 
        work_pool_name="Managed-Pool", 
    )