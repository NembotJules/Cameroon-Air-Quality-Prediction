from prefect import flow, task

@flow(name="Test flow")
def greatings(): 
    print("Hello World")

if __name__ == "__main__": 
    greatings.deploy(
        name="my-first-deployment", 
        work_pool_name="Managed-Pool", 
    )