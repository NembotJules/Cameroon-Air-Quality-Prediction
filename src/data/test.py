from prefect import flow, task

@flow(name="Test flow")
def greatings(): 
    print("Hello World")

if __name__ == "__main__": 
    greatings()