from prefect import flow, task
from prefect.runner.storage import GitRepository
from prefect_github.repository import GitHubRepository
from prefect.blocks.system import Secret

@flow(name="Test flow")
def greatings(): 
    print("Hello World")
    

if __name__ == "__main__": 
    greatings.from_source(
        source = GitHubRepository.load("cameroon-air-quality-prediction-project-repo"), 
        entrypoint = "src/data/test.py:greatings"
    ).deploy(
        name="my-first-deployment", 
         work_pool_name="Managed-Pool", 
     )
#     greatings.from_source(
#         source=GitRepository(
#         url="https://github.com/NembotJules/Cameroon-Air-Quality-Prediction.git",
#         branch="dev",
#         credentials={
#             "access_token": Secret.load("github-access-token")
#         }
#     )).deploy(
#         name="my-first-deployment", 
#         work_pool_name="Managed-Pool", 
#     )



print(GitHubRepository.load("cameroon-air-quality-prediction-project-repo"))