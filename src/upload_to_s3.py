import boto3 
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_aws_credentials(): 
    """
    Load AWS credentials from environment variable or AWS credentials file.
    Returns configured boto3 S3 client.
    """

    try: 
        #Load environment variables from .env file if  exists
        load_dotenv()

        if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ: 
            return boto3.client(
                's3', 
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_REGION', 'us-east-1')
            )


        else: 

            #If no environment variables, boto3 will automatically check: 
            #1. AWS credentials file (.aws/credentials)
            #2. AWS CLI configuration

            return boto3.client('s3')
        

    except Exception as e: 
        logger.error(f"Error loading AWS credentials: {str(e)}")
        raise


def upload_folder_to_s3(local_folder_path, bucket_name, s3_folder_prefix): 

    """
    Upload all files from a local folder to S3 bucket

    Args: 
        local_folder_path(str): Path to local folder containing files
        bucket_name(str): Name of the S3 bucket
        s3_folder_prefix (str): Prefix for S3 keys (folder structure in S3)
    """

    try: 
        # Initialize S3 client
        s3_client =  load_aws_credentials()

        # Ensure local folder path exists
        if not os.path.exists(local_folder_path): 
            raise FileNotFoundError(f"Local folder not found: {local_folder_path}")
        

        # List all files in the local folder
        files  = [f for f in os.listdir(local_folder_path) if os.path.isfile(os.path.join(local_folder_path, f))]

        if not files: 
            logger.warning(f"No files found in {local_folder_path}")
            return
        
        # Upload each file

        for file in files: 
            local_file_path = os.path.join(local_folder_path, file)
            s3_key = f'{s3_folder_prefix.rstrip("/")}/{file}'


            try: 
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                logger.info(f'Succesfully uploaded {file} to s3://{bucket_name}/{s3_key}')
            
            except ClientError as e: 
                logger.error(f'Error uploading {file}: {str(e)}')
                continue


    except Exception as e: 
        logger.error(f"Error in upload process: {str(e)}")
        raise




if __name__=="__main__": 
     # Configuration
    BUCKET_NAME = 'cameroon-air-quality-bucket'
    RAW_DATA_FOLDER = '../data/train_test_data/preprocessed_data/'
    S3_PREFIX = 'data/train_test_data/preprocessed_data/'

    upload_folder_to_s3(RAW_DATA_FOLDER, BUCKET_NAME, S3_PREFIX)
    