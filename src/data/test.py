from prefect_aws.s3 import S3Bucket

s3_bucket_block = S3Bucket.load("cameroon-air-quality-bucket")
print(s3_bucket_block)