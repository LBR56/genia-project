import os

import boto3
from botocore.client import Config

from configs import AWS_S3_PARAM

def s3_upload( dfs:dict, dir=""):
    s3 = boto3.resource(
        's3',
        aws_access_key_id=AWS_S3_PARAM["ACCESS_KEY_ID"],
        aws_secret_access_key=AWS_S3_PARAM["ACCESS_SECRET_KEY"],
        config=Config(signature_version="s3v4")
    )
    
    for f_name, df in dfs.items():
        df.to_csv(f_name)
        s3.Bucket(AWS_S3_PARAM["BUCKET_NAME"]).put_object(
            Key=dir + f_name, Body=f_name, ContentType="application/csv")
        os.remove(f_name)
