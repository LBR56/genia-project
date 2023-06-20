import os

import boto3
from botocore.client import Config
import pandas as pd

from configs import AWS_S3_PARAM

class AwsController:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_S3_PARAM["ACCESS_KEY_ID"],
            aws_secret_access_key=AWS_S3_PARAM["ACCESS_SECRET_KEY"],
            config=Config(signature_version="s3v4")
            )

    def s3_upload(self, dfs:dict, dir=""):
        for f_name, df in dfs.items():
            f_name = dir + f_name

            df.to_csv("/tmp/" + f_name, index=False)

            self.s3.upload_file(
                Bucket=AWS_S3_PARAM["BUCKET_NAME"],
                Key=f_name, 
                Filename="/tmp/" + f_name
                )
            
            os.remove("/tmp/" + f_name)

    def s3_download(self, keys:list, dir=""):
        dfs = {}
        for f_name in keys:
            f_name = dir + f_name
            self.s3.download_file(
                Bucket=AWS_S3_PARAM["BUCKET_NAME"],
                Key=f_name, 
                Filename="/tmp/" + f_name
                )
            
            dfs[f_name.split(".")[0]] = pd.read_csv("/tmp/" + f_name)
            
            os.remove("/tmp/" + f_name)
        return dfs
