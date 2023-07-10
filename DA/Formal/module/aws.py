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
            df.to_csv(dir + f_name, index=False)

            self.s3.upload_file(
                Bucket=AWS_S3_PARAM["BUCKET_NAME"],
                Key=dir + f_name, 
                Filename=dir + f_name
                )
            
            os.remove(dir + f_name)

    def s3_download(self, keys:list, dir=""):
        dfs = {}
        for f_name in keys:
            print(f_name)
            self.s3.download_file(
                Bucket=AWS_S3_PARAM["BUCKET_NAME"],
                Key=dir + f_name, 
                Filename=dir + f_name
                )
            
            dfs[f_name.split(".")[0]] = pd.read_csv(dir + f_name)
            
            os.remove(dir + f_name)
        return dfs


if __name__ == "__main__":
    aws_controller = AwsController()
    # temp_df = aws_controller.s3_download(["stop_words.csv", "math_dict.csv"])
    df = aws_controller.s3_download(["preprocessed.csv"], "src/")["preprocessed"]
    #
    # aws_controller.s3_upload(
    #     {
    #         "videos.csv" : df,
    #         "transcripts.csv" : df
    #     },
    #     "src/"
    # )

# aws_controller.s3_upload(
#     {
#         "preprocessed.csv":preprocessed_df
#     },
#     kwargs["result_dir"]
# )