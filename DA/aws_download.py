import sys, os
from aws import AwsController
from datetime import datetime
import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

today = datetime.now().strftime(r"%Y%m%d")

def set_result_dir(result_dir, query = None, **kwargs):
    if query:
        result_dir = "src/"
        result_dir += datetime.now().strftime(r"%Y%m%d") + "/"
        result_dir += query + "/"
        
    if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
    return result_dir

set_result_dir("src/", "인수분해")


aws_controller = AwsController()
temp_df = aws_controller.s3_download(["stop_words.csv", "math_dict.csv"])
for f_name, df in temp_df.items():
    df.to_csv("./src/words/" + f_name + ".csv")

data = aws_controller.s3_download(["preprocessed.csv"], f"src/{today}/인수분해/")["preprocessed"]
data.to_csv(f"./src/{today}/인수분해/preprocessed.csv")


