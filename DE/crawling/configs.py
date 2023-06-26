import os

from dotenv import load_dotenv

load_dotenv()

YOUTUBE_BUILD_PARAM = {
    "serviceName":os.environ.get("YOUTUBE_API_SERVICE_NAME"), 
    "version":os.environ.get("YOUTUBE_API_VERSION"), 
    "developerKey":os.environ.get("DEVELOPER_KEY")
}

AWS_S3_PARAM = {
    "ACCESS_KEY_ID":os.environ.get("ACCESS_KEY_ID"),
    "ACCESS_SECRET_KEY":os.environ.get("ACCESS_SECRET_KEY"),
    "BUCKET_NAME":os.environ.get("BUCKET_NAME")
}

# 인수분해, 근의공식
REQUESTS = {
    "search_param" : {
        "part" : "snippet",
        "maxResults" : 50,
        "order" : "relevance",
        "q" : "인수분해",
        "type" : "video",
        # videoCaption = "closedCaption", # any
    },
    "videos_param" : {
        "part" : "snippet,contentDetails,statistics",
    }
}

MAIN_PARAM = {
    "query" : os.environ["query"],
    "max_results" : os.environ["max_results"],
    "result_dir" : os.environ["result_dir"],
}