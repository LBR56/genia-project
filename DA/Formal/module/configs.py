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

# 인수분해, 근해공식
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