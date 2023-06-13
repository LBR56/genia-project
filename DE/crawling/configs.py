import os

from dotenv import load_dotenv

load_dotenv()

BUILD_PARAM = {
    "serviceName":os.environ.get("YOUTUBE_API_SERVICE_NAME"), 
    "version":os.environ.get("YOUTUBE_API_VERSION"), 
    "developerKey":os.environ.get("DEVELOPER_KEY")
}

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
