from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpRequest
import pandas as pd

from configs import REQUESTS, BUILD_PARAM

class YoutubeApi(object):    
    def __init__(self):
        self.__service = build(**BUILD_PARAM)

    def __api_request(self, request:HttpRequest):
        assert isinstance(request, HttpRequest)

        self._response = None
        try:
            self._response = request.execute()
        except HttpError as e:
            error_message = "Error response status code : {}, reason : {}"
            print(error_message.format(e.status_code, e.error_details))
            
        return self._response

    def youtube_search_ids(self, **kwargs):
        
        search_param = REQUESTS["search_param"]
        if kwargs["query"]:
            search_param["q"] = kwargs["query"]
        
        video_ids = []
        for _ in range(0, kwargs["max_results"] + 1, 50):
            request = self.__service.search().list(**search_param)
            response = self.__api_request(request)

            video_ids += [item["id"]["videoId"] for item in response["items"]]
            
            search_param["pageToken"] = response["nextPageToken"]
        return list(set(video_ids))[:kwargs["max_results"]]

    def make_video_meta(self, response):
        dfs = [pd.DataFrame(response["items"])]

        for col in ["snippet", "contentDetails", "statistics"]:
            dfs += [pd.DataFrame(dfs[0][col].tolist())]
            dfs[0].drop(col, axis=1, inplace=True)

        return pd.concat(dfs, axis=1)
    
    def youtube_videos_meta(self, video_ids):
        dfs = []
        for start_id in range(0, len(video_ids), 50):
            request = self.__service.videos().list(
                **REQUESTS["videos_param"],
                id = video_ids[start_id:start_id + 50]
                )
            response = self.__api_request(request)

            dfs += [self.make_video_meta(response)]

        return pd.concat(dfs).reset_index(drop=True)
    