from datetime import datetime
import os

import click

from youtube_api import YoutubeApi
from transcript_api import get_youtube_transcript

def set_result_dir(**kwargs):
    result_dir = kwargs["result_dir"]
    
    if kwargs["query"]:
        result_dir = "../../data/"
        result_dir += datetime.now().strftime(r"%Y%m%d") + "/"
        result_dir += kwargs["query"] + "/"

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
    elif not os.path.exists(kwargs["result_dir"]):
        raise FileNotFoundError("파일이 존재하지 않습니다.")
    
    return result_dir

@click.command()
@click.option("-q", "--query", type=str, help="to query term to search for")
@click.option("-r", "--max-results", default=50, type=int, help="max results")
@click.option("--result-dir", default="../../data/sample/")
def main(**kwargs):
    print(kwargs)
    kwargs["result_dir"] = set_result_dir(**kwargs)

    youtubeApi = YoutubeApi()

    video_ids = None
    if kwargs["query"]:
        video_ids = youtubeApi.youtube_search_ids(**kwargs)
    else:
        with open(kwargs["result_dir"] + "target_videos.txt", "r") as fp:
            video_ids = "".join(fp.readlines())
            video_ids = video_ids.split("\n")
    
    video_meta_df = youtubeApi.youtube_videos_meta(video_ids)
    video_meta_df.to_csv(kwargs["result_dir"] + "videos.csv")

    transcript_df, error_video_ids = get_youtube_transcript(video_ids)
    transcript_df.to_csv(kwargs["result_dir"] + "transcripts.csv")

if __name__ == "__main__":
    main()