import click

from utils import set_result_dir
from youtube_api import YoutubeApi
from transcript_api import get_youtube_transcript
from aws import AwsController
from preprocesser import Preprocesser

@click.command()
@click.option("-q", "--query", type=str, help="to query term to search for")
@click.option("-r", "--max-results", default=50, type=int, help="max results")
@click.option("--result-dir", default="src/")
def main(**kwargs):
    kwargs["result_dir"] = set_result_dir(**kwargs)
    youtubeApi = YoutubeApi()

    video_ids = None
    if kwargs["query"]:
        video_ids = youtubeApi.youtube_search_ids(**kwargs)
    else:
        with open(kwargs["result_dir"] + "target_videos.txt", "r") as fp:
            video_ids = "".join(fp.readlines())
            video_ids = video_ids.split("\n")
    
    # video_meta_df = youtubeApi.youtube_videos_meta(video_ids)
    # transcript_df, error_video_ids = get_youtube_transcript(video_ids)

    # is_error = video_meta_df.apply(lambda row:row["id"] in error_video_ids, axis=1)
    # video_meta_df = video_meta_df[~ is_error]

    aws_controller = AwsController()
    # aws_controller.s3_upload(
    #     {
    #         "videos.csv" : video_meta_df,
    #         "transcripts.csv" : transcript_df
    #     },
    #     kwargs["result_dir"]
    # )
    results_df = aws_controller.s3_download(["videos.csv", "transcripts.csv"], kwargs["result_dir"])
    
    video_meta_df = results_df["videos"]
    transcript_df = results_df["transcripts"]

    video_meta_df = video_meta_df.sort_values(by="viewCount", ascending=False)
    video_meta_df = video_meta_df.reset_index(drop=True)
    transcript_df
    
    preprocesser = Preprocesser()
    preprocessed_df = preprocesser.get_preprocessed_df(video_meta_df, transcript_df)

    aws_controller.s3_upload(
        {
            "preprocessed.csv":preprocessed_df
        },
        kwargs["result_dir"]
    )

if __name__ == "__main__":
    main()