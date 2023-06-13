from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import pandas as pd

def get_youtube_transcript(video_ids):
    df = None
    error_video_ids = []
    for video_id in tqdm(video_ids):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_generated_transcript(["ko"])
            
            temp_df = pd.DataFrame(transcript.fetch())
            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df]).reset_index(drop=True)
            
        except Exception as e:
            print(video_id)
            error_video_ids += [video_id]
            print(e)

    return df, error_video_ids