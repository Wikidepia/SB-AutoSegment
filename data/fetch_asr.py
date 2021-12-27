import json
import multiprocessing

from transcript_api import get_transcript


def fetch(video_id):
    try:
        transcript = get_transcript(video_id)
        print(json.dumps({"video_id": video_id, "transcript": transcript}))
    except:
        return


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=8)
    with open("videoIds.txt") as f:
        video_ids = f.read().splitlines()
    pool.map(fetch, video_ids)
