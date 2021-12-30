import multiprocessing
import random

import numpy as np
import requests
import ujson


def safe_arange(start, stop, step):
    return step * np.arange(start / step, stop / step)


def process_data(line):
    try:
        port = random.choice([80, 81, 82, 83])
        data = ujson.loads(line)
        video_id = data["video_id"]
        # Tag data with sponsored or content
        ranges = []
        transcript = data["transcript"]
        r = requests.get(
            f"http://65.108.182.194:{port}/api/skipSegments?videoID={video_id}"
        ).json()
        for segment in r:
            sponsor_segment = segment["segment"]
            start = sponsor_segment[0]
            end = sponsor_segment[1]
            arange = safe_arange(start, end, 0.001)
            arange = np.around(arange, 3)
            ranges.append(arange)

        ret = []
        for ts in transcript:
            is_sponsor = False
            for r in ranges:
                if round(ts["show_s"], 3) in r:
                    ret.append((ts["text"], "SPONSOR"))
                    is_sponsor = True
                    continue
            if not is_sponsor:
                ret.append((ts["text"], "CONTENT"))
        print(ujson.dumps(ret))
    except:
        pass


if __name__ == "__main__":
    with open("split-transcript/xaa") as f:
        all_data = f.read().splitlines()

    with multiprocessing.Pool(processes=8) as pool:
        pool.map(process_data, f)
