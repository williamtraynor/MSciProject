import os
import logging
from ast import literal_eval

from base.utils.os_utils import get_dir, console_logging, shell
from base.api.action import Action
from base.api.item import Item
from base.api.lfm_item import LFMItem
from base.api.catalog import Catalog
from datetime import datetime

from base.datasets.download_file import download_file

TRACKS_ID = '1lTjaWJuzWtU1sA03Gc4LpGPEvnUNZ8mg'
#TRACKS_URL = 'https://drive.google.com/file/d/1lTjaWJuzWtU1sA03Gc4LpGPEvnUNZ8mg/view?usp=share_link'
TRACKS_URL = 'https://drive.google.com/uc?id=1lTjaWJuzWtU1sA03Gc4LpGPEvnUNZ8mg'
LE_ID = '1GJEriUD6PMBi_DPqb0hY0Tuu8FQLJr_W'
#LE_URL = 'https://drive.google.com/file/d/1GJEriUD6PMBi_DPqb0hY0Tuu8FQLJr_W/view?usp=share_link'
LE_URL = 'https://drive.google.com/uc?id=1GJEriUD6PMBi_DPqb0hY0Tuu8FQLJr_W'

FOLDER_NAME = 'mini_le'
LFM_DIR = "data/mini_le"
LFM_DIR_ABSPATH = os.path.join(get_dir(), LFM_DIR)

LISTENING_EVENTS_FILE = os.path.join(LFM_DIR_ABSPATH, 'mini_le.csv')
TRACKS_FILE = os.path.join(LFM_DIR_ABSPATH, 'mini_tracks.tsv')

# DONT BELIEVE THESE FUNCTIONS ARE REQUIRED


def get_lfm_actions():
    #download_file(LE_URL, 'mini_le.csv')
    with open(LISTENING_EVENTS_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                user_id, track_id, album_id, timestamp_str, _, _, skipped = line.strip().split(',')
                timestamp = int((datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")).timestamp())
                yield Action(user_id, track_id, timestamp, {"skipped": literal_eval(skipped)})


def get_tracks_catalog():
    #download_file(TRACKS_URL, 'mini_tracks.tsv')
    catalog = Catalog()
    with open(TRACKS_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                # Line Data - track_id, artist, title, uri, duration_ms, duration_s
                track_id, artist, title, _, _, _ = line.strip().split("\t")
                item = LFMItem(track_id).with_title(title).with_artist(artist)
                catalog.add_item(item)
    return catalog


if __name__ == "__main__":
    console_logging()
