import os
import logging
from ast import literal_eval

from aprec.utils.os_utils import mkdir_p_local, get_dir, console_logging, shell
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.api.lfm_item import LFMItem
from aprec.api.catalog import Catalog
from datetime import datetime

DATASET_NAME = 'ml-20m'
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/{}.zip".format(DATASET_NAME)

LFM_DIR = "data/mini_le"
LFM_DIR_ABSPATH = os.path.join(get_dir(), LFM_DIR)

LISTENING_EVENTS_FILE = os.path.join(LFM_DIR_ABSPATH, 'mini_le.csv')
TRACKS_FILE = os.path.join(LFM_DIR_ABSPATH, 'mini_tracks.tsv')

# DONT BELIEVE THESE FUNCTIONS ARE REQUIRED
'''
def extract_movielens_dataset():
    if os.path.isfile(RATINGS_FILE):
        logging.info("movielens dataset is already extracted")
        return
    shell("unzip -o {} -d {}".format(MOVIELENS_FILE_ABSPATH, MOVIELENS_DIR_ABSPATH))
    dataset_dir = os.path.join(MOVIELENS_DIR_ABSPATH, DATASET_NAME)
    for filename in os.listdir(dataset_dir):
        shell("mv {} {}".format(os.path.join(dataset_dir, filename), MOVIELENS_DIR_ABSPATH))
    shell("rm -rf {}".format(dataset_dir))

def prepare_data():
    download_file(MOVIELENS_URL, MOVIELENS_FILE, MOVIELENS_DIR)
    extract_movielens_dataset()
'''

'''
# This function will only take users with more than 10 listens
def clean_lfm_data():
    with open(LISTENING_EVENTS_FILE, 'r') as file:
        df = pd.read_csv(file, sep='\t')
        user_counts = df.groupby(by=['user_id']).count()
        suitable_users = user_counts.loc[user_counts['track_id']>=20].index
        df = df[suitable_users]
    return df
'''


def get_lfm_actions():
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
    #prepare_data()
