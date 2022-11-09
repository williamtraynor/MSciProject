import os
import requests
from base.utils.os_utils import get_dir
import gdown


def download_file(file_url, filename):

    output = 'base/data/mini_le/' + filename

    gdown.download(file_url, output, quiet=True)

    return

    