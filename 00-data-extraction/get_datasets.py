import os
from pathlib import Path
import subprocess

# Get absolute path of this file and its directory path
FILE_PATH = Path(__file__).resolve()
BASE_PATH = (FILE_PATH.parent).parent

print(BASE_PATH)

DATA_DIR = BASE_PATH / "data"
S3_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
FILE_NAMES = ["green_tripdata_2021-01.parquet", "green_tripdata_2021-02.parquet"]


def download_data(file_name: str) -> None:
    file_path = DATA_DIR / file_name
    url = S3_URL + file_name
    if not os.path.isfile(file_path):
        print("File does not exist, downloading from S3 bucket.")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        subprocess.run(["wget", "-O", file_path, url])
        print(f"File downloaded successfully and saved at {file_path}")
    else:
        print("File already exists.")


for file_name in FILE_NAMES:
    download_data(file_name)
