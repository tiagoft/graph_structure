import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

# URL of the GloVe zip file
url = "https://nlp.stanford.edu/data/glove.6B.zip"

# Directory to save the downloaded file and extract its contents
script_dir = Path(__file__).parent.parent

# Directory to save the downloaded file and extract its contents
data_dir = script_dir / "data"

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

# Path to save the downloaded zip file
zip_file_path = data_dir /  "glove.6B.zip"

# Download the GloVe zip file
print("Downloading GloVe zip file...")
if not zip_file_path.is_file():
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(zip_file_path, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()


print("GloVe zip file downloaded successfully.")

# Unzip the contents
print("Extracting GloVe zip file...")
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    num_files = len(zip_ref.namelist())
    progress_bar = tqdm(total=num_files)
    for file in zip_ref.namelist():
        zip_ref.extract(file, data_dir)
        progress_bar.update(1)
    progress_bar.close()
print("GloVe zip file extracted successfully.")

print("Downloading analogy file")
analogy_file_path = data_dir / "questions-words.txt"
url2 = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
if not analogy_file_path.is_file():
    response = requests.get(url2, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(analogy_file_path, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()