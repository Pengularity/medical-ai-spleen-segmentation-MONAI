import os
from monai.apps import download_and_extract

# 1. Define where to store the dataset (in the data/ folder at the root)
root_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# 2. URL of the dataset Spleen (MSD)
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_dir = os.path.join(root_dir, "Task09_Spleen")

# 3. Download and extract the dataset
if __name__ == "__main__":
    print(f"Starting download in : {root_dir}")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)
        print("Extraction successful.")
    else:
        print("The dataset already exists. Ready for the next step.")