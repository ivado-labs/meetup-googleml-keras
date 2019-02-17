from tensorflow.python.lib.io import file_io
from google.cloud import storage
import os


def copy_file_to_gcs(local_path, gcs_path):
    with file_io.FileIO(local_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(gcs_path, os.path.basename(local_path)), mode='w+') as output_f:
            output_f.write(input_f.read())


def download_directory(gcloud_project, bucket_name, dir_path, target_dir):
    client = storage.Client(gcloud_project)
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=dir_path)
    for b in blobs:
        file_path = os.path.join(target_dir, b.name)
        if not os.path.exists(file_path):
            if not os.path.exists(os.path.dirname(file_path)):
                print('Downloading:', os.path.dirname(b.name))
                os.makedirs(os.path.dirname(file_path))
            b.download_to_filename(file_path)
