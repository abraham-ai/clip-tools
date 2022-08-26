import argparse
import os
# import random
import base64
import hashlib
import PIL
# import moviepy.editor as mpy
# from io import BytesIO

# from minio import Minio
# from minio.error import S3Error

# from settings import StableDiffusionSettings
# import generation
from interrogator import *

from eden.block import Block
from eden.hosting import host_block
from eden.datatypes import Image

eden_block = Block()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-workers', help='maximum number of workers to be run in parallel', required=False, default=1, type=int)
parser.add_argument('-p', '--port', help='localhost port', required=False, type=int, default=5656)
parser.add_argument('-rh', '--redis-host', help='redis host', required=False, type=str, default='localhost')
parser.add_argument('-rp', '--redis-port', help='redis port', required=False, type=int, default=6379)
parser.add_argument('-l', '--logfile', help='filename of log file', required=False, type=str, default=None)
args = parser.parse_args()

# minio_url = os.environ['MINIO_URL']
# minio_bucket_name = os.environ['MINIO_BUCKET_NAME']
# minio_access_key = os.environ['MINIO_ACCESS_KEY']
# minio_secret_key = os.environ['MINIO_SECRET_KEY']

# minio_client = Minio(
#     minio_url,
#     access_key=minio_access_key,
#     secret_key=minio_secret_key
# )

# def get_file_sha256(filepath): 
#     sha256_hash = hashlib.sha256()
#     with open(filepath,"rb") as f:
#         for byte_block in iter(lambda: f.read(4096),b""):
#             sha256_hash.update(byte_block)
#         sha = sha256_hash.hexdigest()
#     return sha

def b64str_to_PIL(data):
    data = data.replace('data:image/png;base64,', '')
    pil_img = PIL.Image.open(BytesIO(base64.b64decode(data)))
    return pil_img


my_args = {
    "mode": "interrogate",
    "image_url": ""
}

@eden_block.run(args=my_args)
def run(config):
    
    mode = config["mode"]
    assert(mode in ["interrogate", "nsfw", "embed"], \
        f"Error: mode {mode} not recognized (interrogate, nsfw, embed allowed)")

    if mode == "interrogate":
        clip_models_enabled = ['ViT-B/32', 'ViT-B/16', 'RN50']
        setup_interrogator(clip_models_enabled)
        image_url = config["image_url"]
        image = clip_model.load_image_path_or_url(image_url)
        final_caption = interrogate(image, clip_models_enabled)
        results = {"caption": final_caption}

    elif mode == "nsfw":
        image_url = config["image_url"],
        pass
    elif mode == "embed":
        pass
        
    print("Results", results)
    return results


host_block(
    block = eden_block,
    port = args.port,
    host = "0.0.0.0",
    max_num_workers = args.num_workers,
    redis_port = args.redis_port,
    redis_host = args.redis_host,
    logfile = args.logfile, 
    log_level = 'debug',
    requires_gpu = True
)
