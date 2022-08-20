import os
import torch
import sklearn.metrics.pairwise
from tqdm import tqdm
from pymongo import MongoClient

import clip_model

MONGO_URL = os.getenv('MONGO_URL')
MONGO_DB = os.getenv('MONGO_DB')
MINIO_URL = os.getenv('MINIO_URL')
BUCKET_NAME = os.getenv('BUCKET_NAME')

clip_model.setup(['ViT-B/32'])

client = MongoClient(MONGO_URL)
db = client[MONGO_DB]
collection = db.creations

features = {}
for doc in tqdm(collection.find()):
    if 'sha' not in doc:
        continue
    sha = doc['sha']
    sha_url = f'{MINIO_URL}/{BUCKET_NAME}/{sha}'
    image = clip_model.load_image_path_or_url(sha_url)
    features[sha] = clip_model.encode_image(image, 'ViT-B/32').cpu()

keys = list(features.keys())
feats = torch.cat([features[f] for f in features])
sim = sklearn.metrics.pairwise.cosine_similarity(feats)
np.fill_diagonal(sim, 0)

idx = np.argsort(-sim.flatten())
for i in idx[0:1000]:
    x, y = int(i % len(feats)), int(i / len(feats))
    sha1, sha2 = keys[x], keys[y]
    sha_url1 = f'{minio_url}/{bucket_name}/{sha1}'
    sha_url2 = f'{minio_url}/{bucket_name}/{sha2}'
    print(sha_url1, sha_url2)


    




import multiprocessing
from multiprocessing.dummy import Pool