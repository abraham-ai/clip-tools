from pymongo import MongoClient
import clip
from PIL import Image
import requests
import torch
from tqdm import tqdm
import sklearn.metrics.pairwise

clip_models = {}

# options: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'ViT-L/14@336px','RN101','RN50x4', 'RN50x16', 'RN50x64'
def setup(clip_models_enabled):
    global clip_models
    for clip_model in clip_models_enabled:
        if clip_model not in clip_models:
            model, preprocess = clip.load(clip_model)
            model.cuda().eval()
            clip_models[clip_model] = (model, preprocess)

def load_image_path_or_url(path):
    if str(path).startswith('http://') or str(path).startswith('https://'):
        image = requests.get(path, stream=True)
        image = Image.open(image.raw).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')
    return image

def encode_image(image, clip_model):
    if clip_model not in clip_models:
        raise Exception(f"{clip_model} not loaded")
    model, preprocess = clip_models[clip_model]
    image = preprocess(image).unsqueeze(0).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def encode_text(text_array, clip_model):
    if clip_model not in clip_models:
        raise Exception(f"{clip_model} not loaded")
    model, preprocess = clip_models[clip_model]
    text_tokens = clip.tokenize([text for text in text_array]).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features
