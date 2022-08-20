import autokeras as ak
from tensorflow.keras.models import load_model
import clip_model
from PIL import Image
import requests
import torch

model_dir_l14 = 'clip_autokeras_binary_nsfw'
model_dir_b32 = 'clip_autokeras_nsfw_b32/'

nsfw_model = None

def setup_nsfw_model(clip):
    global nsfw_model
    model_dir = model_dir_l14 if clip == "ViT-L/14" else model_dir_b32
    nsfw_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
        
def main():
    clip_model.setup(["ViT-L/14"])
    setup_nsfw_model("ViT-L/14")
    url = "https://cdnb.artstation.com/p/assets/images/images/032/142/769/large/ignacio-bazan-lazcano-book-4-final.jpg"
    image = clip_model.load_image_path_or_url(url)
    image_features = clip_model.encode_image(image, "ViT-L/14")
    pred = nsfw_model.predict(image_features.cpu().numpy(), batch_size=1)
    print(pred)

if __name__ == "__main__":
    main()


