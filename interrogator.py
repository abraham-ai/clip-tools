import sys
sys.path.append('BLIP/.')

import os
import torch
import clip_model
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
blip_image_eval_size = 384
blip_model = None
prompts_data_setup = False

def load_list(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def get_prompts_data(data_path):
    global artists, flavors, mediums, movements, sites, trending_list, prompts_data_setup
    if not prompts_data_setup:
        artists = load_list(os.path.join(data_path, 'artists.txt'))
        flavors = load_list(os.path.join(data_path, 'flavors.txt'))
        mediums = load_list(os.path.join(data_path, 'mediums.txt'))
        movements = load_list(os.path.join(data_path, 'movements.txt'))
        sites = load_list(os.path.join(data_path, 'sites.txt'))
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])
        prompts_data_setup = True

def setup_blip():
    global blip_model
    if blip_model is None:
        blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
        blip_model = blip_decoder(med_config='BLIP/configs/med_config.json', pretrained=blip_model_url, image_size=blip_image_eval_size, vit='base')
        blip_model.eval()
        blip_model = blip_model.to(device)
    
def generate_caption(pil_image):
    global blip_model, blip_image_eval_size
    gpu_image = T.Compose([
        T.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

def rank(clip, image_features, text_array, top_count=1):
    text_features = clip_model.encode_text(text_array, clip)
    top_count = min(top_count, len(text_array))
    similarity = torch.zeros((1, len(text_array))).to(device)
    for i in range(image_features.shape[0]):
        similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
    similarity /= image_features.shape[0]
    top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)  
    return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

def interrogate(image, clip_models):
    global artists, flavors, mediums, movements, sites, trending_list
    caption = generate_caption(image)    
    bests = [[('',0)]]*5
    for clip in clip_models:
        image_features = clip_model.encode_image(image, clip)
        ranks = [
            rank(clip, image_features, mediums),
            rank(clip, image_features, ["by "+artist for artist in artists]),
            rank(clip, image_features, trending_list),
            rank(clip, image_features, movements),
            rank(clip, image_features, flavors, top_count=3)
        ]
        for i in range(len(ranks)):
            confidence_sum = 0
            for ci in range(len(ranks[i])):
                confidence_sum += ranks[i][ci][1]
            if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                bests[i] = ranks[i]
    flaves = ', '.join([f"{x[0]}" for x in bests[4]])
    medium = bests[0][0][0]
    if caption.startswith(medium):
        final_caption = f'{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}'
    else:
        final_caption = f'{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}'
    return final_caption

def setup_interrogator(clip_models_enabled):
    clip_model.setup(clip_models_enabled)
    setup_blip()
    get_prompts_data('prompt_data')

def main():
    setup_interrogator(['ViT-B/32', 'ViT-B/16', 'RN50'])
    image = clip_model.load_image_path_or_url("https://cdnb.artstation.com/p/assets/images/images/032/142/769/large/ignacio-bazan-lazcano-book-4-final.jpg")
    final_caption = interrogate(image, clip_models_enabled)
    print(final_caption)

if __name__ == "__main__":
    main()
