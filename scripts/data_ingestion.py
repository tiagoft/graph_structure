# pylint: disable=missing-docstring

# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import requests
# import scipy.stats as st
# import torch
# from PIL import Image
# from sklearn.manifold import TSNE
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics.pairwise import (pairwise_distances,
#                                       pairwise_distances_argmin)
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from tqdm import tqdm
# from transformers import (AutoProcessor, AutoTokenizer, CLIPModel,
#                           CLIPProcessor, CLIPTextModelWithProjection,
#                           CLIPVisionModelWithProjection)

import pickle

from torchvision import datasets
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer,
                          CLIPTextModelWithProjection,
                          CLIPVisionModelWithProjection)


def main():

    dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=None,
    )

    # Taken from: https://github.com/openai/CLIP/blob/main/data/prompts.md .

    classes = [
        'apple',
        'aquarium fish',
        'baby',
        'bear',
        'beaver',
        'bed',
        'bee',
        'beetle',
        'bicycle',
        'bottle',
        'bowl',
        'boy',
        'bridge',
        'bus',
        'butterfly',
        'camel',
        'can',
        'castle',
        'caterpillar',
        'cattle',
        'chair',
        'chimpanzee',
        'clock',
        'cloud',
        'cockroach',
        'couch',
        'crab',
        'crocodile',
        'cup',
        'dinosaur',
        'dolphin',
        'elephant',
        'flatfish',
        'forest',
        'fox',
        'girl',
        'hamster',
        'house',
        'kangaroo',
        'keyboard',
        'lamp',
        'lawn mower',
        'leopard',
        'lion',
        'lizard',
        'lobster',
        'man',
        'maple tree',
        'motorcycle',
        'mountain',
        'mouse',
        'mushroom',
        'oak tree',
        'orange',
        'orchid',
        'otter',
        'palm tree',
        'pear',
        'pickup truck',
        'pine tree',
        'plain',
        'plate',
        'poppy',
        'porcupine',
        'possum',
        'rabbit',
        'raccoon',
        'ray',
        'road',
        'rocket',
        'rose',
        'sea',
        'seal',
        'shark',
        'shrew',
        'skunk',
        'skyscraper',
        'snail',
        'snake',
        'spider',
        'squirrel',
        'streetcar',
        'sunflower',
        'sweet pepper',
        'table',
        'tank',
        'telephone',
        'television',
        'tiger',
        'tractor',
        'train',
        'trout',
        'tulip',
        'turtle',
        'wardrobe',
        'whale',
        'willow tree',
        'wolf',
        'woman',
        'worm',
    ]

    templates = [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ]

    texts = []

    for c in classes:
        for t in templates:
            texts.append(t.format(c))
    print(texts)

    vision_model = CLIPVisionModelWithProjection \
        .from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor \
        .from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModelWithProjection \
        .from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer \
        .from_pretrained("openai/clip-vit-base-patch32")

    # Get vision embeddings.
    image_embeds = []
    y_real = []
    for image, label in tqdm(dataset):
        inputs = processor(images=image, return_tensors="pt")
        outputs = vision_model(**inputs)
        image_embeds.append(outputs.image_embeds.detach().numpy())
        y_real.append(label)

    # Get text embeddings.
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    text_embeds = outputs.text_embeds.detach().numpy()

    # Save embeddings and y_real to pickle.
    output = {
        'image_embeds': image_embeds,
        'text_embeds': text_embeds,
        'y_real': y_real,
        'classes': classes,
        'templates': templates,
        'texts': texts,
    }
    with open('embeds_cifar100.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    main()
