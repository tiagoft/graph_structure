# pylint: disable=missing-docstring

from pathlib import Path
import numpy as np
from tqdm import tqdm

from imagenet_labels import IMAGENET_LABELS

import pickle
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer, CLIPModel,
                          CLIPProcessor, CLIPTextModelWithProjection,
                          CLIPVisionModelWithProjection)

# import pickle

# from torchvision import datasets
# from tqdm import tqdm
# from transformers import (AutoProcessor, AutoTokenizer,
#                           CLIPTextModelWithProjection,
#                           CLIPVisionModelWithProjection)

def main():
    data_dir = Path('/mnt/data/imagenet')

    # Preprocess data batches
    all_data = {
        'labels': [],
        'data': [],
    }

    print("Loading data batches")
    for i in tqdm(range(1, 11)):
        A = np.load(data_dir / f'train_data_batch_{i}', allow_pickle=True)
        all_data['data'].append(A['data'])
        all_data['labels'].append(A['labels'])

    print("Concatenating data batches")
    all_data['data'] = np.concatenate(all_data['data'], axis=0)
    all_data['labels'] = np.concatenate(all_data['labels'], axis=0)
    all_data['data'] = all_data['data'].reshape(-1, 3, 32, 32)
    all_data['data'] = np.transpose(all_data['data'], (0, 2, 3, 1))

    classes = [IMAGENET_LABELS[n] for n in range(1000)]



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

    print("Loading models")
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

    print("Getting vision embeddings")
    for idx in tqdm(range(len(all_data['data']))):
        image = all_data['data'][idx]
        label = all_data['labels'][idx]
        inputs = processor(images=image, return_tensors="pt")
        outputs = vision_model(**inputs)
        image_embeds.append(outputs.image_embeds.detach().numpy())
        y_real.append(label)

    print("Getting text embeddings")
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

    print("Saving embeddings to embeds_imagenet32.pkl")
    
    with open('embeds_imagenet32.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    main()
