# pylint: disable=missing-docstring

import pickle
from pathlib import Path

import numpy as np
import torch
from imagenet_labels import IMAGENET_LABELS
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import (AutoProcessor, Blip2Processor,
                          Blip2TextModelWithProjection,
                          Blip2VisionModelWithProjection)


class TextDatasetFromList(Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)


def main():
    data_dir = Path('/mnt/data/imagenet/ilsvrc2012/')
    print("Loading ImageNet dataset")
    imagenet_dataset = datasets.ImageNet(
        root=data_dir,
        split='val',
    )
    print("Loaded ImageNet dataset")

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

    print("Loading text models")
    model = Blip2TextModelWithProjection.from_pretrained(
        "Salesforce/blip2-itm-vit-g",
        device_map="auto",
        torch_dtype=torch.float16,
    ).cuda()
    processor = AutoProcessor \
        .from_pretrained("Salesforce/blip2-itm-vit-g")

    print("Getting text embeddings")
    # Get text embeddings.
    dataset = TextDatasetFromList(texts)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4)
    all_embeds = []
    for batch in tqdm(data_loader):
        inputs = processor(text=batch, padding=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds.detach().cpu().numpy()[:,0,:]
        all_embeds.append(text_embeds)
    text_embeds = np.concatenate(all_embeds, axis=0)
    print(text_embeds.shape, len(texts))
    # print("Tokenizing")
    # print(len(texts))
    # inputs = tokenizer(texts, padding=True, return_tensors="pt")
    # print("Sending tokens to GPU")
    # inputs = {k: v.cuda() for k, v in inputs.items()}
    # print("Getting embeddings")
    # outputs = model(**inputs)
    # print("Detaching")
    # text_embeds = outputs.text_embeds.detach().cpu().numpy()

    print("Saving text embeddings to embeds_texts_imagenet32_blip.pkl")
    # Save embeddings and y_real to pickle.
    output = {
        #'image_embeds': image_embeds,
        'text_embeds': text_embeds,
        #'y_real': y_real,
        'classes': classes,
        'templates': templates,
        'texts': texts,
    }

    del model
    del processor

    with open('embeds_texts_imagenet32_blip.pkl', 'wb') as f:
        pickle.dump(output, f)

    del text_embeds
    del output
    del templates

    print("Loading vision models")
    vision_model = Blip2VisionModelWithProjection \
        .from_pretrained("Salesforce/blip2-itm-vit-g", device_map="auto").cuda()
    processor = AutoProcessor \
        .from_pretrained("Salesforce/blip2-itm-vit-g")

    # Get vision embeddings.
    image_embeds = []
    y_real = []

    def get_one_embedding(image, vision_model):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = vision_model(**inputs)
        return outputs.image_embeds.detach().cpu().numpy()

    print("Getting vision embeddings")
    for image, label in tqdm(imagenet_dataset):
        y_real.append(label)
        image_embeds.append(get_one_embedding(image, vision_model))

    # Save embeddings and y_real to pickle.
    output = {
        'image_embeds': image_embeds,
        #'text_embeds': text_embeds,
        'y_real': y_real,
        #'classes': classes,
        #'templates': templates,
        #'texts': texts,
    }

    print("Saving image embeddings to embeds_images_imagenet32_blip.pkl")

    with open('embeds_images_imagenet32_blip.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    main()
