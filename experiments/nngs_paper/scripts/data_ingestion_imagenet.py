# pylint: disable=missing-docstring

from pathlib import Path
import numpy as np
from tqdm import tqdm

from imagenet_labels import IMAGENET_LABELS

import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import (AutoProcessor, AutoTokenizer, CLIPModel,
                          CLIPProcessor, CLIPTextModelWithProjection,
                          CLIPVisionModelWithProjection)


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
    model = CLIPTextModelWithProjection \
        .from_pretrained("openai/clip-vit-base-patch32").cuda()
    tokenizer = AutoTokenizer \
        .from_pretrained("openai/clip-vit-base-patch32")


    print("Getting text embeddings")
    # Get text embeddings.
    dataset = TextDatasetFromList(texts)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4)
    all_embeds = []
    for batch in tqdm(data_loader):
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds.detach().cpu().numpy()
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

    print("Saving text embeddings to embeds_texts_imagenet32.pkl")
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
    del tokenizer

    with open('embeds_texts_imagenet32.pkl', 'wb') as f:
        pickle.dump(output, f)

    del text_embeds
    del output
    del templates


    print("Loading vision models")
    vision_model = CLIPVisionModelWithProjection \
        .from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = AutoProcessor \
        .from_pretrained("openai/clip-vit-base-patch32")

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


    print("Saving image embeddings to embeds_images_imagenet32.pkl")
   
    with open('embeds_images_imagenet32.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    main()
