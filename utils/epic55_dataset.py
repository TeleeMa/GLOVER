import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
import pdb

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

def init_epic55(base_image_dir):
    with open("annotations/train/epic55.json", "r") as f:
        epic_annos = json.load(f)

    epic_questions = []
    epic_answers = []
    epic_labels = []
    epic_images = []

    for item in epic_annos:
        label = os.path.join(base_image_dir, 'GT_gaussian', item['gt_path'])
        epic_labels.append(label)

        image = os.path.join(base_image_dir, 'frames', item['img_path'])
        epic_images.append(image)

        object = item['noun']
        action = item['verb']

        question = f'<image>\nWhere should I interact with the {object} to {action} it?'
        questions = question + ' Please output segmentation mask.'
        answers = 'You can interact with the highlighted area' + ' ' + '[SEG]'+'.'

        epic_questions.append(questions)
        epic_answers.append(answers)

    print("epic55: ", len(epic_images))
    return epic_images, epic_labels, epic_questions, epic_answers


class EPIC55Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        data_name="ade20k||epic55",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        base_image_dir = os.path.join(base_image_dir, 'EPIC_55')

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2texts = {}

        ds = data_name
        self.data_name = data_name
        images, labels, questions, answers = eval("init_{}".format(ds))(base_image_dir)
        self.data2list[ds] = (images, labels)
        self.data2texts[ds] = (questions, answers)


    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, labels = self.data2list[self.data_name]
        questions, answers = self.data2texts[self.data_name]
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]


        label = Image.open(label_path)
        label = np.array(label)

        question = questions[idx]
        answer = answers[idx]

        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.from_numpy(label/255.0)
        masks = masks.unsqueeze(0)
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            resize,
            question,
        )
    