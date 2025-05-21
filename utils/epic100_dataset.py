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


def init_epic100(base_image_dir, aff_type):
    if aff_type == "mp":
        aff_dir = "GT_gaussian"
    elif aff_type == "sp" or aff_type == "adapt":
        aff_dir = "GT_gaussian_sp"
    else:
        raise ValueError("epic100: aff_type should be mp or sp or adapt")

    with open("annotations/train/epic100.json", "r") as f:
        annos = json.load(f)

    epic_questions = []
    epic_answers = []
    epic_labels_path = []
    epic_imgs_path = []

    for item in annos:
        label_path = item["gt_path"]
        object = item["noun"]
        if ":" in object:
            continue
        label_path = os.path.join(base_image_dir, aff_dir, label_path)
        epic_labels_path.append(label_path)

        img_path = os.path.join(base_image_dir, item["img_path"])
        epic_imgs_path.append(img_path)

        action = item["verb"]

        question = f"<image>\nWhere should I interact with the {object} to {action} it?"
        questions = question + " Please output segmentation mask."
        answers = "You can interact with the highlighted area" + " " + "[SEG]" + "."

        epic_questions.append(questions)
        epic_answers.append(answers)

    print("epic100: ", len(epic_imgs_path))
    return epic_imgs_path, epic_labels_path, epic_questions, epic_answers


class Epic100Dataset(torch.utils.data.Dataset):
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
        data_name="epic100",
        aff_type="mp",
    ):
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = os.path.join(
            base_image_dir, "epic-100/EPIC-KITCHENS_frames"
        )
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
        imgs_path, labels_path, questions, answers = eval("init_{}".format(ds))(
            self.base_image_dir, aff_type
        )
        self.data2list[ds] = (imgs_path, labels_path)
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
        imgs_path, labels_path = self.data2list[self.data_name]
        questions, answers = self.data2texts[self.data_name]
        idx = random.randint(0, len(imgs_path) - 1)
        img_path = imgs_path[idx]
        label_path = labels_path[idx]

        label = Image.open(label_path)
        label = np.array(label)

        question = questions[idx]
        answer = answers[idx]

        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.from_numpy(label / 255.0)
        masks = masks.unsqueeze(0)

        return (
            img_path,
            image,
            image_clip,
            conversations,
            masks,
            resize,
            question,
        )
