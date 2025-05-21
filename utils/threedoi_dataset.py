import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


import pdb


def init_3doi(base_image_dir, aff_type):
    with open("annotations/train/3doi.json", "r") as f:
        anno_3doi = json.load(f)

    tdoi_questions = []
    tdoi_answers = []
    tdoi_labels_path = []
    tdoi_images_path = []
    for item in anno_3doi:
        label_path = os.path.join(base_image_dir, "GT_gaussian", item["gt_path"])
        tdoi_labels_path.append(label_path)

        object = item["object"]
        question = f"<image>\nWhere should I interact with the {object}?"
        questions = question + " Please output segmentation mask."
        answers = "You can interact with the highlighted area" + " " + "[SEG]" + "."
        tdoi_questions.append(questions)
        tdoi_answers.append(answers)

        img_name = item["img_name"]
        image_path = os.path.join(base_image_dir, "images", img_name)
        tdoi_images_path.append(image_path)

    print("3doi: ", len(tdoi_images_path))
    return tdoi_images_path, tdoi_labels_path, tdoi_questions, tdoi_answers


class DoiDataset(torch.utils.data.Dataset):
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
        data_name="3doi",
        aff_type="mp",
    ):
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = os.path.join(base_image_dir, "3doi_data")
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
        images_path, labels_path, questions, answers = eval("init_{}".format(ds))(
            self.base_image_dir, aff_type
        )
        self.data2list[ds] = (images_path, labels_path)
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
        images_path, labels_path = self.data2list[self.data_name]
        questions, answers = self.data2texts[self.data_name]
        idx = random.randint(0, len(images_path) - 1)
        image_path = images_path[idx]

        img = cv2.imread(image_path)
        width = img.shape[1]
        height = img.shape[0]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        label_path = labels_path[idx]

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        question = questions[idx]
        answer = answers[idx]

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
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            resize,
            question,
        )
