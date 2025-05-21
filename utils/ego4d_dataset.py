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


def init_ego4d(base_image_dir, aff_type):
    if aff_type == "mp":
        aff_dir = "GT_gaussian"
    elif aff_type == "sp":
        aff_dir = "GT_gaussian_sp"
    elif aff_type == "adapt":
        aff_dir = "GT_gaussian_sp_70"
    else:
        raise ValueError("Ego4d: aff_type should be mp or sp or adapt")

    with open("annotations/train/ego4d.json", "r") as f:
        ego4d_annos = json.load(f)

    ego4d_questions = []
    ego4d_answers = []
    ego4d_labels = []
    ego4d_images = []

    for item in ego4d_annos:
        label = os.path.join(base_image_dir, aff_dir, item["gt_path"])
        ego4d_labels.append(label)

        image = os.path.join(base_image_dir, "frames_copy", item["img_path"] + ".jpg")
        ego4d_images.append(image)

        object = item["noun"]
        if "(" in object:
            object = object.split("(")[1].split(")")[0].split(",")
            object = [obj.strip().replace("_", "") for obj in object]
            selected_obj = random.choice(object)
        else:
            selected_obj = object

        action = item["action"]
        if "(" in action:
            action = action.split("(")[1].split(")")[0].split(",")
            action = [act.strip().replace("_", "") for act in action]
            selected_act = random.choice(action)
        else:
            selected_act = action

        question = f"<image>\nWhere should I interact with the {selected_obj} to {selected_act} it?"
        questions = question + " Please output segmentation mask."
        answers = "You can interact with the highlighted area" + " " + "[SEG]" + "."

        ego4d_questions.append(questions)
        ego4d_answers.append(answers)

    print("ego4d: ", len(ego4d_images))
    return ego4d_images, ego4d_labels, ego4d_questions, ego4d_answers


class Ego4DDataset(torch.utils.data.Dataset):
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
        data_name="ego4d",
        aff_type="mp",
    ):
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = os.path.join(base_image_dir, "Ego4D")
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
        images, labels, questions, answers = eval("init_{}".format(ds))(
            self.base_image_dir, aff_type
        )

        for img in images:
            if os.path.exists(img) is not True:
                print(img)

        self.data2list[ds] = (images, labels)
        self.data2texts[ds] = (questions, answers)

    def __len__(self):
        return len(self.data2list[self.data_name][0])

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
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
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
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            resize,
            question,
        )
