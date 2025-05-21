import argparse
import os
import sys

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from utils.visualizer import draw_affordance_center
import pdb
import json
from tqdm import tqdm
import pdb


# 理想值为 0
def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12, is_part=False) -> np.ndarray:
    map1 = pred / (pred.sum() + eps)
    if is_part:
        map2 = gt
    else:
        map2 = gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)


def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if image[xx][yy] > threshold:
                output[xx][yy] = 1
    return output


def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / std
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)

    return nss


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLOVER++ chat")
    parser.add_argument("--version", default="/path/to/GLOVER++/model", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--use_text_emb_in_suffix_sam",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--dataset_dir",
        default="/path/to/HOVA-500K/dataset",
        type=str,
    )
    parser.add_argument(
        "--model_dir",
        default="/path/to/GLOVER++/model",
        type=str,
    )
    parser.add_argument(
        "--model_arch",
        default="glover++",
        type=str,
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def add_data(img_paths, gt_paths, mask_part_paths, actions, nouns, root_dir):
    with open("annotations/test/handal_2000_sp.json", "r") as f:
        test_json = json.load(f)
    for item in test_json:
        img_paths.append(
            os.path.join(
                root_dir,
                "HANDAL/images",
                item["img_path"],
            )
        )
        gt_paths.append(
            os.path.join(
                root_dir,
                "HANDAL/annotations/GT_gaussian_test_2000_sp",
                item["gt_path"],
            )
        )
        mask_part_paths.append(
            os.path.join(
                root_dir,
                "HANDAL/images",
                item["mask_path"],
            )
        )
        actions.append("pick up")
        nouns.append(item["noun"])

    with open("annotations/test/ego4d.json", "r") as f:
        test_json = json.load(f)
    for item in test_json:
        img_paths.append(
            os.path.join(
                root_dir,
                "Ego4D/frames_copy",
                item["img_path"] + ".jpg",
            )
        )
        gt_paths.append(
            os.path.join(root_dir, "Ego4D/GT_gaussian_sp_70", item["gt_path"])
        )
        action = item["action"]
        if "(" in action:
            action = action.split("(")[1].split(")")[0].split(",")
            action = [act.strip().replace("_", "") for act in action]
            action = action[0]
        actions.append(action)
        noun = item["noun"]
        if "(" in noun:
            noun = noun.split("(")[1].split(")")[0].split(",")
            noun = [obj.strip().replace("_", "") for obj in noun]
            noun = noun[0]
        nouns.append(noun)

    with open("annotations/test/epic100_2000.json", "r") as f:
        test_json = json.load(f)
    for item in test_json:
        img_paths.append(
            os.path.join(
                root_dir,
                "epic-100/EPIC-KITCHENS_frames",
                item["img_path"],
            )
        )
        gt_paths.append(
            os.path.join(
                root_dir,
                "epic-100/EPIC-KITCHENS_frames/GT_gaussian_sp/test",
                item["gt_path"],
            )
        )
        actions.append(item["verb"])
        nouns.append(item["noun"])


def main(args):
    args = parse_args(args)
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_arch == "glover++":
        from model.GLOVER_plus import GloverForCausalLM
    elif args.model_arch == "glover":
        from model.GLOVER import GloverForCausalLM

    model = GloverForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    img_paths = []
    gt_paths = []
    mask_part_paths = []
    actions = []
    nouns = []
    root_dir = args.dataset_dir
    add_data(img_paths, gt_paths, mask_part_paths, actions, nouns, root_dir)

    KLs = []
    SIM = []
    NSS = []
    SIM_PART = []

    for i in tqdm(range(len(img_paths))):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        object = nouns[i]
        action = actions[i]
        prompt = f"<image>\nWhere should I interact with the {object} to {action} it?"
        prompt = prompt + " Please output segmentation mask."
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = img_paths[i]
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        gt_path = gt_paths[i]
        gt_mask = Image.open(gt_path)
        gt_mask = np.array(gt_mask) / 255.0

        if i < 2000:
            mask_part_path = mask_part_paths[i]
            mask_part = Image.open(mask_part_path)
            mask_part = np.array(mask_part) / 255.0

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
            **(
                {"use_text_emb_in_suffix_sam": args.use_text_emb_in_suffix_sam}
                if args.model_arch == "glover++"
                else {}
            ),
        )
        # pdb.set_trace()
        # output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        # text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        # text_output = text_output.replace("\n", "").replace("  ", " ")
        # print("text_output: ", text_output)

        for j, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask[0].sigmoid()
            pred_mask = pred_mask.detach().cpu().numpy()

            if j < 2000:
                _, arg_mask = draw_affordance_center(pred_mask, image_np, radius=50)
            elif j < 4000:
                _, arg_mask = draw_affordance_center(pred_mask, image_np, radius=70)
            else:
                _, arg_mask = draw_affordance_center(pred_mask, image_np, radius=20)

            kld, sim, nss = (
                cal_kl(pred_mask, gt_mask),
                cal_sim(pred_mask, gt_mask),
                cal_nss(pred_mask, gt_mask),
            )
            if i < 2000:
                sim_part = cal_sim(arg_mask, mask_part, is_part=True)
                SIM_PART.append(sim_part)

            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)

        if i % 2000 == 0 and i != 0:
            print("mKLD: ", sum(KLs) / len(KLs))
            print("mSIM: ", sum(SIM) / len(SIM))
            print("mNSS: ", sum(NSS) / len(NSS))
            print("mSIM_PART: ", sum(SIM_PART) / len(SIM_PART))
    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)
    mSIM_PART = sum(SIM_PART) / len(SIM_PART)

    print("mKLD: ", mKLD)
    print("mSIM: ", mSIM)
    print("mNSS: ", mNSS)
    print("mSIM_PART: ", mSIM_PART)
    with open("eval_results.txt", "a") as f:
        f.write("model name: %s\n" % args.version.split("/")[-1])
        f.write("mKLD: %s\n" % str(mKLD))
        f.write("mSIM: %s\n" % str(mSIM))
        f.write("mNSS: %s\n" % str(mNSS))
        f.write("mSIM_PART: %s\n" % str(mSIM_PART))
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
