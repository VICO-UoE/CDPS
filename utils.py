import argparse
import os
import torch
import random
import numpy as np

# import open_clip
import CDPS_clip as open_clip
import json
from matplotlib import pyplot as plt

OPENCLIP_MODEL_DIC = {
    "laion400m": {
        "vitb32": ("laion400m_e32", "ViT-B-32-quickgelu"),
        "vitb16": ("laion400m_e32", "ViT-B-16"),
        "vitl14": ("laion400m_e32", "ViT-L-14"),
    },
    "openai": {
        "vitb32": ("openai", "ViT-B-32-quickgelu"),
        "vitb16": ("openai", "ViT-B-16"),
        "vitl14": ("openai", "ViT-L-14"),
        "rn50": ("openai", "RN50"),
    },
    "laion2b": {
        "vitb32": ("laion2b_s34b_b79k", "ViT-B-32"),
        "vitb16": ("laion2b_s34b_b88k", "ViT-B-16"),
        "vitl14": ("laion2b_s32b_b82k", "ViT-L-14"),
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configs for CDPS")
    parser.add_argument("--dataset", type=str, default="fgvc-aircraft")
    parser.add_argument("--model_cfg", type=str, default="vitb16_openclip_laion2b")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fewshot", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_confuse", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--cache_root", type=str, default="./caches")
    parser.add_argument("--output_file", type=str, default="results.txt")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def get_models(args, mode="val"):
    model_cfg = args.model_cfg
    device = args.device

    cfgs = model_cfg.split("_")
    arch = cfgs[0]
    model_name = cfgs[1]
    pretraining_dataset = cfgs[2] if len(cfgs) == 3 else None

    corpus_config, model_arch = OPENCLIP_MODEL_DIC[pretraining_dataset][arch]
    model, train_preprocess, preprocess = open_clip.create_model_and_transforms(
        model_arch, pretrained=corpus_config, cache_dir="./pretrained_models"
    )
    tokenizer = open_clip.get_tokenizer(model_arch)

    # not using mixed precision
    model.float()
    model.to(device)

    if mode == "train":
        return model, train_preprocess, preprocess, tokenizer
    elif mode == "val":
        return model, preprocess, tokenizer
    else:
        raise NotImplementedError


def prompt_maker(dataset_name, name_type, metrics):
    prompts = {}
    if dataset_name in ["ucf101", "caltech101", "sun397"]:
        name_type = "name"
    if dataset_name == "semi-aves":
        name_type = "most_common_name"
        prompt_templates = TEMPLATES_DIC[dataset_name][name_type]
    else:
        prompt_templates = TEMPLATES_DIC[dataset_name]

    for i, key in enumerate(metrics.keys()):
        label = metrics[key][name_type]
        if name_type == "alternates":
            prompt_lst = []
            for alt_name, ct in label.items():
                alt_name = alt_name.replace("_", " ")
                pt = [template.format(alt_name) for template in prompt_templates]
                prompt_lst.extend(pt)
            prompts[key] = {"corpus": prompt_lst}
        else:
            label = label.replace("_", " ")
            prompts[key] = {
                "corpus": [template.format(label) for template in prompt_templates]
            }
    prompts = dict(sorted(prompts.items(), key=lambda x: x[0]))
    return prompts


def operate_on_prompt(model, text, tokenizer):
    tokens = tokenizer(text)
    features = model.encode_text(tokens.cuda())
    model.transformer.feats_each_layer.clear()
    model.transformer.attn_each_layer.clear()
    features /= features.norm(dim=-1, keepdim=True)  # Normalization.
    return features


@torch.no_grad()
def get_text_features(model, prompt_dict, tokenizer):
    model.eval()
    tensor_dict = {}
    num_classes = len(prompt_dict)
    for i in range(num_classes):
        key = str(i)
        info = prompt_dict[key]
        source = {}
        prompts = []
        for prompt in info["corpus"]:
            prompts.append(prompt)

        stacked_tensor = operate_on_prompt(model, prompts, tokenizer)
        stacked_tensor.cpu()

        source["all"] = stacked_tensor
        mean_tensor = torch.mean(stacked_tensor, dim=0)
        mean_tensor /= mean_tensor.norm(dim=-1, keepdim=True)
        source["mean"] = mean_tensor
        tensor_dict[key] = source
    return tensor_dict


def get_prompts(args, tokenizer, model):
    prompt_name = args.prompt_name
    text_prompts = prompt_maker(args.dataset, prompt_name, args.metrics)
    prompt_tensors = get_text_features(model, text_prompts, tokenizer)
    return prompt_tensors


def verify(comp_array, save_file, num_files):
    save_list = []
    for i in range(num_files):
        save_list.append(torch.load(save_file.format(i)))
    save_array = torch.cat(save_list, dim=0)
    print("Verification:", torch.abs(comp_array.cpu() - save_array.cpu()).max())
    return


def patch_visualization(dataset, pick_indexes, vis_dir):

    for i in range(len(dataset)):
        if i > 400:
            break
        vis_img = dataset.get_vis_img(i)
        indexes = pick_indexes[i]
        h, w = vis_img.shape[1:]
        vis_mask = np.zeros((h, w, 4))
        for idx in indexes:
            row = idx // 14
            col = idx % 14
            vis_mask[row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16, 0] = 1.0
            vis_mask[row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16, 1] = 0.25
            vis_mask[row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16, 3] = 0.5
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(np.transpose(vis_img, (1, 2, 0)))
        ax.imshow(vis_mask)
        plt.axis("off")
        plt.savefig(
            os.path.join(vis_dir, f"{i}.png"), bbox_inches="tight", pad_inches=0
        )
        plt.close()

    return


caltech101_templates = [
    "a photo of a {}.",
]

describabletextures_templates = [
    "a photo of a {} texture.",
    "a photo of a {} pattern.",
    "a photo of a {} thing.",
    "a photo of a {} object.",
    "a photo of the {} texture.",
    "a photo of the {} pattern.",
    "a photo of the {} thing.",
    "a photo of the {} object.",
]

eurosat_templates = [
    "a centered satellite photo of {}.",
    "a centered satellite photo of a {}.",
    "a centered satellite photo of the {}.",
]

fgvcaircraft_templates = [
    "a photo of a {}, a type of aircraft.",
    "a photo of the {}, a type of aircraft.",
]

flowers102_templates = [
    "a photo of a {}, a type of flower.",
]

food101_templates = [
    "a photo of {}, a type of food.",
]

oxfordpets_templates = [
    "a photo of a {}, a type of pet.",
]

sun397_templates = [
    "a photo of a {}.",
]

# stanfordcars_templates = [
#    "a photo of a {}.",
#    "a photo of the {}.",
#    "a photo of my {}.",
#    "i love my {}!",
#    "a photo of my dirty {}.",
#    "a photo of my clean {}.",
#    "a photo of my new {}.",
#    "a photo of my old {}.",
# ]
stanfordcars_templates = ["a photo of a {}, a type of car."]

imagenet_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


semi_aves_templates = {
    "s-name": ["a photo of a {}, a type of bird."],
    "c-name": ["a photo of a {}, a type of bird."],
    "t-name": ["a photo of a {}, a type of bird, commonally known as {}."],
    "f-name": ["a photo of a {}, a type of bird."],
    "most_common_name": ["a photo of a {}, a type of bird."],
    "alternates": ["a photo of a {}, a type of bird."],
    "most_common_name_REAL": ["a photo of a {}, a type of bird."],
    "name": ["a photo of a {}, a type of bird."],
    # 'name': imagenet_templates, # openai templates
    "c-name-80prompts": imagenet_templates,
}

ucf101_templates = [
    "a photo of a person doing {}.",
]

TEMPLATES_DIC = {
    "imagenet": imagenet_templates,
    "imagenet_1k": imagenet_templates,
    "imagenet_1k_mined": imagenet_templates,
    "flowers102": flowers102_templates,
    "food101": food101_templates,
    "stanford_cars": stanfordcars_templates,
    "stancars": stanfordcars_templates,
    "fgvc-aircraft": fgvcaircraft_templates,
    "oxford_pets": oxfordpets_templates,
    "imagenet_v2": imagenet_templates,
    "dtd": describabletextures_templates,
    "dtd_selected": describabletextures_templates,
    "semi-aves": semi_aves_templates,
    "caltech101": caltech101_templates,
    "eurosat": eurosat_templates,
    "sun397": sun397_templates,
    "ucf101": ucf101_templates,
}
