import os
import numpy as np
import json
import argparse
import random

NUM_SHOTS = [1, 2, 4, 8, 16]
IMAGE_FOLDERS_MAP = {
    "imagenet": ["images/train", "images/val"],
    "caltech101": ["101_ObjectCategories", "101_ObjectCategories"],
    "oxford_pets": ["images", "images"],
    "stanford_cars": ["", ""],
    "flowers102": ["jpg", "jpg"],
    "food101": ["images", "images"],
    "fgvc-aircraft": ["images", "images"],
    "sun397": ["images", "images"],
    "dtd": ["images", "images"],
    "eurosat": ["EuroSAT_RGB", "EuroSAT_RGB"],
    "ucf101": ["ucf-101-midframes", "ucf-101-midframes"],
    "semi-aves": ["trainval_images", "test"],
}

parser = argparse.ArgumentParser(description="Generate Fewshot Indexes")
parser.add_argument("--dataset", type=str, default="fgvc-aircraft")
parser.add_argument("--data_root", type=str, default="./datasets")
parser.add_argument("--meta_root", type=str, default="./meta_data")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

if args.dataset == "stanford_cars":
    root_path = os.path.join(args.data_root, args.dataset)
    test_root = os.path.join(args.data_root, args.dataset)
else:
    root_path = os.path.join(
        args.data_root, args.dataset, IMAGE_FOLDERS_MAP[args.dataset][0]
    )
    test_root = os.path.join(
        args.data_root, args.dataset, IMAGE_FOLDERS_MAP[args.dataset][1]
    )

if args.dataset not in ["imagenet", "semi-aves", "fgvc-aircraft"]:
    json_file = os.path.join(args.data_root, args.dataset, "split.json")
    with open(json_file, "r") as f:
        infos = json.load(f)
    labels = []
    img_pathes = []
    for train_item in infos["train"]:
        img_pathes.append(f"{root_path}/{train_item[0]}")
        labels.append(train_item[1])
    test_labels = []
    test_im_pathes = []
    for test_item in infos["test"]:
        test_im_pathes.append(f"{test_root}/{test_item[0]}")
        test_labels.append(test_item[1])
elif args.dataset == "fgvc-aircraft":
    label_file = os.path.join(args.data_root, args.dataset, "variants.txt")
    label_map = {}
    with open(label_file, "r") as f:
        lines = f.readlines()
        for label, line in enumerate(lines):
            line = line.strip()
            label_map[line] = label

    labels = []
    img_pathes = []
    train_file = os.path.join(args.data_root, args.dataset, "images_variant_train.txt")
    with open(train_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            file, name = line.strip().split(None, 1)
            file = file + ".jpg"
            labels.append(label_map[name])
            img_pathes.append(os.path.join(f"{root_path}/{file}"))

    test_labels = []
    test_im_pathes = []
    test_file = os.path.join(args.data_root, args.dataset, "images_variant_test.txt")
    with open(test_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            file, name = line.strip().split(None, 1)
            file = file + ".jpg"
            test_labels.append(label_map[name])
            test_im_pathes.append(os.path.join(f"{test_root}/{file}"))

elif args.dataset == "semi-aves":
    subfolders = os.listdir(root_path)
    img_pathes = []
    labels = []
    for subfolder in subfolders:
        subfolder = os.path.join(root_path, subfolder)
        if os.path.isdir(subfolder):
            files = os.listdir(subfolder)
            for file in files:
                if file.endswith(".jpg"):
                    img_pathes.append(os.path.join(subfolder, file))
                    labels.append(int(subfolder.split("/")[-1]))

    test_csv = os.path.join(args.data_root, args.dataset, "solution.csv")
    test_im_pathes = []
    test_labels = []
    with open(test_csv, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            file, lbl = line.strip().split(",")
            file = file + ".jpg"
            test_labels.append(int(lbl))
            test_im_pathes.append(f"{test_root}/{file}")

elif args.dataset == "imagenet":
    mapping_txt = os.path.join(args.data_root, args.dataset, "LOC_synset_mapping.txt")
    name_label_mapping = {}
    with open(mapping_txt, "r") as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            sub_name = line.strip().split(" ")[0]
            name_label_mapping[sub_name] = ind

    img_files = os.listdir(root_path)
    labels = []
    img_pathes = []
    for img_file in img_files:
        if not img_file.endswith(".JPEG"):
            continue
        label = name_label_mapping[img_file.split("_")[0]]
        img_pathes.append(os.path.join(root_path, img_file))
        labels.append(label)

    test_labels = []
    test_im_pathes = []
    test_txt = os.path.join(args.data_root, args.dataset, "val_labels.txt")
    with open(test_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            file_name, lbl = line.strip().split(" ")
            test_im_pathes.append(f"{test_root}/{file_name}")
            test_labels.append(int(lbl))


labels = np.array(labels)
num_classes = labels.max() + 1
os.makedirs(os.path.join(args.meta_root, args.dataset), exist_ok=True)
for num_shot in NUM_SHOTS:
    fewshot_txt = os.path.join(
        args.meta_root, args.dataset, f"fewshot{num_shot}_seed{args.seed}.txt"
    )
    with open(fewshot_txt, "w") as f:
        for cls in range(num_classes):
            cls_indices = np.where(labels == cls)[0]
            np.random.shuffle(cls_indices)
            selected_indices = cls_indices[:num_shot]
            for idx in selected_indices:
                f.write(f"{img_pathes[idx]} {labels[idx]} \n")

test_txt = os.path.join(args.meta_root, args.dataset, f"test.txt")
with open(test_txt, "w") as f:
    for im_path, label in zip(test_im_pathes, test_labels):
        f.write(f"{im_path} {label}\n")
