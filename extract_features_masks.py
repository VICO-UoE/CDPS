from DatasetWarp import WarpDataset
from utils import get_models, get_prompts
import os
import argparse
import torch
import json
from tqdm import tqdm


def get_logits(global_feats, prompts, device, loader_id):
    print(f"Get logits of {loader_id}")
    text_feat_list = []
    for key in prompts:
        text_feat_list.append(prompts[key]["mean"])
    text_features = torch.stack(text_feat_list, dim=0)
    text_features = text_features.to(device)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    global_feats = global_feats.to(device)
    logits = global_feats @ text_features.T
    logits = torch.nn.functional.softmax(logits, dim=-1)
    return logits


def get_mask(attn_weights, num_patches, tgt_area, top_percent, topk, loader_id):
    def bfs(labels):
        visited = torch.zeros_like(labels)
        record_area = []
        record_sequence = []
        for i in range(num_patches):
            for j in range(num_patches):
                if labels[i, j] == 1:
                    queue = [(i, j)]
                    visited[i, j] = 1
                    sequence = [(i, j)]
                    area = 1
                    while queue:
                        x, y = queue.pop(0)
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < num_patches
                                and 0 <= ny < num_patches
                                and labels[nx, ny] == 1
                                and visited[nx, ny] == 0
                            ):
                                visited[nx, ny] = 1
                                queue.append((nx, ny))
                                sequence.append((nx, ny))
                                area += 1
                    record_area.append(area)
                    record_sequence.append(sequence)
        mask = torch.zeros_like(labels)
        record_area = torch.tensor(record_area)
        num_area = 0
        while num_area < tgt_area or mask.sum() < args.topk:
            num_area += 1
            max_id = torch.argmax(record_area)
            sequence = record_sequence[max_id]
            for x, y in sequence:
                mask[x, y] = 1
            record_area[max_id] = -1
        return mask

    res = []
    pbar = tqdm(attn_weights, total=len(attn_weights), desc=f"Get mask of {loader_id}")
    for weight in pbar:
        labels = torch.zeros(num_patches * num_patches, dtype=torch.float32)
        idx = torch.argsort(weight.reshape(-1), descending=True)[
            : int(num_patches * num_patches * top_percent)
        ]
        labels[idx] = 1.0
        labels = labels.reshape(num_patches, num_patches)
        labels = bfs(labels)
        res.append(labels)
    res = torch.stack(res, dim=0)
    return res


def extract_features_masks(model, dataloader, device, loader_id):
    model.eval()
    features = []
    attn_weights = []
    global_feats = []
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Extract Feature of {loader_id}",
    )
    with torch.no_grad():
        for i, (images, labels, _) in pbar:
            images = images.to(device)
            global_feat = model.visual(images)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            global_feats.append(global_feat.detach().clone())
            feats = model.visual.transformer.feats_each_layer[-1].detach().clone()
            feats = feats[:, 1:, :]
            feats = model.visual.ln_post(feats)
            feats = feats @ model.visual.proj
            feats /= feats.norm(dim=-1, keepdim=True)
            attn_weight = model.visual.transformer.attn_each_layer[-1].detach().clone()
            features.append(feats)
            attn_weights.append(attn_weight)
            model.visual.transformer.feats_each_layer.clear()
            model.visual.transformer.attn_each_layer.clear()

    features = torch.cat(features, dim=0)
    attn_weights = torch.cat(attn_weights, dim=0)
    global_feats = torch.cat(global_feats, dim=0)
    return features, attn_weights, global_feats


parser = argparse.ArgumentParser(description="Extract features and masks")
parser.add_argument("--dataset", type=str, default="fgvc-aircraft")
parser.add_argument("--meta_root", type=str, default="meta_data")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--fewshot", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_cfg", type=str, default="vitb16_openclip_laion2b")
parser.add_argument("--cache_path", type=str, default="./caches")
parser.add_argument("--num_area", type=int, default=2)
parser.add_argument("--top_percent", type=float, default=0.4)
parser.add_argument("--force_test_compute", action="store_true")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument(
    "--prompt_name",
    type=str,
    default="alternates",
    choices=["most_common_name", "name"],
)
args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"
metric_file = os.path.join(args.meta_root, args.dataset, f"{args.dataset}_names.json")
with open(metric_file, "r") as f:
    args.metrics = json.load(f)


save_path = os.path.join(args.cache_path, args.model_cfg, args.dataset)
os.makedirs(save_path, exist_ok=True)
if not os.path.exists(
    os.path.join(save_path, "test_global_feats.pt") or args.force_test_compute
):
    args.force_test_compute = True
num_patches = 14 if "16" in args.model_cfg else 7

model, preprocess, tokenizer = get_models(args)

dataset = WarpDataset(
    args.meta_root,
    args.dataset,
    preprocess,
    split="train",
    fewshot=args.fewshot,
    seed=args.seed,
)
test_dataset = WarpDataset(
    args.meta_root,
    args.dataset,
    preprocess,
    split="test",
    fewshot=-1,
    seed=args.seed,
)
train_labels = torch.tensor(dataset.labels)
test_labels = torch.tensor(test_dataset.labels)
if args.dataset == "imagenet":
    num_classes = train_labels.max() + 1
    store_batch = args.fewshot
    sub_datasets = []
    sub_loaders = []
    for i in range(num_classes):
        indexes = torch.where(train_labels == i)[0]
        subset = torch.utils.data.Subset(dataset, indexes)
        sub_datasets.append(subset)
        sub_loaders.append(
            torch.utils.data.DataLoader(
                subset, batch_size=args.batch_size, shuffle=False
            )
        )
    sub_labels = [
        train_labels[i : i + store_batch]
        for i in range(0, len(train_labels), store_batch)
    ]
    sub_testsets = []
    sub_testloaders = []
    sub_testlabels = []
    for i in range(num_classes):
        indexes = torch.where(test_labels == i)[0]
        subset = torch.utils.data.Subset(test_dataset, indexes)
        sub_testsets.append(subset)
        sub_testloaders.append(
            torch.utils.data.DataLoader(
                subset, batch_size=args.batch_size, shuffle=False
            )
        )
        sub_testlabels.append(test_labels[indexes])
else:
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_labels = torch.tensor(test_dataset.labels)

# first get the textual features
prompts = get_prompts(args, tokenizer, model)

# extract the patch features and attention weight
if args.dataset == "imagenet":
    for i, sub_loader in enumerate(sub_loaders):
        train_feats, train_attn_weights, train_global_feats = extract_features_masks(
            model, sub_loader, args.device, loader_id=f"train_{i}"
        )
        train_mask = get_mask(
            train_attn_weights,
            num_patches,
            args.num_area,
            args.top_percent,
            args.topk,
            loader_id=f"train_{i}",
        )
        train_logits = get_logits(
            train_global_feats, prompts, args.device, loader_id=f"train_{i}"
        )
        print(f"Save train_{i} caches...")
        torch.save(
            train_feats.cpu(),
            os.path.join(
                save_path,
                f"train_feats_fewshot{args.fewshot}_seed{args.seed}_batch{i}.pt",
            ),
        )
        torch.save(
            train_global_feats.cpu(),
            os.path.join(
                save_path,
                f"train_global_feats_fewshot{args.fewshot}_seed{args.seed}_batch{i}.pt",
            ),
        )
        torch.save(
            train_mask,
            os.path.join(
                save_path,
                f"train_mask_fewshot{args.fewshot}_seed{args.seed}_batch{i}.pt",
            ),
        )
        torch.save(
            train_logits.cpu(),
            os.path.join(
                save_path,
                f"train_text_logits_fewshot{args.fewshot}_seed{args.seed}_batch{i}.pt",
            ),
        )
        torch.save(
            sub_labels[i],
            os.path.join(
                save_path,
                f"train_labels_fewshot{args.fewshot}_seed{args.seed}_batch{i}.pt",
            ),
        )
        if os.path.exists(os.path.join(save_path, f"test_feats_batch{i}.pt")):
            continue
        test_feats, test_attn_weights, test_global_feats = extract_features_masks(
            model, sub_testloaders[i], args.device, loader_id=f"test_{i}"
        )
        test_mask = get_mask(
            test_attn_weights,
            num_patches,
            args.num_area,
            args.top_percent,
            args.topk,
            loader_id=f"test_{i}",
        )
        test_logits = get_logits(
            test_global_feats, prompts, args.device, loader_id=f"test_{i}"
        )
        print(f"Save test_{i} caches...")
        torch.save(test_feats.cpu(), os.path.join(save_path, f"test_feats_batch{i}.pt"))
        torch.save(
            test_global_feats.cpu(),
            os.path.join(save_path, f"test_global_feats_batch{i}.pt"),
        )
        torch.save(test_mask, os.path.join(save_path, f"test_mask_batch{i}.pt"))
        torch.save(
            test_logits.cpu(), os.path.join(save_path, f"test_text_logits_batch{i}.pt")
        )
        torch.save(
            sub_testlabels[i], os.path.join(save_path, f"test_labels_batch{i}.pt")
        )
else:
    train_feats, train_attn_weights, train_global_feats = extract_features_masks(
        model, dataloader, args.device, loader_id="train"
    )
    # get the mask
    train_mask = get_mask(
        train_attn_weights,
        num_patches,
        args.num_area,
        args.top_percent,
        args.topk,
        loader_id="train",
    )
    # get the logits
    train_logits = get_logits(
        train_global_feats, prompts, args.device, loader_id="train"
    )

    print("Save train caches...")
    torch.save(
        train_feats.cpu(),
        os.path.join(
            save_path, f"train_feats_fewshot{args.fewshot}_seed{args.seed}.pt"
        ),
    )
    torch.save(
        train_global_feats.cpu(),
        os.path.join(
            save_path, f"train_global_feats_fewshot{args.fewshot}_seed{args.seed}.pt"
        ),
    )
    torch.save(
        train_mask,
        os.path.join(save_path, f"train_mask_fewshot{args.fewshot}_seed{args.seed}.pt"),
    )
    torch.save(
        train_logits.cpu(),
        os.path.join(
            save_path, f"train_text_logits_fewshot{args.fewshot}_seed{args.seed}.pt"
        ),
    )
    torch.save(
        train_labels,
        os.path.join(
            save_path, f"train_labels_fewshot{args.fewshot}_seed{args.seed}.pt"
        ),
    )

    if not os.path.exists(os.path.join(save_path, "test_feats.pt")):
        test_feats, test_attn_weights, test_global_feats = extract_features_masks(
            model, test_loader, args.device, loader_id="test"
        )
        test_mask = get_mask(
            test_attn_weights,
            num_patches,
            args.num_area,
            args.top_percent,
            args.topk,
            loader_id="test",
        )
        test_logits = get_logits(
            test_global_feats, prompts, args.device, loader_id="test"
        )

        print("Save test caches...")
        torch.save(test_feats.cpu(), os.path.join(save_path, "test_feats.pt"))
        torch.save(
            test_global_feats.cpu(), os.path.join(save_path, "test_global_feats.pt")
        )
        torch.save(test_mask, os.path.join(save_path, "test_mask.pt"))
        torch.save(test_logits.cpu(), os.path.join(save_path, "test_text_logits.pt"))
        torch.save(test_labels, os.path.join(save_path, "test_labels.pt"))
