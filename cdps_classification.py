import torch
from DatasetWarp import WarpDataset
import os
import torch
from utils import parse_arguments
import numpy as np
from tqdm import tqdm


def load_caches(args, test=False):
    root = os.path.join(args.cache_root, args.model_cfg, args.dataset)
    if args.dataset == "imagenet":
        if test:
            file_name_template = f"{root}" + "/test_{}_batch{}.pt"
        else:
            file_name_template = (
                f"{root}"
                + "/train_{}"
                + f"_fewshot{args.fewshot}_seed{args.seed}"
                + "_batch{}.pt"
            )
    else:
        if test:
            file_name_template = f"{root}" + "/test_{}.pt"
        else:
            file_name_template = (
                f"{root}" + "/train_{}" + f"_fewshot{args.fewshot}_seed{args.seed}.pt"
            )

    if args.dataset == "imagenet":
        feats, global_feats, mask, logits, labels = [], [], [], [], []
        for i in range(1000):
            feats.append(file_name_template.format("feats", i))
            global_feats.append(file_name_template.format("global_feats", i))
            mask.append(file_name_template.format("mask", i))
            logits.append(torch.load(file_name_template.format("text_logits", i)))
            labels.append(torch.load(file_name_template.format("labels", i)))
        labels = torch.cat(labels, dim=0)
        logits = torch.cat(logits, dim=0)
    else:
        feats = torch.load(file_name_template.format("feats"))
        global_feats = torch.load(file_name_template.format("global_feats"))
        mask = torch.load(file_name_template.format("mask"))
        logits = torch.load(file_name_template.format("text_logits"))
        labels = torch.load(file_name_template.format("labels"))
    return feats, global_feats, mask, logits, labels


def mask_filter(feats, masks, load_files=False):
    feats_vec = []
    if load_files:
        for feat_file, mask_file in zip(feats, masks):
            load_feats = torch.load(feat_file)
            load_masks = torch.load(mask_file)
            for feat, mask in zip(load_feats, load_masks):
                mask_index = torch.where(mask.reshape(-1) == 1)[0]
                feat = feat[mask_index]
                feats_vec.append(feat)
    else:
        for feat, mask in zip(feats, masks):
            mask_index = torch.where(mask.reshape(-1) == 1)[0]
            feat = feat[mask_index]
            feats_vec.append(feat)
    return feats_vec


@torch.no_grad()
def construct_cdps_feats(
    feats,
    global_feats,
    num_classes,
    omega,
    num_confuse,
    labels,
    topk,
    load_files=False,
    class_indices=None,
):
    if class_indices is None:
        class_indices = [np.flatnonzero(labels == cls) for cls in range(num_classes)]

    def get_confuse_groups(global_feats, num_classes, labels):
        confuse_ids = []
        for cls in range(num_classes):
            cls_inds = class_indices[cls]
            cls_global_feats = global_feats[cls_inds]
            other_cls_inds = np.where(labels != cls)[0]
            other_labels = labels[other_cls_inds]
            other_global_feats = global_feats[other_cls_inds]
            similarity = cls_global_feats @ other_global_feats.t()

            sort_index = torch.argsort(similarity, dim=1, descending=True)
            # The original nested loop scans the sorted columns column-wise.
            # Flatten the same traversal once on CPU to avoid repeated scalar
            # device transfers and Python list membership checks on tensors.
            ranked_labels = other_labels[sort_index.T.reshape(-1)].detach().cpu().numpy()
            confuse_id = []
            seen_labels = set()
            for cur_label in ranked_labels:
                cur_label = int(cur_label)
                if cur_label not in seen_labels:
                    seen_labels.add(cur_label)
                    confuse_id.append(cur_label)
                    if len(confuse_id) >= num_confuse:
                        break
            confuse_id = np.array(confuse_id)
            confuse_ids.append(confuse_id)
        confuse_ids = np.stack(confuse_ids, axis=0)
        return confuse_ids

    if load_files:
        glb_feats = []
        for glb in global_feats:
            glb_feats.append(torch.load(glb))
        global_feats = torch.cat(glb_feats, dim=0)
    confuse_ids = get_confuse_groups(global_feats, num_classes, labels)
    score_list = []
    feats_list = []
    pbar = tqdm(range(num_classes), desc="Construct CDPS Feats of Each Class")
    for cls in pbar:
        cls_inds = class_indices[cls]
        concat_feats = []
        indexes = [0]
        for cls_ind in cls_inds:
            c_feat = feats[cls_ind]
            concat_feats.append(c_feat)
            indexes.append(indexes[-1] + c_feat.shape[0])
        concat_feats = torch.cat(concat_feats, dim=0)
        sim = concat_feats @ concat_feats.t()
        scores = torch.zeros(sim.shape[0]).to(sim.device)
        for i in range(len(indexes) - 1):
            cur_sim = sim[:, indexes[i] : indexes[i + 1]]
            scores += cur_sim.max(dim=-1).values
        scores = scores / len(cls_inds)

        other_concat_feats = []
        other_indexes = [0]
        for other_cls in confuse_ids[cls]:
            other_cls_ids = class_indices[other_cls]
            for other_cls_id in other_cls_ids:
                c_feat = feats[other_cls_id]
                other_concat_feats.append(c_feat)
                other_indexes.append(other_indexes[-1] + c_feat.shape[0])
        other_concat_feats = torch.cat(other_concat_feats, dim=0)
        inter_sim = concat_feats @ other_concat_feats.t()
        inter_scores = torch.zeros(inter_sim.shape[0]).to(inter_sim.device)
        for i in range(len(other_indexes) - 1):
            cur_sim = inter_sim[:, other_indexes[i] : other_indexes[i + 1]]
            inter_scores += cur_sim.max(dim=-1).values
        inter_scores = inter_scores / len(other_cls_ids)
        scores -= omega / (len(confuse_ids[cls]) - 1) * inter_scores

        represent_feats = []
        for i in range(len(indexes) - 1):
            cur_feats = concat_feats[indexes[i] : indexes[i + 1]]
            cur_score = scores[indexes[i] : indexes[i + 1]]
            pick_indexes = cur_score.argsort()[-topk:]
            represent_feats.append(cur_feats[pick_indexes])
        feats_list.append(torch.cat(represent_feats, dim=0))
    return torch.stack(feats_list, dim=0)


@torch.no_grad()
def compute_patch_logits(
    feats,
    cdps_feats,
    num_classes,
    device,
    topk=0,
    bsz=32,
    is_train=False,
    labels=None,
    fewshot=0,
    class_indices=None,
):
    assert (labels is not None and is_train and fewshot > 0) or not is_train
    logits = []
    if is_train and class_indices is None:
        class_indices = [np.flatnonzero(labels == cls) for cls in range(num_classes)]
    cdps_feats = cdps_feats.to(device)

    # These class blocks are identical for every image batch. Build them once
    # instead of repeatedly slicing/stacking features and reconstructing the
    # training-image lookup tensor inside the nested loop.
    class_blocks = []
    for cls in range(0, num_classes, bsz):
        cls_end = min(cls + bsz, num_classes)
        sample_locations = None
        if is_train:
            sample_locations = {}
            for row_id, class_id in enumerate(range(cls, cls_end)):
                for col_id, sample_id in enumerate(class_indices[class_id]):
                    sample_locations[int(sample_id)] = (row_id, col_id)
        class_blocks.append(
            (cdps_feats[cls:cls_end].unsqueeze(0), sample_locations)
        )

    pbar = tqdm(range(0, len(feats), bsz), desc="Compute Patch Logits")
    for i in pbar:
        concat_feats = []
        indexes = [0]
        for j in range(i, min(i + bsz, len(feats))):
            concat_feats.append(feats[j])
            indexes.append(indexes[-1] + feats[j].shape[0])
        concat_feats = torch.cat(concat_feats, dim=0)
        batch_logits = []
        concat_feats = concat_feats.unsqueeze(1).unsqueeze(2).to(device)
        for disc_feats, sample_locations in class_blocks:
            sim = (concat_feats * disc_feats).sum(dim=-1)
            sim_list = []
            if is_train:
                for c_iter, concat_id in enumerate(range(i, min(i + bsz, len(feats)))):
                    location = sample_locations.get(concat_id)
                    if location is None:
                        continue
                    row_id, col_id = location
                    sim[
                        indexes[c_iter] : indexes[c_iter + 1],
                        row_id,
                        col_id * topk : (col_id + 1) * topk,
                    ] = -1.0
            for index in range(len(indexes) - 1):
                cur_sim = sim[indexes[index] : indexes[index + 1]]
                cur_sim = cur_sim.max(dim=-1).values
                cal_topk = min(topk, cur_sim.shape[0])
                cur_sim = torch.topk(cur_sim, cal_topk, dim=0).values.mean(dim=0)
                sim_list.append(cur_sim.cpu())
            batch_logits.append(torch.stack(sim_list, dim=0))
        logits.append(torch.cat(batch_logits, dim=1))
    logits = torch.cat(logits, dim=0)
    return torch.nn.functional.softmax(logits, dim=-1)


def grid_search(patch_logits, text_logits, train_labels):
    best_lambda = 0
    best_acc = 0
    for lambda_val in np.arange(0.0, 1.1, 0.1):
        combined_logits = lambda_val * text_logits + (1 - lambda_val) * patch_logits
        preds = combined_logits.argmax(dim=-1)
        correct = (preds == train_labels).sum()
        acc = correct / len(train_labels)
        if acc > best_acc:
            best_acc = acc
            best_lambda = lambda_val
    return best_lambda


args = parse_arguments()
print("Loading precomputed features...")
train_feats, train_global_feats, train_mask, train_logits, train_labels = load_caches(
    args
)
masked_train_feats = mask_filter(
    train_feats, train_mask, load_files=args.dataset == "imagenet"
)
test_feats, test_global_feats, test_mask, test_logits, test_labels = load_caches(
    args, test=True
)
masked_test_feats = mask_filter(
    test_feats, test_mask, load_files=args.dataset == "imagenet"
)
if args.dataset == "imagenet":
    num_classes = 1000
else:
    num_classes = int(train_labels.max().item() + 1)

class_indices = [np.flatnonzero(train_labels.numpy() == cls) for cls in range(num_classes)]

print("Constructing CDPS features...")
cdps_feats = construct_cdps_feats(
    masked_train_feats,
    train_global_feats,
    num_classes,
    args.omega,
    args.num_confuse,
    train_labels,
    args.topk,
    load_files=args.dataset == "imagenet",
    class_indices=class_indices,
)

print("Computing patch logits...")
train_patch_logits = compute_patch_logits(
    masked_train_feats,
    cdps_feats,
    num_classes,
    device=args.device,
    is_train=True,
    bsz=args.batch_size,
    labels=train_labels,
    fewshot=args.fewshot,
    topk=args.topk,
    class_indices=class_indices,
)

test_patch_logits = compute_patch_logits(
    masked_test_feats,
    cdps_feats,
    num_classes,
    device=args.device,
    topk=args.topk,
    bsz=args.batch_size,
)

print("Finding best lambda...")
best_lambda = grid_search(train_patch_logits, train_logits, train_labels)

print("Computing final logits...")
prediction = best_lambda * test_logits + (1.0 - best_lambda) * test_patch_logits
correct = (prediction.argmax(dim=-1).cpu() == test_labels).sum()
acc = correct / len(test_labels)
print(f"test acc: {acc}")

with open(args.output_file, "a") as f:
    f.write(f"-------------{args.dataset}-------------\n")
    f.write(f"fewshot: {args.fewshot}, seed: {args.seed}, accuracy: {acc*100:.2f}\n \n")
