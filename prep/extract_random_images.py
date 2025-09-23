from __future__ import annotations
import argparse, random, shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png"}

TARGET_PER_CLASS: Dict[str, int] = {
    "Invasive_Tumor":         10_000,
    "DCIS_1":                 10_000,
    "DCIS_2":                 10_000,
    "Prolif_Invasive_Tumor":  10_000,
}


def plan_augments(n_orig:int, target:int)->List[int]:
    need = target - n_orig
    base, extra = divmod(need, n_orig)
    plan = [base]*n_orig
    for idx in random.sample(range(n_orig), extra):
        plan[idx] += 1
    return plan

def save_img(arr, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def process_class(cls_dir:Path, dst_root:Path, test_count:int):
    cls = cls_dir.name
    target = TARGET_PER_CLASS.get(cls, 10_000)

    files = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    if not files:
        print(f"⚠ No images in {cls}")
        return
    random.shuffle(files)

    test_n   = min(test_count, len(files)//5)
    test_set = files[:test_n]
    train_set_orig = files[test_n:] if target == -1 else files[test_n: min(len(files), target)]

    dst_train = dst_root/"train"/cls
    dst_test  = dst_root/"test"/cls
    dst_train.mkdir(parents=True, exist_ok=True); dst_test.mkdir(parents=True, exist_ok=True)

    for src in test_set:
        shutil.copy2(src, dst_test/src.name)
    for src in train_set_orig:
        shutil.copy2(src, dst_train/src.name)

    print(f"{cls:26s}  orig={len(files):5d}  "
          f"train={len(train_set_orig):5d}  test={len(test_set):4d}  target={target if target!=-1 else 'ALL'}")

    if target != -1 and len(train_set_orig) < target:
        aug_plan = plan_augments(len(train_set_orig), target)
        pipe = AUG_PIPE[cls]
        counts = Counter()
        for src, n_aug in tqdm(list(zip(train_set_orig, aug_plan)),
                               desc=f"Aug {cls}", ncols=80):
            img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
            for _ in range(n_aug):
                aug_img = pipe(image=img)["image"]
                fname = f"{src.stem}_aug{counts[src.stem]}.jpg"
                counts[src.stem] += 1
                save_img(aug_img, dst_train/fname)

def main(src:Path, dst:Path, test_count:int, seed:int):
    random.seed(seed)
    if dst.exists():
        print(f"{dst} already exists, files may be overwritten.")
    for cls_dir in sorted(src.iterdir()):
        if cls_dir.is_dir():
            process_class(cls_dir, dst, test_count)
    print("\nDataset ready:", dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/50",
                    help="Root folder with raw class sub-dirs")
    ap.add_argument("--dst", default="data/interim/original",
                    help="Destination root (creates train/ & test/)")
    ap.add_argument("--test_count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(Path(args.src), Path(args.dst), args.test_count, args.seed)
