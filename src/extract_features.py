"""Sample frames from each video and encode them with a frozen CLIP image encoder.

Data: the 10-class UCF101 subset published on the Hugging Face Hub (sayakpaul/ucf101-subset),
which keeps the original train/val/test folders. For every clip we sample T frames uniformly
in time and store a [T, D] array of L2-normalised CLIP embeddings plus the class label.
"""
import argparse
import re
import tarfile
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def fetch() -> Path:
    target = DATA / "UCF101_subset"
    if not target.exists():
        tar = hf_hub_download("sayakpaul/ucf101-subset", "UCF101_subset.tar.gz", repo_type="dataset", cache_dir=DATA / "hf")
        with tarfile.open(tar) as t:
            t.extractall(DATA)
    return target


def sample_frames(path: Path, t: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(n - 1, 0), t).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    while frames and len(frames) < t:  # pad very short clips by repeating the last frame
        frames.append(frames[-1])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = ap.parse_args()

    root = fetch()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    out = {}
    for split in ("train", "val", "test"):
        feats, labels, names = [], [], []
        videos = sorted((root / split).glob("*/*.avi"))
        for k, v in enumerate(videos):
            frames = sample_frames(v, args.frames)
            if not frames:
                continue
            with torch.no_grad():
                x = torch.stack([preprocess(f) for f in frames]).to(device)
                e = model.encode_image(x)
                e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            feats.append(e)
            labels.append(classes.index(v.parent.name))
            names.append(v.name)
            if k % 50 == 0:
                print(f"{split} {k}/{len(videos)}", flush=True)
        out[split] = (np.stack(feats), np.array(labels), names)
        print(split, out[split][0].shape)

    tokenizer = open_clip.get_tokenizer(args.model)
    words = lambda c: re.sub(r"(?<!^)(?=[A-Z])", " ", c).lower()  # "BasketballDunk" -> "basketball dunk"
    prompts = [f"a video of {words(c)}" for c in classes]
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(device))
        t = (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    np.savez_compressed(DATA / f"clip_{args.model}_T{args.frames}.npz", classes=np.array(classes), text=t,
                        **{f"{s}_x": out[s][0] for s in out}, **{f"{s}_y": out[s][1] for s in out},
                        **{f"{s}_names": np.array(out[s][2]) for s in out})
    print("saved")


if __name__ == "__main__":
    main()
