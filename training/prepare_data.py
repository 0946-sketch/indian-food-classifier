"""Clean raw data and split 70/15/15 with no leakage.
Expects data/raw/<class_name>/*.jpg  ->  writes data/split/{train,val,test}/<class_name>/
"""
import hashlib, json, random, shutil
from pathlib import Path
from PIL import Image

RAW, OUT = Path("data/raw"), Path("data/split")
MIN_IMAGES, MAX_IMAGES, SEED = 150, 300, 42
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    seen, report = set(), {"classes": {}, "skipped_classes": {}}
    for cls_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        good, corrupt, dupes = [], 0, 0
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() not in EXT:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    if min(im.size) < 64:
                        corrupt += 1
                        continue
            except Exception:
                corrupt += 1
                continue
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in seen:  # global set -> also catches duplicates across classes
                dupes += 1
                continue
            seen.add(h)
            good.append(f)
        if len(good) < MIN_IMAGES:
            report["skipped_classes"][cls_dir.name] = len(good)
            continue
        random.shuffle(good)
        good = good[:MAX_IMAGES]  # cap for faster training and class balance
        n = len(good)
        a, b = int(0.70 * n), int(0.85 * n)
        splits = {"train": good[:a], "val": good[a:b], "test": good[b:]}
        for name, files in splits.items():
            d = OUT / name / cls_dir.name
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files):
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    im.thumbnail((512, 512))
                    im.save(d / f"{i:04d}.jpg", quality=90)
        report["classes"][cls_dir.name] = {
            "kept": n, "corrupt_removed": corrupt, "duplicates_removed": dupes,
            **{k: len(v) for k, v in splits.items()},
        }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/data_report.json").write_text(json.dumps(report, indent=2))
    print(f"Kept {len(report['classes'])} classes. Skipped (<{MIN_IMAGES} imgs): {report['skipped_classes']}")
    for c, r in report["classes"].items():
        print(f"{c:30s} train={r['train']:4d} val={r['val']:4d} test={r['test']:4d}")


if __name__ == "__main__":
    main()
