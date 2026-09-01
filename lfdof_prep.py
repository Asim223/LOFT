"""
lfdof_prep.py -- turn an extracted LFDOF tree into two flat, 1:1, sorted-aligned
folders that loft.py's PairedFolder can consume directly.

LFDOF (Ruan et al., IEEE TCI 2021) is many-to-one: each all-in-focus ground-truth
image is rendered into several defocused versions at different apertures/focal
depths. loft.py's PairedFolder matches blur[i] to sharp[i] by sorted filename
order and asserts equal counts, so feeding it the raw folders either crashes on
the count assert or -- worse -- silently pairs the wrong images. This script
resolves the mapping explicitly and emits one symlink per side with a shared
name, so sorted order is correct by construction.

Nothing is copied. 11 GB stays where it is; the output is a few MB of symlinks.

  python lfdof_prep.py --scan /kaggle/working/lfdof_raw
  python lfdof_prep.py --src /kaggle/working/lfdof_raw --out /kaggle/working/lfdof

Then:
  python loft.py --train --fast --tag lfdof \
      --train-blur /kaggle/working/lfdof/train/blur \
      --train-sharp /kaggle/working/lfdof/train/sharp \
      --val-blur  /kaggle/working/dpdd/val/blur \
      --val-sharp /kaggle/working/dpdd/val/sharp
"""

import argparse
import os
import re
import sys
from collections import defaultdict

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Folder-name vocabulary, most specific first. Each pair is (blur_side, sharp_side).
NAME_PAIRS = [
    ("input", "ground_truth"),
    ("input", "groundtruth"),
    ("input", "gt"),
    ("blur", "sharp"),
    ("blurred", "sharp"),
    ("source", "target"),
    ("defocus", "aif"),
    ("image", "gt"),
]


def images_in(d):
    """All images under d, recursively. LFDOF nests renders in per-scene folders."""
    out = []
    for dirpath, _, filenames in os.walk(d):
        for f in filenames:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def scan(root, max_depth=4):
    """Print every directory that holds images, with its count. Layout discovery."""
    root = os.path.abspath(root)
    base_depth = root.rstrip("/").count("/")
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count("/") - base_depth > max_depth:
            dirnames[:] = []
            continue
        n = sum(1 for f in filenames if f.lower().endswith(IMG_EXT))
        if n:
            found.append((os.path.relpath(dirpath, root), n,
                          sorted(f for f in filenames
                                 if f.lower().endswith(IMG_EXT))[:3]))
    if not found:
        print("no image folders under", root)
        return found
    w = max(len(p) for p, _, _ in found)
    for p, n, sample in sorted(found):
        print("  %-*s  %7d  e.g. %s" % (w, p, n, ", ".join(sample)))
    return found


def find_pairs(root):
    """
    Locate (blur_dir, sharp_dir, split_label) triples.

    Matches on directory *name* wherever it appears, and treats each side as a
    subtree rather than a flat folder, so nesting like input/<scene>/<render>.png
    is handled the same as a flat input/.
    """
    root = os.path.abspath(root)
    by_parent = defaultdict(dict)
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            by_parent[dirpath][d.lower()] = os.path.join(dirpath, d)

    out = []
    for parent, leaves in sorted(by_parent.items()):
        for bname, sname in NAME_PAIRS:
            if bname in leaves and sname in leaves:
                label = os.path.basename(parent)
                if label.lower() in ("", os.path.basename(root).lower()):
                    label = "train"
                out.append((leaves[bname], leaves[sname], label))
                break
    return out


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def match_keys(path):
    """
    Candidate ground-truth stems for a blurred image, best first.

    LFDOF names renders <scene>_ap<N>_<K>.png inside a folder called <scene>, so
    the containing folder is the most reliable key. Falls back to stripping
    trailing underscore-separated tokens off the filename.
    """
    s = stem(path)
    keys = [s, os.path.basename(os.path.dirname(path))]
    parts = re.split(r"[_-]", s)
    for k in range(1, min(4, len(parts))):
        keys.append("_".join(parts[:-k]))
        keys.append("-".join(parts[:-k]))
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def build(blur_dir, sharp_dir, out_blur, out_sharp, limit=None, per_scene=None,
          dry_run=False):
    blurs = images_in(blur_dir)
    sharps = {}
    for p in images_in(sharp_dir):
        sharps.setdefault(stem(p), p)
        sharps.setdefault(os.path.basename(os.path.dirname(p)), p)
    if not blurs or not sharps:
        raise SystemExit("empty: %s (%d) / %s (%d)"
                         % (blur_dir, len(blurs), sharp_dir, len(sharps)))

    matched, unmatched, rule_used = [], [], defaultdict(int)
    for b in blurs:
        for ri, key in enumerate(match_keys(b)):
            if key in sharps:
                matched.append((b, sharps[key], key))
                rule_used[ri] += 1
                break
        else:
            unmatched.append(b)

    if per_scene:
        keep, seen = [], defaultdict(int)
        for b, s, scene in matched:
            if seen[scene] < per_scene:
                keep.append((b, s, scene))
                seen[scene] += 1
        matched = keep
    if limit:
        matched = matched[:limit]

    print("  %s" % os.path.relpath(blur_dir))
    print("    blurred %d | ground-truth %d | matched %d | unmatched %d"
          % (len(blurs), len(sharps), len(matched), len(unmatched)))
    for ri, n in sorted(rule_used.items()):
        print("      rule %d matched %d" % (ri, n))
    if unmatched:
        print("      unmatched examples:",
              ", ".join(os.path.basename(u) for u in unmatched[:5]))
    if not matched:
        raise SystemExit("no pairs resolved -- inspect filenames and extend RELAX")
    if dry_run:
        for b, s, _ in matched[:5]:
            print("      %s  <-  %s" % (os.path.basename(b), os.path.basename(s)))
        return len(matched)

    os.makedirs(out_blur, exist_ok=True)
    os.makedirs(out_sharp, exist_ok=True)
    n = 0
    for i, (b, s, _) in enumerate(matched):
        name = "%06d_%s.png" % (i, stem(b))
        for src, dst in ((b, os.path.join(out_blur, name)),
                         (s, os.path.join(out_sharp, name))):
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        n += 1
    print("    wrote %d symlinked pairs -> %s" % (n, os.path.dirname(out_blur)))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="list every image folder under this root and exit")
    ap.add_argument("--src", help="extracted LFDOF root")
    ap.add_argument("--out", default="/kaggle/working/lfdof")
    ap.add_argument("--limit", type=int, help="cap total pairs per split")
    ap.add_argument("--per-scene", type=int,
                    help="keep at most this many renders per ground-truth scene "
                         "(2-3 cuts near-duplicate redundancy a lot)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the pairing without creating symlinks")
    a = ap.parse_args()

    if a.scan:
        scan(a.scan)
        return
    if not a.src:
        ap.error("pass --scan to inspect, or --src to build")

    pairs = find_pairs(a.src)
    if not pairs:
        print("no blur/sharp folder pair recognised. Layout:")
        scan(a.src)
        print("\nAdd the two folder names to NAME_PAIRS and re-run.")
        sys.exit(1)

    total = 0
    for blur_dir, sharp_dir, label in pairs:
        split = "train" if "train" in label.lower() else \
                "val" if "val" in label.lower() else \
                "test" if "test" in label.lower() else label.lower()
        total += build(blur_dir, sharp_dir,
                       os.path.join(a.out, split, "blur"),
                       os.path.join(a.out, split, "sharp"),
                       limit=a.limit, per_scene=a.per_scene, dry_run=a.dry_run)
    print("total pairs:", total)


if __name__ == "__main__":
    main()
