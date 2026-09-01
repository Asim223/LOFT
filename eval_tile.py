"""
eval_tile.py -- decide whether LOFT's DPDD gap is overfitting or a train/test
resolution mismatch.

LOFT trains on 192x192 crops and is evaluated on whole 1680x1120 images. FBT
pools any input to a fixed 32x32 token grid, so a token covers ~6x6 pixels during
training and ~35x52 pixels at test time. The module is not computing the same
thing in the two regimes.

This script runs the SAME checkpoint over the SAME test images twice: once whole,
once in overlapping tiles the size of the training crops. Nothing is retrained.

  tiled ~= whole   -> the model genuinely fails to generalise; overfitting is real
                      and LFDOF pretraining is the right lever.
  tiled >> whole   -> the model is fine and full-resolution inference is breaking
                      it. Fix FBT; LFDOF would not have helped.

Usage (drop next to loft.py, or point PYTHONPATH at it):

  python eval_tile.py --tag v4 --ckpt-dir /kaggle/working/ckpt \
      --data /kaggle/working/dpdd --split test --tile 192 --overlap 48

  # sanity check on a handful of images first (~1 min)
  python eval_tile.py --tag v4 --ckpt-dir /kaggle/working/ckpt \
      --data /kaggle/working/dpdd --limit 5
"""

import argparse
import json
import os
import statistics as st

import torch
import torch.nn.functional as F

import loft  # importing is safe: loft.py guards its CLI with __name__ == "__main__"


def cosine_window(t, device, dtype):
    """Separable raised-cosine taper, so overlap-add leaves no visible seams."""
    w = torch.hann_window(t, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return (w[:, None] * w[None, :])[None, None]


@torch.no_grad()
def forward_tiled(model, y, tile=192, overlap=48):
    """
    Run the model on overlapping tile x tile patches and blend the results.

    Each tile is exactly the size the model was trained on, so every resolution-
    dependent component (FBT's pooling ratio, band_energy's spectrum) sees the
    same statistics it saw during training.
    """
    N, C, H, W = y.shape
    stride = tile - overlap
    assert stride > 0, "overlap must be smaller than tile"

    # pad up so an integer number of strides covers the image
    ph = (stride - (H - tile) % stride) % stride if H >= tile else tile - H
    pw = (stride - (W - tile) % stride) % stride if W >= tile else tile - W
    yp = F.pad(y, (0, pw, 0, ph), mode="reflect")
    Hp, Wp = yp.shape[-2:]

    win = cosine_window(tile, y.device, y.dtype)
    out = torch.zeros(N, C, Hp, Wp, device=y.device, dtype=y.dtype)
    nrm = torch.zeros(N, 1, Hp, Wp, device=y.device, dtype=y.dtype)

    ys = list(range(0, Hp - tile + 1, stride))
    xs = list(range(0, Wp - tile + 1, stride))
    for y0 in ys:
        for x0 in xs:
            patch = yp[:, :, y0:y0 + tile, x0:x0 + tile]
            pred = model(patch)["pred"]
            out[:, :, y0:y0 + tile, x0:x0 + tile] += pred * win
            nrm[:, :, y0:y0 + tile, x0:x0 + tile] += win
    return (out / nrm.clamp_min(1e-6))[..., :H, :W]


def find_ckpt(ckpt_dir, tag):
    for suffix in ("_best.pt", "_last.pt"):
        p = os.path.join(ckpt_dir, tag + suffix)
        if os.path.exists(p):
            return p
    raise SystemExit("no checkpoint for tag '%s' in %s" % (tag, ckpt_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--ckpt-dir", default="/kaggle/working/ckpt")
    ap.add_argument("--data", default="/kaggle/working/dpdd",
                    help="root holding <split>/blur and <split>/sharp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tile", type=int, default=192,
                    help="must match the training patch size")
    ap.add_argument("--overlap", type=int, default=48)
    ap.add_argument("--limit", type=int, help="only the first N images (quick check)")
    ap.add_argument("--out", default="eval_tile_results.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = loft.Cfg()
    model = loft.build_model(loft.asdict(cfg)).to(dev)
    ck = find_ckpt(a.ckpt_dir, a.tag)
    step, _ = loft.load_ckpt(ck, model, device=dev)
    model.eval()
    print("checkpoint %s (step %d) on %s" % (os.path.basename(ck), step, dev))

    bd = os.path.join(a.data, a.split, "blur")
    sd = os.path.join(a.data, a.split, "sharp")
    ds = loft.PairedFolder(bd, sd, train=False)
    n = len(ds) if a.limit is None else min(a.limit, len(ds))
    print("evaluating %d images, tile %d overlap %d" % (n, a.tile, a.overlap))

    p_in, p_whole, p_tiled, rows = [], [], [], []
    with torch.no_grad():
        for i in range(n):
            b, s = ds[i]
            b, s = b[None].to(dev), s[None].to(dev)
            pw_ = model(b)["pred"]
            pt_ = forward_tiled(model, b, a.tile, a.overlap)
            r = dict(i=i,
                     input=loft.psnr(b, s),
                     whole=loft.psnr(pw_, s),
                     tiled=loft.psnr(pt_, s))
            p_in.append(r["input"]); p_whole.append(r["whole"]); p_tiled.append(r["tiled"])
            rows.append(r)
            if (i + 1) % 10 == 0 or i + 1 == n:
                print("  %3d/%d  input %.3f  whole %.3f  tiled %.3f"
                      % (i + 1, n, st.mean(p_in), st.mean(p_whole), st.mean(p_tiled)),
                      flush=True)

    mi, mw, mt = st.mean(p_in), st.mean(p_whole), st.mean(p_tiled)
    print("\n%-22s %8.3f dB" % ("blurred input", mi))
    print("%-22s %8.3f dB  (%+.3f over input)" % ("whole-image", mw, mw - mi))
    print("%-22s %8.3f dB  (%+.3f over input)" % ("tiled @%d" % a.tile, mt, mt - mi))
    print("%-22s %8.3f dB" % ("tiled - whole", mt - mw))

    d = mt - mw
    print("\nreading:")
    if d > 0.5:
        print("  Tiling recovers %.2f dB. The model is not the problem; feeding it" % d)
        print("  full-resolution images is. FBT's fixed 32x32 pooling is the first")
        print("  place to look. LFDOF pretraining would not have addressed this.")
    elif d < 0.1:
        print("  Tiling changes nothing (%.2f dB). Resolution is not the issue," % d)
        print("  so the gap really is generalisation. LFDOF pretraining is the")
        print("  right lever; proceed with the two-stage plan.")
    else:
        print("  Ambiguous (%.2f dB). Some resolution sensitivity, but not enough to" % d)
        print("  explain the gap on its own. Worth doing both.")

    json.dump(dict(ckpt=ck, step=step, n=n, tile=a.tile, overlap=a.overlap,
                   input=mi, whole=mw, tiled=mt, per_image=rows),
              open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
