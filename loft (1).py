"""
================================================================================
 LOFT - Single Image Defocus Deblurring via a Learned Optical Transfer Function
 SINGLE FILE: model + training + evaluation.  No other files needed.
================================================================================

 QUICK START (Colab / Kaggle)
 ----------------------------
   !python loft.py --selftest          # verify the model builds and trains (1 min)
   !python loft.py --fetch-dpdd        # download DPDD (~8 min, once)
   !python loft.py --train --fast --tta --tag v3
   !python loft.py --eval --tag v3     # PSNR/SSIM on the test split

 Training RESUMES automatically. Re-run the same --train command after any
 disconnect, or to continue with a larger --steps, and it picks up from the last
 checkpoint. Ten short sessions are equivalent to one long one.

 THE IDEA
 --------
 Kernel-based defocus deblurring assumes an isotropic Gaussian PSF. Measuring real
 OTFs from aligned DPDD pairs shows the radial MAGNITUDE is in fact well fit by a
 Gaussian, but the PHASE is 4.6x larger than a matched Gaussian control (p ~ 1e-53).
 A real-valued symmetric kernel has zero OTF phase by construction, so no mixture of
 such kernels can represent real defocus. LOFT therefore estimates a COMPLEX OTF and
 inverts it, per overlapping window, in the frequency domain.

 CONTENTS
   [1] helpers, NAF blocks        [6] FBT (band-biased attention)
   [2] OTF basis (complex)        [7] LOFT model + x8 self-ensemble
   [3] LOE (coefficient maps)     [8] config, losses, data, metrics
   [4] LWI (learned Wiener)       [9] train / eval / dataset setup
   [5] windowed inversion         [10] CLI
================================================================================
"""

__version__ = "1.4.0"
__changelog__ = """
1.4.0  --no-otf control: run the refinement network alone, to test whether the OTF
       inversion is contributing or costing accuracy
1.3.0  LFDOF pretraining (--fetch-lfdof) and --init for fine-tuning from a checkpoint
1.2.0  merged model+training into one file; added --selftest and --eval
1.1.0  NAF blocks, base/blocks configurable, x8 self-ensemble (--tta), w_perc default 0,
       LR warmup, --fast preset for short sessions
1.0.0  windowed spatially-varying OTF inversion (replaced the global-OTF simplification),
       EMA weights, orthonormal FFT loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, k=3, s=1):
    return nn.Conv2d(cin, cout, k, s, k // 2)


class CA(nn.Module):
    """Channel attention: cheap, and consistently worth a little accuracy."""

    def __init__(self, c, r=8):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv(c, max(4, c // r), 1),
                               nn.ReLU(inplace=True), conv(max(4, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class ResBlock(nn.Module):
    def __init__(self, c, ca=False):
        super().__init__()
        self.b = nn.Sequential(conv(c, c), nn.ReLU(inplace=True), conv(c, c))
        self.ca = CA(c) if ca else nn.Identity()

    def forward(self, x):
        return x + self.ca(self.b(x))


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.w[None, :, None, None] + self.b[None, :, None, None]


class SimpleGate(nn.Module):
    """Multiplicative gate: replaces an activation with a cheap nonlinearity."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free block (Chen et al., ECCV 2022 design).

    Depthwise conv + SimpleGate + simplified channel attention, then a gated FFN.
    Consistently outperforms plain residual blocks per parameter on deblurring,
    which is why the refinement stage uses these instead of conv-relu-conv.
    """

    def __init__(self, c, dw=2, ffn=2):
        super().__init__()
        d = c * dw
        self.n1 = LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, d, 1)
        self.c2 = nn.Conv2d(d, d, 3, 1, 1, groups=d)          # depthwise
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.c3 = nn.Conv2d(d // 2, c, 1)
        self.n2 = LayerNorm2d(c)
        self.c4 = nn.Conv2d(c, c * ffn, 1)
        self.c5 = nn.Conv2d(c * ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.c2(self.c1(self.n1(x))))
        y = y * self.sca(y)
        x = x + self.c3(y) * self.beta
        y = self.sg(self.c4(self.n2(x)))
        return x + self.c5(y) * self.gamma


def radial_grid(h, w, device, dtype):
    """Radial frequency on the rfft2 half-plane -> (h, w//2+1)."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).view(1, -1)
    return torch.sqrt(fy ** 2 + fx ** 2)


def hann2d(n, device, dtype):
    """2-D Hann window: tapers each block so overlap-add leaves no seams."""
    w = torch.hann_window(n, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return (w[:, None] * w[None, :])


# ============================================================= [2] OTF BASIS
class OTFBasis(nn.Module):
    """
    K complex OTFs psi_k, Gaussian-initialised (zero phase) and free to learn away
    from that in BOTH magnitude and phase. Phase is what a real symmetric kernel
    cannot represent, and what the empirical OTF analysis identifies as the gap.
    """

    def __init__(self, K=5, res=64, sigmas=None, complex_basis=True):
        super().__init__()
        if sigmas is None:
            sigmas = tuple(float(s) for s in torch.logspace(math.log10(0.6), math.log10(9.5), K))
        assert len(sigmas) == K, "sigmas has %d entries but K=%d" % (len(sigmas), K)
        self.K, self.complex_basis = K, complex_basis
        fy = torch.fft.fftfreq(res).view(-1, 1)
        fx = torch.fft.rfftfreq(res).view(1, -1)
        rho = torch.sqrt(fy ** 2 + fx ** 2)
        mag = torch.stack([torch.exp(-2 * math.pi ** 2 * (s ** 2) * rho ** 2) for s in sigmas])
        self.log_mag = nn.Parameter(torch.log(mag.clamp_min(1e-6)))
        self.phase = nn.Parameter(torch.zeros(K, *rho.shape), requires_grad=complex_basis)
        self._cache = {}

    def forward(self, h, w):
        """(K, h, w//2+1) complex, resampled to the requested block size."""
        key = (h, w, self.log_mag.device, self.training)
        tgt = (h, w // 2 + 1)
        mag = torch.exp(self.log_mag).unsqueeze(1)
        mag = F.interpolate(mag, size=tgt, mode="bilinear", align_corners=False).squeeze(1)
        mag = mag.clamp(0.0, 1.0)
        if not self.complex_basis:
            return torch.complex(mag, torch.zeros_like(mag))
        ph = F.interpolate(self.phase.unsqueeze(1), size=tgt,
                           mode="bilinear", align_corners=False).squeeze(1)
        return torch.polar(mag, ph)


# ==================================================================== [3] LOE
class LOE(nn.Module):
    """Predicts per-pixel coefficients alpha_k (softmax over k), ConvGRU across scales."""

    def __init__(self, K=5, base=48, in_ch=3):
        super().__init__()
        self.K = K
        self.enc1 = nn.Sequential(conv(in_ch, base), nn.ReLU(inplace=True), ResBlock(base, True))
        self.enc2 = nn.Sequential(conv(base, base * 2, s=2), nn.ReLU(inplace=True),
                                  ResBlock(base * 2, True))
        self.enc3 = nn.Sequential(conv(base * 2, base * 4, s=2), nn.ReLU(inplace=True),
                                  ResBlock(base * 4, True))
        self.gru_z = conv(base * 8, base * 4)
        self.gru_r = conv(base * 8, base * 4)
        self.gru_h = conv(base * 8, base * 4)
        self.dec2 = nn.Sequential(conv(base * 4, base * 2), nn.ReLU(inplace=True),
                                  ResBlock(base * 2))
        self.dec1 = nn.Sequential(conv(base * 3, base), nn.ReLU(inplace=True), ResBlock(base))
        self.head = conv(base * 2, K)

    def _gru(self, x, state):
        if state is None:
            state = torch.zeros_like(x)
        elif state.shape[-2:] != x.shape[-2:]:
            state = F.interpolate(state, size=x.shape[-2:], mode="bilinear", align_corners=False)
        xs = torch.cat([x, state], 1)
        z = torch.sigmoid(self.gru_z(xs))
        r = torch.sigmoid(self.gru_r(xs))
        n = torch.tanh(self.gru_h(torch.cat([x, r * state], 1)))
        return (1 - z) * state + z * n

    def forward(self, y, state=None):
        e1 = self.enc1(y)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        state = self._gru(e3, state)
        d2 = self.dec2(F.interpolate(state, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2u = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2u, e1], 1))
        return torch.softmax(self.head(torch.cat([d1, e1], 1)), dim=1), state


# ==================================================================== [4] LWI
class LWI(nn.Module):
    """
    Wiener inversion with a learned per-frequency regulariser:

        Xhat = conj(H) Y / ( |H|^2 + lambda(u,v) )

    lambda is predicted from |H|, the input spectrum and rho, so damping rises
    exactly where the OTF is weak and plain division would amplify noise.
    """

    def __init__(self, hidden=32, lam_min=1e-4, fixed_lambda=None):
        super().__init__()
        self.lam_min, self.fixed_lambda = lam_min, fixed_lambda
        self.net = nn.Sequential(conv(3, hidden, 1), nn.ReLU(inplace=True),
                                 conv(hidden, hidden, 1), nn.ReLU(inplace=True),
                                 conv(hidden, 1, 1))

    def forward(self, Y, H, rho):
        magH = H.abs()
        if self.fixed_lambda is not None:
            lam = torch.full_like(magH, float(self.fixed_lambda))
        else:
            magY = Y.abs().mean(1, keepdim=True)
            feat = torch.cat([magH, torch.log1p(magY), rho.expand_as(magH)], 1)
            lam = F.softplus(self.net(feat.float())).to(magH.dtype) + self.lam_min
        return (H.conj() * Y) / (magH ** 2 + lam), lam


# ==================================================== [5] WINDOWED INVERSION
def windowed_inverse(y, alpha, basis, lwi, win=64, chunk=768):
    """
    Spatially-varying frequency-domain inversion.

    Each overlapping win x win block gets its own OTF, built from alpha pooled over
    that block, and is inverted independently; blocks are recombined by Hann-weighted
    overlap-add. This is the core of v2: the inverse filter now changes across the
    image, which is what spatially-varying defocus requires.

    y     : (N,C,H,W)      alpha : (N,K,H,W)
    returns restored (N,C,H,W) and the mean lambda (for logging)
    """
    N, C, H, W = y.shape
    bs = min(win, H, W)
    bs = max(8, bs - (bs % 2))                     # even block size
    st = bs // 2                                   # 50% overlap

    ph = (st - (H - bs) % st) % st if H >= bs else bs - H
    pw = (st - (W - bs) % st) % st if W >= bs else bs - W
    yp = F.pad(y, (0, pw, 0, ph), mode="reflect")
    ap = F.pad(alpha, (0, pw, 0, ph), mode="reflect")
    Hp, Wp = yp.shape[-2:]

    blocks = F.unfold(yp, kernel_size=bs, stride=st)            # (N, C*bs*bs, L)
    L = blocks.shape[-1]
    blocks = blocks.view(N, C, bs, bs, L).permute(0, 4, 1, 2, 3).reshape(N * L, C, bs, bs)

    a = F.avg_pool2d(ap, kernel_size=bs, stride=st)             # (N,K,gh,gw)
    a = a.flatten(2).transpose(1, 2).reshape(N * L, -1)         # (N*L,K)
    a = a / a.sum(1, keepdim=True).clamp_min(1e-8)

    psi = basis(bs, bs)                                         # (K,bs,bsf) complex
    wnd = hann2d(bs, y.device, y.dtype)
    rho = radial_grid(bs, bs, y.device, y.dtype)[None, None]

    outs, lam_sum, nchunk = [], 0.0, 0
    for i in range(0, N * L, chunk):                            # chunk to bound memory
        b = blocks[i:i + chunk]
        ab = a[i:i + chunk]
        Hb = (ab.to(psi.dtype)[:, :, None, None] * psi[None]).sum(1, keepdim=True)
        Yb = torch.fft.rfft2((b * wnd).float())
        Xb, lam = lwi(Yb, Hb.to(Yb.dtype), rho)
        xb = torch.fft.irfft2(Xb, s=(bs, bs)).to(y.dtype) * wnd
        outs.append(xb)
        lam_sum += float(lam.detach().mean()); nchunk += 1
    xb = torch.cat(outs, 0)

    xb = xb.reshape(N, L, C, bs, bs).permute(0, 2, 3, 4, 1).reshape(N, C * bs * bs, L)
    out = F.fold(xb, output_size=(Hp, Wp), kernel_size=bs, stride=st)
    w2 = (wnd * wnd).view(1, 1, bs, bs).repeat(N, C, 1, 1).reshape(N, C * bs * bs, 1).expand(-1, -1, L)
    nrm = F.fold(w2, output_size=(Hp, Wp), kernel_size=bs, stride=st).clamp_min(1e-6)
    out = (out / nrm)[..., :H, :W]
    return out, (lam_sum / max(nchunk, 1))


# ==================================================================== [6] FBT
class FBT(nn.Module):
    """Global context with attention biased by radial-frequency-band energy."""

    def __init__(self, ch, heads=4, bands=8, tokens=32):
        super().__init__()
        self.h, self.bands, self.tokens = heads, bands, tokens
        self.qkv = conv(ch, ch * 3, 1)
        self.proj = conv(ch, ch, 1)
        self.band_mlp = nn.Sequential(nn.Linear(bands, 32), nn.ReLU(inplace=True),
                                      nn.Linear(32, heads))
        self.norm = nn.GroupNorm(1, ch)

    def band_energy(self, x):
        N, C, Hh, Ww = x.shape
        X = torch.fft.rfft2(x.mean(1, keepdim=True).float())
        rho = radial_grid(Hh, Ww, x.device, torch.float32)
        edges = torch.linspace(0, 0.5, self.bands + 1, device=x.device)
        p = X.abs() ** 2
        e = []
        for b in range(self.bands):
            m = ((rho >= edges[b]) & (rho < edges[b + 1])).to(p.dtype)
            e.append((p * m).sum((-2, -1)) / m.sum().clamp_min(1))
        return torch.log1p(torch.stack(e, -1)).to(x.dtype)

    def forward(self, x):
        N, C, Hh, Ww = x.shape
        t = min(self.tokens, Hh, Ww)
        xs = F.adaptive_avg_pool2d(self.norm(x), t)
        q, k, v = self.qkv(xs).chunk(3, dim=1)
        L = t * t
        q, k, v = [z.reshape(N, self.h, C // self.h, L).transpose(-2, -1) for z in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
        att = att + self.band_mlp(self.band_energy(x)).permute(0, 2, 1).unsqueeze(-1)
        out = (att.softmax(-1) @ v).transpose(-2, -1).reshape(N, C, t, t)
        out = F.interpolate(out, size=(Hh, Ww), mode="bilinear", align_corners=False)
        return x + self.proj(out)


# =================================================================== [7] LOFT
class LOFT(nn.Module):
    """
    Coarse-to-fine. Per scale:
        LOE -> alpha_k
        windowed Wiener inversion using a per-block OTF
        refinement (residual blocks + channel attention + band-biased attention)
        residual output on the input
    """

    def __init__(self, K=5, base=48, scales=3, complex_basis=True, use_fbt=True,
                 learn_basis=True, fixed_lambda=None, win=64, global_otf=False,
                 cfg_blocks=8, no_otf=False):
        super().__init__()
        self.scales, self.K, self.use_fbt = scales, K, use_fbt
        self.win, self.global_otf, self.no_otf = win, global_otf, no_otf
        self.basis = OTFBasis(K=K, complex_basis=complex_basis)
        if not learn_basis:
            for p in self.basis.parameters():
                p.requires_grad_(False)
        self.loe = LOE(K=K, base=base)
        self.lwi = LWI(fixed_lambda=fixed_lambda)
        self.head_in = conv(9, base)                      # [inverse | previous | input]
        nb = cfg_blocks
        self.refine = nn.Sequential(*[NAFBlock(base) for _ in range(nb)])
        self.fbt = FBT(base) if use_fbt else nn.Identity()
        self.tail = nn.Sequential(NAFBlock(base), NAFBlock(base), conv(base, 3))

    def _global_inverse(self, y, alpha):
        N, C, h, w = y.shape
        psi = self.basis(h, w)
        a = alpha.mean((-2, -1))
        Hf = (a.to(psi.dtype)[:, :, None, None] * psi[None]).sum(1, keepdim=True)
        Y = torch.fft.rfft2(y.float())
        rho = radial_grid(h, w, y.device, torch.float32)[None, None]
        X, lam = self.lwi(Y, Hf.to(Y.dtype), rho)
        return torch.fft.irfft2(X, s=(h, w)).to(y.dtype), float(lam.detach().mean())

    def step(self, y, prev, state):
        alpha, state = self.loe(y, state)
        if self.no_otf:
            # Control: skip inversion, feed the input in its place. Everything else
            # (LOE, refinement, FBT, losses) is unchanged, so any difference in
            # accuracy is attributable to the OTF inversion itself.
            x_inv, lam = y, 0.0
        elif self.global_otf:
            x_inv, lam = self._global_inverse(y, alpha)
        else:
            x_inv, lam = windowed_inverse(y, alpha, self.basis, self.lwi, win=self.win)
        x_inv = x_inv.clamp(-1.0, 2.0)

        if prev is None:
            prev = y
        f = self.refine(self.head_in(torch.cat([x_inv, prev, y], 1)))
        f = self.fbt(f) if self.use_fbt else f
        x = (y + self.tail(f)).clamp(0.0, 1.0)
        return x, alpha, x_inv, lam, state

    def forward(self, y):
        pyr = [y]
        for _ in range(self.scales - 1):
            pyr.append(F.avg_pool2d(pyr[-1], 2))
        pyr = pyr[::-1]
        out, state, prev = [], None, None
        alpha = lam = None
        for ys in pyr:
            if prev is not None:
                prev = F.interpolate(prev, size=ys.shape[-2:], mode="bilinear", align_corners=False)
            x, alpha, x_inv, lam, state = self.step(ys, prev, state)
            out.append(x)
            prev = x
        return dict(out=out, pred=out[-1], alpha=alpha, lam=lam)

    def reblur(self, x, alpha):
        """
        Apply the estimated (windowed) OTF to a restored image, for the
        reblur-consistency loss. Differentiable on purpose.
        """
        N, C, H, W = x.shape
        bs = min(self.win, H, W); bs = max(8, bs - (bs % 2)); st = bs // 2
        ph = (st - (H - bs) % st) % st if H >= bs else bs - H
        pw = (st - (W - bs) % st) % st if W >= bs else bs - W
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        ap = F.pad(alpha, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = xp.shape[-2:]
        blocks = F.unfold(xp, kernel_size=bs, stride=st)
        L = blocks.shape[-1]
        blocks = blocks.view(N, C, bs, bs, L).permute(0, 4, 1, 2, 3).reshape(N * L, C, bs, bs)
        a = F.avg_pool2d(ap, kernel_size=bs, stride=st).flatten(2).transpose(1, 2).reshape(N * L, -1)
        a = a / a.sum(1, keepdim=True).clamp_min(1e-8)
        psi = self.basis(bs, bs)
        wnd = hann2d(bs, x.device, x.dtype)
        Hb = (a.to(psi.dtype)[:, :, None, None] * psi[None]).sum(1, keepdim=True)
        Yb = torch.fft.rfft2((blocks * wnd).float()) * Hb.to(torch.complex64)
        yb = torch.fft.irfft2(Yb, s=(bs, bs)).to(x.dtype) * wnd
        yb = yb.reshape(N, L, C, bs, bs).permute(0, 2, 3, 4, 1).reshape(N, C * bs * bs, L)
        o = F.fold(yb, output_size=(Hp, Wp), kernel_size=bs, stride=st)
        w2 = (wnd * wnd).view(1, 1, bs, bs).repeat(N, C, 1, 1).reshape(N, C * bs * bs, 1).expand(-1, -1, L)
        nrm = F.fold(w2, output_size=(Hp, Wp), kernel_size=bs, stride=st).clamp_min(1e-6)
        return (o / nrm)[..., :H, :W]


@torch.no_grad()
def forward_x8(model, y):
    """
    x8 self-ensemble: average the prediction over the 8 flip/rotate symmetries.

    Standard practice in image restoration and typically worth 0.1-0.3 dB. It is a
    TEST-TIME technique, so the paper must state that results use self-ensembling
    (and the complexity table should note the 8x inference cost).
    """
    outs = []
    for k in range(4):
        z = torch.rot90(y, k, (-2, -1))
        for f in (False, True):
            zz = z.flip(-1) if f else z
            p = model(zz)["pred"]
            if f:
                p = p.flip(-1)
            outs.append(torch.rot90(p, -k, (-2, -1)))
    return torch.stack(outs, 0).mean(0)


# =========================================================== [8] BUILD_MODEL
def build_model(cfg=None):
    """
    Ablation flags:
        {}                        full LOFT (windowed)
        {"global_otf": True}      v1 behaviour: one OTF per image
        {"complex_basis": False}  real-valued basis (tests the phase claim)
        {"learn_basis": False}    fixed Gaussian basis
        {"fixed_lambda": 0.01}    no learned regularisation
        {"use_fbt": False}        no global context
    """
    cfg = cfg or {}
    return LOFT(K=cfg.get("K", 5), base=cfg.get("base", 48), scales=cfg.get("scales", 3),
                complex_basis=cfg.get("complex_basis", True),
                use_fbt=cfg.get("use_fbt", True),
                learn_basis=cfg.get("learn_basis", True),
                fixed_lambda=cfg.get("fixed_lambda", None),
                win=cfg.get("win", 64),
                global_otf=cfg.get("global_otf", False),
                cfg_blocks=cfg.get("blocks", 8),
                no_otf=cfg.get("no_otf", False))


# ============================================================== [9] SELF-TEST
def _check(name, cfg, shape=(2, 3, 128, 128)):
    m = build_model(cfg)
    y = torch.rand(*shape)
    o = m(y)
    assert o["pred"].shape == y.shape, "shape %s" % (o["pred"].shape,)
    assert torch.allclose(o["alpha"].sum(1), torch.ones_like(o["alpha"][:, 0]), atol=1e-4)
    o["pred"].mean().backward()
    tr = [p for p in m.parameters() if p.requires_grad]
    got = sum(p.grad is not None for p in tr)
    n = sum(p.numel() for p in m.parameters())
    print("  [%s] %-24s pred %-18s params %5.2fM  grads %d/%d"
          % ("OK " if got == len(tr) else "WARN", name, str(tuple(o["pred"].shape)),
             n / 1e6, got, len(tr)))
    return m


import argparse, glob, os, random, time, json
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader

# ================================================================== [1] CONFIG
@dataclass
class Cfg:
    # data
    train_blur: str = "/content/dpdd/train/blur"
    train_sharp: str = "/content/dpdd/train/sharp"
    val_blur: str = "/content/dpdd/val/blur"
    val_sharp: str = "/content/dpdd/val/sharp"
    patch: int = 192
    batch: int = 8                 # 4 fits a 16GB T4 at patch 256; drop to 2 if OOM
    workers: int = 2

    # model
    K: int = 5
    base: int = 48
    scales: int = 3
    complex_basis: bool = True
    use_fbt: bool = True
    learn_basis: bool = True
    blocks: int = 4
    win: int = 64
    global_otf: bool = False
    no_otf: bool = False
    init_from: str = None
    fixed_lambda: float = None

    # optimisation
    steps: int = 30_000           # total optimiser steps, not epochs (resume-friendly)
    lr: float = 3e-4
    lr_min: float = 1e-6
    wd: float = 1e-4
    clip: float = 1.0
    warmup: int = 500
    amp: bool = True               # mixed precision; big speedup on T4

    # loss weights
    w_char: float = 1.0
    w_freq: float = 0.1
    w_perc: float = 0.0    # 0 = PSNR-optimal; raise only for an LPIPS-focused variant
    w_otf: float = 0.1
    use_perc: bool = True          # set False to skip the VGG download

    # bookkeeping
    ckpt_dir: str = "/content/drive/MyDrive/loft_ckpt"
    tag: str = "dpdd"
    save_every: int = 1000
    val_every: int = 2000
    log_every: int = 50
    seed: int = 0


# ================================================================== [2] LOSSES
def charbonnier(x, y, eps=1e-3):
    """Smooth L1. Less sensitive than L2 to the small misalignments in DPDD pairs."""
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))


def freq_l1(x, y):
    """
    L1 between complex spectra: supervises magnitude and phase together.

    norm="ortho" matters. An unnormalised FFT has magnitudes that grow with image
    size, which made this term roughly 100x the pixel term and let it dominate the
    total. Orthonormal scaling satisfies Parseval, so spectral and pixel errors are
    on the same footing and the loss weights mean what they say.
    """
    X = torch.fft.rfft2(x, norm="ortho")
    Y = torch.fft.rfft2(y, norm="ortho")
    return torch.mean(torch.abs(X - Y))


class Perceptual(nn.Module):
    """VGG19 relu2_2 / relu3_4 features. Downloaded once, then cached by torchvision."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        v = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.s1, self.s2 = v[:9], v[9:26]
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x, y = (x - self.mu) / self.sd, (y - self.mu) / self.sd
        a1, b1 = self.s1(x), self.s1(y)
        a2, b2 = self.s2(a1), self.s2(b1)
        return F.l1_loss(a1, b1) + F.l1_loss(a2, b2)


class LOFTLoss(nn.Module):
    """
    Total loss. Multi-scale supervision weights each scale by 2^-(depth), so the
    finest scale dominates while coarse scales still receive signal.
    """

    def __init__(self, cfg, device):
        super().__init__()
        self.c = cfg
        self.perc = Perceptual().to(device) if (cfg.use_perc and cfg.w_perc > 0) else None

    def forward(self, model, out, y_blur, x_sharp):
        preds = out["out"]                       # coarse -> fine
        S = len(preds)
        L_char = L_freq = L_perc = 0.0
        for i, p in enumerate(preds):
            w = 2.0 ** (i - (S - 1))             # 0.25, 0.5, 1.0 for S=3
            tgt = x_sharp if p.shape[-2:] == x_sharp.shape[-2:] else \
                F.interpolate(x_sharp, size=p.shape[-2:], mode="bilinear", align_corners=False)
            L_char = L_char + w * charbonnier(p, tgt)
            L_freq = L_freq + w * freq_l1(p, tgt)
            if self.perc is not None and i == S - 1:      # perceptual on finest only (cost)
                L_perc = L_perc + w * self.perc(p.clamp(0, 1), tgt)

        # reblur consistency: estimated OTF applied to the prediction must give the input
        if getattr(self.c, "no_otf", False):
            L_otf = torch.zeros((), device=out["pred"].device)
        else:
            reblurred = model.reblur(out["pred"], out["alpha"])
            L_otf = charbonnier(reblurred, y_blur)

        total = (self.c.w_char * L_char + self.c.w_freq * L_freq +
                 self.c.w_perc * L_perc + self.c.w_otf * L_otf)

        def d(v):                    # detach before logging: these are numbers, not graph nodes
            return float(v.detach()) if torch.is_tensor(v) else float(v)
        return total, dict(char=d(L_char), freq=d(L_freq), perc=d(L_perc),
                           otf=d(L_otf), total=d(total))


# ==================================================================== [3] DATA
IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def listdir(d):
    fs = sorted(sum([glob.glob(os.path.join(d, e)) for e in IMG_EXT], []))
    return fs


class PairedFolder(Dataset):
    """
    Blurred/sharp pairs from two parallel folders, matched by sorted filename order.

    Training returns random crops with random flips. Validation returns whole images
    (batch size must be 1, since DPDD images share a size but other sets may not).
    Crops are taken, never resized: defocus blur is measured in pixels, so resizing
    would rescale the very quantity the model estimates.
    """

    def __init__(self, blur_dir, sharp_dir, patch=256, train=True):
        self.b, self.s = listdir(blur_dir), listdir(sharp_dir)
        assert self.b and self.s, "no images found in %s / %s" % (blur_dir, sharp_dir)
        assert len(self.b) == len(self.s), \
            "count mismatch: %d blurred vs %d sharp" % (len(self.b), len(self.s))
        self.patch, self.train = patch, train

    def __len__(self):
        return len(self.b)

    def _load(self, p):
        import cv2
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError("cannot read " + p)
        return torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, i):
        b, s = self._load(self.b[i]), self._load(self.s[i])
        h = min(b.shape[1], s.shape[1]); w = min(b.shape[2], s.shape[2])
        b, s = b[:, :h, :w], s[:, :h, :w]
        if self.train:
            p = self.patch
            if h < p or w < p:                        # pad small images by reflection
                ph, pw = max(0, p - h), max(0, p - w)
                b = F.pad(b[None], (0, pw, 0, ph), mode="reflect")[0]
                s = F.pad(s[None], (0, pw, 0, ph), mode="reflect")[0]
                h, w = b.shape[1], b.shape[2]
            y0, x0 = random.randint(0, h - p), random.randint(0, w - p)
            b, s = b[:, y0:y0 + p, x0:x0 + p], s[:, y0:y0 + p, x0:x0 + p]
            if random.random() < 0.5:
                b, s = b.flip(-1), s.flip(-1)
            if random.random() < 0.5:
                b, s = b.flip(-2), s.flip(-2)
        else:
            m = 8                                     # keep dims divisible for the pyramid
            b = b[:, :h // m * m, :w // m * m]
            s = s[:, :h // m * m, :w // m * m]
        return b, s


# ================================================================= [4] METRICS
def psnr(a, b):
    mse = torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(a, b):
    """Gaussian-window SSIM on [0,1] tensors, averaged over the batch and channels."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)],
                     device=a.device, dtype=a.dtype)
    k = (k / k.sum())
    win = (k[:, None] @ k[None, :]).expand(a.shape[1], 1, 11, 11).contiguous()

    def f(x):
        return F.conv2d(x, win, padding=5, groups=x.shape[1])
    mu_a, mu_b = f(a), f(b)
    sa = f(a * a) - mu_a ** 2
    sb = f(b * b) - mu_b ** 2
    sab = f(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return s.mean().item()


# ============================================================ [5] CHECKPOINTS
def save_ckpt(cfg, model, opt, sched, scaler, step, best, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"                    # write then rename: a drop mid-write cannot corrupt
    torch.save(dict(step=step, best=best, cfg=asdict(cfg),
                    model=model.state_dict(), opt=opt.state_dict(),
                    sched=sched.state_dict(),
                    scaler=scaler.state_dict() if scaler is not None else None), tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt=None, sched=None, scaler=None, device="cpu"):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    if sched is not None and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("best", -1e9)


# ============================================================== [6] TRAIN LOOP
TTA = False   # set by --tta


def validate(model, loader, device, limit=20):
    model.eval()
    ps, ss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (b, s) in enumerate(loader):
            if i >= limit:
                break
            b, s = b.to(device), s.to(device)
            p = forward_x8(model, b) if TTA else model(b)["pred"]
            ps += psnr(p, s); ss += ssim(p.clamp(0, 1), s); n += 1
    model.train()
    return (ps / max(n, 1), ss / max(n, 1), n)


def train(cfg):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| torch", torch.__version__)

    model = build_model(asdict(cfg)).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print("params: %.2fM" % (nparam / 1e6))

    tr = PairedFolder(cfg.train_blur, cfg.train_sharp, cfg.patch, True)
    dl = DataLoader(tr, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
                    pin_memory=True, drop_last=True, persistent_workers=cfg.workers > 0)
    have_val = os.path.isdir(cfg.val_blur) and listdir(cfg.val_blur)
    vl = DataLoader(PairedFolder(cfg.val_blur, cfg.val_sharp, train=False),
                    batch_size=1, shuffle=False, num_workers=1) if have_val else None
    print("train pairs:", len(tr), "| val:", len(vl.dataset) if vl else 0)

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}

    def ema_update(d=0.999):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema:
                    ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def with_ema(fn):
        """Evaluate/save using EMA weights, then restore the live ones."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in ema}
        model.load_state_dict({k: ema[k].to(v.dtype) for k, v in model.state_dict().items()
                               if k in ema}, strict=False)
        try:
            return fn()
        finally:
            model.load_state_dict(backup, strict=False)

    crit = LOFTLoss(cfg, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.wd)
    def lr_at(i):
        """Linear warmup then cosine decay. Warmup matters most on short schedules."""
        if i < cfg.warmup:
            return (i + 1) / max(1, cfg.warmup)
        t = (i - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return (cfg.lr_min / cfg.lr) + (1 - cfg.lr_min / cfg.lr) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    # --init: load weights from another run (e.g. an LFDOF pretrain) and start a
    # fresh schedule. Distinct from resume, which also restores step/optimiser state.
    if getattr(cfg, "init_from", None) and not os.path.exists(
            os.path.join(cfg.ckpt_dir, "%s_last.pt" % cfg.tag)):
        ck = torch.load(cfg.init_from, map_location=device)
        sd = ck["model"] if "model" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("initialised from %s (missing %d, unexpected %d)"
              % (cfg.init_from, len(missing), len(unexpected)))

    last = os.path.join(cfg.ckpt_dir, "%s_last.pt" % cfg.tag)
    best_p = os.path.join(cfg.ckpt_dir, "%s_best.pt" % cfg.tag)
    step, best = 0, -1e9
    if not os.path.isdir(os.path.dirname(cfg.ckpt_dir)) and cfg.ckpt_dir.startswith("/content/drive"):
        cfg.ckpt_dir = "/content/ckpt"
        print("Drive not mounted -> checkpointing to", cfg.ckpt_dir,
              "(download it before the session ends)")
        # --init: load weights from another run (e.g. an LFDOF pretrain) and start a
    # fresh schedule. Distinct from resume, which also restores step/optimiser state.
    if getattr(cfg, "init_from", None) and not os.path.exists(
            os.path.join(cfg.ckpt_dir, "%s_last.pt" % cfg.tag)):
        ck = torch.load(cfg.init_from, map_location=device)
        sd = ck["model"] if "model" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("initialised from %s (missing %d, unexpected %d)"
              % (cfg.init_from, len(missing), len(unexpected)))

    last = os.path.join(cfg.ckpt_dir, "%s_last.pt" % cfg.tag)
    best_p = os.path.join(cfg.ckpt_dir, "%s_best.pt" % cfg.tag)
    if os.path.exists(last):
        step, best = load_ckpt(last, model, opt, sched, scaler, device)
        print("resumed from step %d (best PSNR %.3f)" % (step, best))
    else:
        print("starting fresh")

    t0, it = time.time(), iter(dl)
    while step < cfg.steps:
        try:
            b, s = next(it)
        except StopIteration:
            it = iter(dl); b, s = next(it)
        b, s = b.to(device, non_blocking=True), s.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(cfg.amp and device == "cuda")):
            out = model(b)
            loss, parts = crit(model, out, b, s)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)   # OTF inversion can spike grads
        scaler.step(opt); scaler.update(); sched.step()
        ema_update()
        step += 1

        if step % cfg.log_every == 0:
            ips = cfg.log_every * cfg.batch / (time.time() - t0); t0 = time.time()
            print("step %6d | total %.4f char %.4f freq %.4f perc %.4f otf %.4f | lr %.2e | %.1f img/s"
                  % (step, parts["total"], parts["char"], parts["freq"], parts["perc"],
                     parts["otf"], sched.get_last_lr()[0], ips), flush=True)

        if vl is not None and step % cfg.val_every == 0:
            p, ssv, n = with_ema(lambda: validate(model, vl, device))
            print(">> val step %d: PSNR %.3f dB  SSIM %.4f  (n=%d)" % (step, p, ssv, n), flush=True)
            if p > best:
                best = p
                with_ema(lambda: save_ckpt(cfg, model, opt, sched, scaler, step, best, best_p))
                print(">> new best, saved", best_p, flush=True)

        if step % cfg.save_every == 0:
            save_ckpt(cfg, model, opt, sched, scaler, step, best, last)

    save_ckpt(cfg, model, opt, sched, scaler, step, best, last)
    print("done at step", step, "| best val PSNR %.3f" % best)


# =========================================================== [7] DATASET SETUP
def fetch_dpdd(dest="/content/dpdd", zip_to=None, resume=True):
    """
    Download DPDD into `dest` as train/val/test folders of blur+sharp PNGs.

    RESUMABLE: files that already exist are skipped, so if the session drops you can
    simply re-run this and it continues. Point `dest` at Drive to make the progress
    survive disconnects (slower per file, but you only pay for what is missing).
    """
    import numpy as np
    os.system("pip -q install datasets")
    from datasets import load_dataset
    import cv2

    got = {}
    for names, sub in ((("train",), "train"),
                       (("val", "validation"), "val"),
                       (("test",), "test")):
        bd, sd = os.path.join(dest, sub, "blur"), os.path.join(dest, sub, "sharp")
        os.makedirs(bd, exist_ok=True); os.makedirs(sd, exist_ok=True)
        have = len([f for f in os.listdir(bd) if f.endswith(".png")])

        ds, err = None, ""
        for nm in names:
            try:
                ds = load_dataset("JacobLinCool/DPDD", "combined", split=nm)
                break
            except Exception as e:
                err = str(e)[:100]
        if ds is None:
            print("  split %s unavailable: %s" % (names[0], err)); continue
        if resume and have >= len(ds):
            print("  %-5s already complete (%d pairs), skipping" % (sub, have)); got[sub] = have
            continue

        cols = list(ds.features.keys())
        kb = next((c for c in ("blur", "source", "blurred", "input") if c in cols), None)
        ks = next((c for c in ("sharp", "target", "gt", "allinfocus") if c in cols), None)
        if kb is None or ks is None:
            print("  cannot identify columns in %s: %s" % (sub, cols)); continue

        written = 0
        for i, r in enumerate(ds):
            pb, ps = os.path.join(bd, "%04d.png" % i), os.path.join(sd, "%04d.png" % i)
            if resume and os.path.exists(pb) and os.path.exists(ps):
                continue
            cv2.imwrite(pb, np.array(r[kb].convert("RGB"))[..., ::-1])
            cv2.imwrite(ps, np.array(r[ks].convert("RGB"))[..., ::-1])
            written += 1
            if written % 50 == 0:
                print("    %s: %d/%d written" % (sub, i + 1, len(ds)), flush=True)
        got[sub] = len(ds)
        print("  %-5s %d pairs  (columns: %s / %s)" % (sub, len(ds), kb, ks), flush=True)

    if not got:
        raise SystemExit("no splits downloaded; check the dataset name/network")
    if zip_to:
        os.makedirs(os.path.dirname(zip_to), exist_ok=True)
        print("archiving to", zip_to)
        os.system("cd %s && zip -qr %s ." % (dest, zip_to))
    print("dataset ready at", dest)
    return got


DRIVE_DATA = "/content/drive/MyDrive/loft_data/dpdd"


def setup_dpdd(drive_dir=None):
    """
    Write DPDD straight to Drive as folders (no zip). Resumable: re-run after any
    disconnect and it fills in only what is missing.
    """
    dest = drive_dir or DRIVE_DATA
    if not os.path.isdir("/content/drive/MyDrive"):
        raise SystemExit("Drive not mounted. Run:\n"
                         "  from google.colab import drive; drive.mount('/content/drive')")
    return fetch_dpdd(dest, zip_to=None, resume=True)



def fetch_lfdof(dest="/content/lfdof", limit=None):
    """
    LFDOF: 11,986 light-field-generated defocus pairs (Ruan et al., IEEE TCI 2021).

    34x larger than DPDD's training split, which is why it is used for PRETRAINING:
    a 3.7M-parameter model cannot reach competitive accuracy from 350 images alone.
    Resumable and skip-existing, like the DPDD fetch.

    If the HF mirrors below are unavailable, download from the authors' page
    (https://sweb.cityu.edu.hk/miullam/AIFNET/), unzip, and pass the folders
    directly with --train-blur / --train-sharp.
    """
    import numpy as np
    os.system("pip -q install datasets")
    from datasets import load_dataset
    import cv2

    ds = None
    for name in ("JacobLinCool/LFDOF", "danjacobellis/LFDOF", "lfdof"):
        for split in ("train", "all"):
            try:
                ds = load_dataset(name, split=split); print("using", name, split); break
            except Exception:
                continue
        if ds is not None:
            break
    if ds is None:
        raise SystemExit(
            "LFDOF not found on the HF mirrors tried.\n"
            "Download it from https://sweb.cityu.edu.hk/miullam/AIFNET/ , unzip, then run\n"
            "  --train --train-blur <dir>/blur --train-sharp <dir>/sharp")

    cols = list(ds.features.keys())
    kb = next((c for c in ("blur", "source", "blurred", "input") if c in cols), None)
    ks = next((c for c in ("sharp", "target", "gt", "allinfocus") if c in cols), None)
    if kb is None or ks is None:
        raise SystemExit("unexpected columns: %s" % cols)

    bd, sd = os.path.join(dest, "blur"), os.path.join(dest, "sharp")
    os.makedirs(bd, exist_ok=True); os.makedirs(sd, exist_ok=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        pb, ps = os.path.join(bd, "%05d.png" % i), os.path.join(sd, "%05d.png" % i)
        if os.path.exists(pb) and os.path.exists(ps):
            continue
        r = ds[i]
        cv2.imwrite(pb, np.array(r[kb].convert("RGB"))[..., ::-1])
        cv2.imwrite(ps, np.array(r[ks].convert("RGB"))[..., ::-1])
        if i % 500 == 0:
            print("  %d/%d" % (i, n), flush=True)
    print("LFDOF ready at %s (%d pairs)" % (dest, n))
    return n


def prepare(drive_dir=None, local="/content/dpdd"):
    """Each session: copy the Drive dataset to fast local disk (skips what's already there)."""
    src = drive_dir or DRIVE_DATA
    if not os.path.isdir(src):
        raise SystemExit("no dataset at %s -- run --setup-dpdd first" % src)
    os.makedirs(local, exist_ok=True)
    os.system("cp -ru %s/. %s/" % (src, local))
    for sub in ("train", "val", "test"):
        d = os.path.join(local, sub, "blur")
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print("  %-5s %d pairs" % (sub, n))


def selftest():
    """Build every configuration, check shapes/gradients, and run 3 real train steps."""
    torch.manual_seed(0)
    print("torch %s | cuda %s" % (torch.__version__, torch.cuda.is_available()))
    for name, cfg in [("full (windowed)", {}), ("global OTF", {"global_otf": True}),
                      ("real basis", {"complex_basis": False}), ("no FBT", {"use_fbt": False}),
                      ("fixed basis", {"learn_basis": False}),
                      ("fixed lambda", {"fixed_lambda": 0.01})]:
        m = build_model(cfg)
        y = torch.rand(2, 3, 128, 128)
        o = m(y)
        assert o["pred"].shape == y.shape
        assert torch.allclose(o["alpha"].sum(1), torch.ones_like(o["alpha"][:, 0]), atol=1e-4)
        o["pred"].mean().backward()
        tr = [p for p in m.parameters() if p.requires_grad]
        got = sum(p.grad is not None for p in tr)
        n = sum(p.numel() for p in m.parameters())
        print("  [%s] %-16s params %5.2fM  grads %d/%d"
              % ("OK " if got == len(tr) else "WARN", name, n / 1e6, got, len(tr)))

    a = build_model({}); b = build_model({"global_otf": True})
    b.load_state_dict(a.state_dict())
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        d = (a(x)["pred"] - b(x)["pred"]).abs().mean().item()
    assert d > 1e-6, "windowed inversion inactive"
    print("  windowed vs global |delta| = %.5f (must be > 0)" % d)

    # three real optimiser steps through the full loss
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Cfg(); cfg.use_perc = False
    m = build_model(asdict(cfg)).to(dev)
    crit = LOFTLoss(cfg, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    yb = torch.rand(2, 3, 128, 128, device=dev); xs = torch.rand(2, 3, 128, 128, device=dev)
    for i in range(3):
        out = m(yb)
        loss, parts = crit(m, out, yb, xs)
        opt.zero_grad(); loss.backward()
        gn = nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        print("  step %d: %s grad-norm %.3f" % (i, {k: round(v, 4) for k, v in parts.items()}, float(gn)))
    print("self-test passed")


def evaluate_split(cfg, split="test"):
    """PSNR/SSIM on a held-out split using the best checkpoint (EMA weights)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(asdict(cfg)).to(dev)
    ck = os.path.join(cfg.ckpt_dir, "%s_best.pt" % cfg.tag)
    if not os.path.exists(ck):
        ck = os.path.join(cfg.ckpt_dir, "%s_last.pt" % cfg.tag)
    if not os.path.exists(ck):
        raise SystemExit("no checkpoint in %s for tag '%s'" % (cfg.ckpt_dir, cfg.tag))
    step, best = load_ckpt(ck, m, device=dev)
    m.eval()
    root = os.path.dirname(os.path.dirname(cfg.val_blur))
    bd, sd = os.path.join(root, split, "blur"), os.path.join(root, split, "sharp")
    ds = PairedFolder(bd, sd, train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    ps, ss, rows = [], [], []
    with torch.no_grad():
        for i, (b, s) in enumerate(dl):
            b, s = b.to(dev), s.to(dev)
            p = forward_x8(m, b) if TTA else m(b)["pred"]
            pv, sv = psnr(p, s), ssim(p.clamp(0, 1), s)
            ps.append(pv); ss.append(sv); rows.append(dict(i=i, psnr=pv, ssim=sv))
    import statistics as st
    print("checkpoint %s (step %d)" % (os.path.basename(ck), step))
    print("%s: n=%d  PSNR %.4f +/- %.4f  SSIM %.4f  %s"
          % (split, len(ps), st.mean(ps), st.pstdev(ps), st.mean(ss),
             "(x8 self-ensemble)" if TTA else ""))
    json.dump(dict(split=split, n=len(ps), psnr=st.mean(ps), ssim=st.mean(ss),
                   tta=bool(TTA), per_image=rows),
              open(os.path.join(cfg.ckpt_dir, "%s_%s_results.json" % (cfg.tag, split)), "w"), indent=1)
    return st.mean(ps), st.mean(ss)


# ===================================================================== [10] CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LOFT training")
    ap.add_argument("--version", action="store_true", help="print version and changelog")
    ap.add_argument("--selftest", action="store_true", help="verify model + losses, no data")
    ap.add_argument("--smoke", action="store_true", help="alias for --selftest")
    ap.add_argument("--eval", action="store_true", help="PSNR/SSIM on the test split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--setup-dpdd", action="store_true", help="one-time: DPDD -> Drive zip")
    ap.add_argument("--fetch-lfdof", action="store_true",
                    help="download LFDOF (11,986 pairs) for pretraining")
    ap.add_argument("--lfdof-limit", type=int, help="cap LFDOF pairs (e.g. 4000)")
    ap.add_argument("--init", type=str, help="initialise weights from a checkpoint")
    ap.add_argument("--fetch-dpdd", action="store_true",
                    help="download DPDD to local disk only (no Drive needed)")
    ap.add_argument("--prepare", action="store_true", help="each session: Drive zip -> local")
    ap.add_argument("--train", action="store_true")
    # common overrides
    ap.add_argument("--steps", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--patch", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--tag", type=str);   ap.add_argument("--ckpt-dir", type=str)
    ap.add_argument("--train-blur", type=str); ap.add_argument("--train-sharp", type=str)
    ap.add_argument("--val-blur", type=str);   ap.add_argument("--val-sharp", type=str)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--blocks", type=int, help="refinement NAF blocks (8 default)")
    ap.add_argument("--tta", action="store_true", help="x8 self-ensemble at validation")
    ap.add_argument("--fast", action="store_true",
                    help="~2h-on-a-T4 preset: base 48, 4 blocks, patch 192, batch 8, 30k steps")
    ap.add_argument("--base", type=int, help="model width (48 default)")
    ap.add_argument("--w-perc", type=float, help="perceptual loss weight")
    ap.add_argument("--win", type=int, help="OTF window size (64 default)")
    ap.add_argument("--global-otf", action="store_true", help="v1 behaviour: one OTF per image")
    ap.add_argument("--no-otf", action="store_true",
                    help="CONTROL: disable OTF inversion, refinement network only")
    ap.add_argument("--no-perc", action="store_true", help="disable VGG perceptual loss")
    # ablation switches (one per paper table row)
    ap.add_argument("--real-basis", action="store_true", help="w/o complex basis")
    ap.add_argument("--no-fbt", action="store_true")
    ap.add_argument("--fixed-basis", action="store_true", help="w/o LOE (fixed Gaussian)")
    ap.add_argument("--fixed-lambda", type=float, help="w/o LWI")
    a = ap.parse_args()

    print("LOFT v%s" % __version__)
    cfg = Cfg()
    for k in ("steps", "batch", "patch", "lr", "tag", "train_blur", "train_sharp",
              "val_blur", "val_sharp", "save_every", "base", "w_perc", "win", "blocks"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if a.ckpt_dir:      cfg.ckpt_dir = a.ckpt_dir
    if a.global_otf:    cfg.global_otf = True;      cfg.tag += "_globalotf"
    if a.no_otf:        cfg.no_otf = True;          cfg.tag += "_nootf"
    if a.fast:
        cfg.base, cfg.blocks, cfg.patch, cfg.batch = 48, 4, 192, 8
        cfg.steps, cfg.lr, cfg.w_perc = 30000, 3e-4, 0.0
        cfg.val_every, cfg.save_every = 1000, 250
    if a.init:          cfg.init_from = a.init
    if a.tta:           globals()["TTA"] = True
    if a.no_perc:       cfg.use_perc = False
    if a.real_basis:    cfg.complex_basis = False; cfg.tag += "_realbasis"
    if a.no_fbt:        cfg.use_fbt = False;       cfg.tag += "_nofbt"
    if a.fixed_basis:   cfg.learn_basis = False;   cfg.tag += "_fixedbasis"
    if a.fixed_lambda is not None:
        cfg.fixed_lambda = a.fixed_lambda;         cfg.tag += "_fixedlam"

    if a.version:
        print("LOFT v%s" % __version__); print(__changelog__); raise SystemExit
    if a.selftest or a.smoke:  selftest()
    elif a.eval:           evaluate_split(cfg, a.split)
    elif a.fetch_lfdof:    fetch_lfdof(limit=a.lfdof_limit)
    elif a.fetch_dpdd:     fetch_dpdd()
    elif a.setup_dpdd:     setup_dpdd()
    elif a.prepare:        prepare()
    elif a.train:          train(cfg)
    else:                  ap.print_help()
