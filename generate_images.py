"""
Generate SSIM-emphasis figures for the paper.

Two figure types:
  A) 2-row layout per image:
       Row 1: HR patch | Bicubic patch | Ours patch
       Row 2: (blank)  | Bicubic SSIM error map | Ours SSIM error map
     Brighter = worse SSIM. Ours should be darker near edges.

  B) Difference-map figure:
       HR patch | Bicubic |Ours | Bic error (hot) | Ours error (hot)
     Highlights where pixel-level error is reduced.

Images chosen: the ones where Ours has the highest SSIM gain over Bicubic.
"""

import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

BASE     = os.path.dirname(os.path.abspath(__file__))
OURS_DIR = os.path.join(BASE, "EPGDUN_x4", "results")
DATA_DIR = os.path.join(BASE, "data", "benchmark")
OUT_DIR  = os.path.join(BASE, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)
SCALE = 4

# ── Helpers ────────────────────────────────────────────────────────────────
def load(p):
    return np.array(Image.open(p).convert("RGB"))

def to_y(rgb):
    r,g,b = rgb[...,0]/255., rgb[...,1]/255., rgb[...,2]/255.
    return np.clip(16 + 65.481*r + 128.553*g + 24.966*b, 0, 255).astype(np.float64)

def metrics(sr, hr):
    s  = SCALE
    sy = to_y(sr)[s:-s, s:-s]
    hy = to_y(hr)[s:-s, s:-s]
    p  = calc_psnr(hy, sy, data_range=255.)
    s2 = calc_ssim(hy, sy, data_range=255.)
    return round(p, 2), round(s2, 4)

def ssim_map(sr_patch, hr_patch):
    """Per-pixel SSIM map on Y channel. Returns (H,W) float in [0,1]."""
    sy = to_y(sr_patch)
    hy = to_y(hr_patch)
    _, S = calc_ssim(hy, sy, data_range=255., full=True)
    # S is in [-1,1]; clip to [0,1] and invert so brighter = worse
    S = np.clip(S, 0, 1)
    return 1.0 - S          # error map: brighter = worse SSIM

def diff_map(sr_patch, hr_patch):
    """Absolute pixel-difference map, normalised to [0,1]."""
    d = np.abs(sr_patch.astype(np.float32) - hr_patch.astype(np.float32))
    d = d.mean(axis=2)       # collapse RGB
    return d / d.max() if d.max() > 0 else d

def bicubic_up(lr, h, w):
    return np.array(Image.fromarray(lr).resize((w, h), Image.BICUBIC))

def best_crop(hr_img, cw=128, ch=128, stride=16):
    from scipy.ndimage import sobel
    gray = np.mean(hr_img.astype(np.float32), axis=2)
    edge = np.abs(sobel(gray, 0)) + np.abs(sobel(gray, 1))
    H, W = gray.shape
    best, xy = -1, (0, 0)
    for y in range(0, H-ch, stride):
        for x in range(0, W-cw, stride):
            sc = edge[y:y+ch, x:x+cw].mean()
            if sc > best:
                best, xy = sc, (x, y)
    return xy[0], xy[1], cw, ch

def load_images(ds_name, fname):
    hp  = os.path.join(DATA_DIR, ds_name, "HR",         "X%d"%SCALE, fname)
    lp  = os.path.join(DATA_DIR, ds_name, "LR_bicubic", "X%d"%SCALE, fname)
    op  = os.path.join(OURS_DIR, "SR",     ds_name,      "X%d"%SCALE, fname)
    for p in [hp, op]:
        if not os.path.exists(p):
            return None, None, None, fname
    hr  = load(hp); our = load(op)
    H, W = hr.shape[:2]
    lr   = load(lp) if os.path.exists(lp) else None
    bic  = bicubic_up(lr, H, W) if lr is not None else bicubic_up(
           np.array(Image.fromarray(hr).resize((W//SCALE,H//SCALE),Image.BICUBIC)), H, W)
    h=min(hr.shape[0],bic.shape[0],our.shape[0])
    w=min(hr.shape[1],bic.shape[1],our.shape[1])
    return hr[:h,:w], bic[:h,:w], our[:h,:w], fname


# ── Figure A: Patch row + SSIM error map row ──────────────────────────────
def make_ssim_figure(ds_name, fname, crop_box=None, out_name=None, crop_size=128):
    hr, bic, our, fname = load_images(ds_name, fname)
    if hr is None:
        print("  [SKIP] %s/%s" % (ds_name, fname)); return

    if crop_box is None:
        cx, cy, cw, ch = best_crop(hr, cw=crop_size, ch=crop_size)
    else:
        cx, cy, cw, ch = crop_box
        h, w = hr.shape[:2]
        cx=min(cx,w-cw-1); cy=min(cy,h-ch-1)

    def crop(img): return img[cy:cy+ch, cx:cx+cw]

    hr_p  = crop(hr);  bic_p = crop(bic);  our_p = crop(our)

    bic_psnr, bic_ssim = metrics(bic_p, hr_p)
    our_psnr, our_ssim = metrics(our_p, hr_p)

    bic_smap = ssim_map(bic_p, hr_p)
    our_smap = ssim_map(our_p, hr_p)

    short = fname.replace('_SRF_4_HR','').replace('.png','')
    out_name = out_name or ("ssim_%s_%s" % (ds_name, short))

    # ── 2-row figure ────────────────────────────────────────────────────────
    # Row 0: full image | HR patch | Bic patch | Ours patch
    # Row 1: (blank)    | (blank)  | Bic SSIM error | Ours SSIM error
    fig = plt.figure(figsize=(13, 7))
    gs  = GridSpec(2, 4, figure=fig,
                   width_ratios=[2.5, 1.8, 1.8, 1.8],
                   height_ratios=[1, 1],
                   wspace=0.05, hspace=0.35)

    # ── Row 0 ───────────────────────────────────────────────────────────────
    # Full image with crop box
    ax_full = fig.add_subplot(gs[:, 0])   # span both rows
    ax_full.imshow(hr, interpolation='bilinear')
    ax_full.add_patch(mpatches.Rectangle(
        (cx,cy), cw, ch, linewidth=2.5, edgecolor='red', facecolor='none'))
    ax_full.set_xlabel("Image: %s\nfrom %s ×%d" % (short,ds_name,SCALE),
                       fontsize=9, labelpad=4)
    ax_full.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    for sp in ax_full.spines.values(): sp.set_visible(False)

    patch_axes = [
        (gs[0,1], "HR\n(Ground Truth)", hr_p,  None,  None,  False),
        (gs[0,2], "Bicubic",            bic_p, bic_psnr, bic_ssim, False),
        (gs[0,3], "Ours (Proposed)",    our_p, our_psnr, our_ssim, True),
    ]
    for spec, title, patch, p, s, is_ours in patch_axes:
        ax = fig.add_subplot(spec)
        ax.imshow(patch, interpolation='nearest')
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor('red'); sp.set_linewidth(2); sp.set_visible(True)
        lbl = ("%s\n%.2fdB / %.4f"%(title,p,s)) if p is not None else title
        ax.set_xlabel(lbl, fontsize=9,
                      fontweight='bold' if is_ours else 'normal',
                      color='#CC0000'   if is_ours else 'black',
                      labelpad=4)

    # ── Row 1: SSIM error maps ───────────────────────────────────────────────
    # Shared colour scale so maps are directly comparable
    vmax = max(bic_smap.max(), our_smap.max())
    norm = Normalize(vmin=0, vmax=vmax)

    # Empty cell under full image already occupied by gs[:,0]
    # Blank cell under HR patch
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_blank.axis('off')
    ax_blank.set_facecolor('white')

    error_axes = [
        (gs[1,2], "SSIM Error Map\n(Bicubic)",   bic_smap, False),
        (gs[1,3], "SSIM Error Map\n(Ours)",       our_smap, True),
    ]

    for spec, title, smap, is_ours in error_axes:
        ax  = fig.add_subplot(spec)
        im  = ax.imshow(smap, cmap='hot', norm=norm, interpolation='nearest')
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor('#CC0000' if is_ours else 'grey')
            sp.set_linewidth(2 if is_ours else 1)
            sp.set_visible(True)
        mean_err = smap.mean()
        ax.set_xlabel("%s\nmean error: %.4f" % (title, mean_err),
                      fontsize=9,
                      fontweight='bold' if is_ours else 'normal',
                      color='#CC0000'   if is_ours else 'black',
                      labelpad=4)

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.38])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('SSIM Error\n(1 − SSIM)', fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7)

    # Annotation arrow/text explaining the maps
    fig.text(0.62, 0.09,
             "Darker = better structural similarity",
             fontsize=8.5, color='#555555', ha='center',
             style='italic')

    out_path = os.path.join(OUT_DIR, "%s.png" % out_name)
    plt.savefig(out_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] %s" % out_path)
    print("       Bicubic %.2f/%.4f | Ours %.2f/%.4f | dSSIM=%+.4f"
          % (bic_psnr,bic_ssim,our_psnr,our_ssim,our_ssim-bic_ssim))


# ── Figure B: Pixel difference map comparison ────────────────────────────
def make_diff_figure(ds_name, fname, crop_box=None, out_name=None, crop_size=128):
    hr, bic, our, fname = load_images(ds_name, fname)
    if hr is None:
        print("  [SKIP] %s/%s" % (ds_name, fname)); return

    if crop_box is None:
        cx, cy, cw, ch = best_crop(hr, cw=crop_size, ch=crop_size)
    else:
        cx, cy, cw, ch = crop_box
        h, w = hr.shape[:2]
        cx=min(cx,w-cw-1); cy=min(cy,h-ch-1)

    def crop(img): return img[cy:cy+ch, cx:cx+cw]

    hr_p=crop(hr); bic_p=crop(bic); our_p=crop(our)
    bic_psnr,bic_ssim = metrics(bic_p, hr_p)
    our_psnr,our_ssim = metrics(our_p, hr_p)
    bic_d = diff_map(bic_p, hr_p)
    our_d = diff_map(our_p, hr_p)

    short    = fname.replace('_SRF_4_HR','').replace('.png','')
    out_name = out_name or ("diff_%s_%s" % (ds_name, short))

    fig = plt.figure(figsize=(15, 3.8))
    gs  = GridSpec(1, 6, figure=fig,
                   width_ratios=[2.5, 1.8, 1.8, 1.8, 1.8, 1.8],
                   wspace=0.05)

    # Full image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(hr, interpolation='bilinear')
    ax0.add_patch(mpatches.Rectangle(
        (cx,cy), cw, ch, linewidth=2.5, edgecolor='red', facecolor='none'))
    ax0.set_xlabel("Image: %s\nfrom %s ×%d" % (short,ds_name,SCALE),
                   fontsize=8.5, labelpad=4)
    ax0.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    for sp in ax0.spines.values(): sp.set_visible(False)

    vmax = max(bic_d.max(), our_d.max())
    norm = Normalize(vmin=0, vmax=vmax)

    panels = [
        (gs[0,1], "HR\n(Ground Truth)", hr_p,  None,       None,       'patch', False),
        (gs[0,2], "Bicubic",            bic_p, bic_psnr,   bic_ssim,   'patch', False),
        (gs[0,3], "Ours",               our_p, our_psnr,   our_ssim,   'patch', True),
        (gs[0,4], "Bicubic Error",      bic_d, bic_psnr,   bic_ssim,   'diff',  False),
        (gs[0,5], "Ours Error",         our_d, our_psnr,   our_ssim,   'diff',  True),
    ]

    for spec, title, data, p, s, kind, is_ours in panels:
        ax = fig.add_subplot(spec)
        if kind == 'patch':
            ax.imshow(data, interpolation='nearest')
            lbl = ("%s\n%.2fdB/%.4f"%(title,p,s)) if p is not None else title
        else:
            ax.imshow(data, cmap='hot', norm=norm, interpolation='nearest')
            mean_e = data.mean()
            lbl = "%s\nmean=%.4f" % (title, mean_e)

        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor('#CC0000' if is_ours else 'grey')
            sp.set_linewidth(2 if is_ours else 1.2)
            sp.set_visible(True)
        ax.set_xlabel(lbl, fontsize=8.5,
                      fontweight='bold' if is_ours else 'normal',
                      color='#CC0000'   if is_ours else 'black',
                      labelpad=4)

    plt.suptitle(
        "Pixel Error Maps — darker = less reconstruction error   "
        "(dSSIM = %+.4f,  dPSNR = %+.2f dB)" % (our_ssim-bic_ssim, our_psnr-bic_psnr),
        fontsize=10, fontweight='bold', y=1.02
    )

    out_path = os.path.join(OUT_DIR, "%s.png" % out_name)
    plt.savefig(out_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] %s" % out_path)
    print("       Bicubic %.2f/%.4f | Ours %.2f/%.4f | dSSIM=%+.4f dP=%+.2f"
          % (bic_psnr,bic_ssim,our_psnr,our_ssim,our_ssim-bic_ssim,our_psnr-bic_psnr))


# ── Run on best-gain images ────────────────────────────────────────────────
TARGETS = [
    # (dataset,     filename,                  crop_box,  out_prefix)
    ("Set14",    "img_014_SRF_4_HR.png",       None,     "img014_Set14"),
    ("Urban100", "img_004_SRF_4_HR.png",       None,     "img004_Urban100"),
    ("Urban100", "img_023_SRF_4_HR.png",       None,     "img023_Urban100"),
    ("Urban100", "img_011_SRF_4_HR.png",       None,     "img011_Urban100"),
    ("Set5",     "butterfly.png",              None,     "butterfly_Set5"),
]

if __name__ == "__main__":
    print("\n=== Figure A: SSIM error map figures ===\n")
    for ds, fn, crop, pfx in TARGETS:
        print("  %s / %s" % (ds, fn))
        make_ssim_figure(ds, fn, crop, "ssimfig_%s" % pfx)

    print("\n=== Figure B: Pixel difference map figures ===\n")
    for ds, fn, crop, pfx in TARGETS:
        print("  %s / %s" % (ds, fn))
        make_diff_figure(ds, fn, crop, "difffig_%s" % pfx)

    print("\nAll saved to: %s" % OUT_DIR)
