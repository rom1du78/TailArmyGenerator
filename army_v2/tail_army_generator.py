#!/usr/bin/env python3
# tail_army_generator.py — V6.7
# Pipeline automatique :
#   Étape 1 : Génération de tuiles hexagonales depuis sources/ vers ground/ (même nom)
#   Étape 2 : Copie de chaque ground seul dans result/
#              puis superposition intelligente (ground + object → result/)
# Usage :
#   python tail_army_generator.py
# (aucun argument requis)

from PIL import Image
import numpy as np
import sys, os, glob, shutil
from pathlib import Path

# ============================================================
# === CONFIGURATION GÉNÉRALE ===
# ============================================================
DATA_DIR = Path("data_army_generator")

TEMPLATE_BORDER = DATA_DIR / "hex_border_overlay.png"
INNER_MASK = DATA_DIR / "hex_inner_mask.png"
OUTER_MASK = DATA_DIR / "hex_outer_mask.png"
MASK_PATH = DATA_DIR / "mask.png"

# Dossiers de travail
SOURCES_DIR = Path("sources")
GROUND_DIR = Path("ground")
OBJECT_DIR = Path("object")
RESULT_DIR = Path("result")

# Paramètres par défaut
DEFAULT_OPACITY = 0.85

# ============================================================
# === ÉTAPE 1 — GÉNÉRATION DES TUILES HEXAGONALES (sources → ground)
# ============================================================
def generate_tiles():
    os.makedirs(GROUND_DIR, exist_ok=True)

    border = Image.open(TEMPLATE_BORDER).convert("RGBA")
    inner_mask = Image.open(INNER_MASK).convert("L")
    outer_mask = Image.open(OUTER_MASK).convert("L")
    W, H = outer_mask.size

    inner_np = np.array(inner_mask)
    ys, xs = np.nonzero(inner_np > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    target_w = x1 - x0 + 1
    target_h = y1 - y0 + 1

    count = 0
    for p in glob.glob(str(SOURCES_DIR / "*")):
        if os.path.isdir(p):
            continue
        try:
            art = Image.open(p).convert("RGBA")
        except Exception as e:
            print("Skip:", p, e)
            continue

        art.info.pop("icc_profile", None)

        EXPAND_X = 16
        EXPAND_Y = 0

        target_w_expanded = target_w + EXPAND_X
        target_h_expanded = target_h + EXPAND_Y

        art_aspect = art.width / art.height
        target_aspect = target_w_expanded / target_h_expanded

        if art_aspect > target_aspect:
            new_h = target_h_expanded
            new_w = round(new_h * art_aspect)
        else:
            new_w = target_w_expanded
            new_h = round(new_w / art_aspect)

        art_resized = art.resize((new_w, new_h), resample=Image.LANCZOS)

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ox = x0 + (target_w - new_w) // 2
        oy = y0 + (target_h - new_h) // 2
        canvas.alpha_composite(art_resized, (ox, oy))

        canvas.putalpha(
            Image.composite(
                Image.new("L", (W, H), 255), Image.new("L", (W, H), 0), inner_mask
            )
        )
        content = Image.composite(canvas, Image.new("RGBA", (W, H), (0, 0, 0, 0)), inner_mask)

        final = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        final.alpha_composite(content)
        final.alpha_composite(border)
        final.putalpha(Image.composite(final.getchannel("A"), outer_mask, outer_mask))

        base = os.path.basename(p)
        out_path = GROUND_DIR / base  # ✅ conserve le même nom que la source
        final.save(out_path, optimize=True)
        count += 1
        print("Wrote", out_path)

    print(f"\n✅ Étape 1 terminée : {count} tuiles hexagonales générées dans {GROUND_DIR}\n")

# ============================================================
# === ÉTAPE 2 — SUPERPOSITION GROUND / OBJECT (ground + object → result)
# ============================================================

def trim_smart(img: Image.Image, tolerance=15):
    """
    Supprime les bords vides, semi-transparents ou de couleur uniforme (vert clair, beige, etc.).
    Cette version corrige le problème des objets trop petits en rognant les marges colorées.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    data = np.array(img)
    alpha = data[..., 3]
    rgb = data[..., :3]

    # Couleur dominante du fond (zones quasi transparentes ou bords)
    if np.any(alpha < 250):
        bg_color = np.median(rgb[alpha < 250], axis=0)
    else:
        bg_color = np.median(rgb[:10, :10], axis=(0, 1))

    # Différence de couleur par rapport au fond
    diff = np.sqrt(((rgb - bg_color) ** 2).sum(axis=2))
    # "Vide" = pixels très proches du fond ou presque transparents
    mask = (diff > tolerance) & (alpha > 30)

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return img.crop((x1, y1, x2 + 1, y2 + 1))

def fits_in_mask_pixelperfect(overlay_rgba, mask_array, pos_x, pos_y):
    ov = np.array(overlay_rgba)
    alpha = ov[..., 3] > 0
    h, w = alpha.shape
    if pos_x < 0 or pos_y < 0 or pos_x + w > mask_array.shape[1] or pos_y + h > mask_array.shape[0]:
        return False
    mask_zone = mask_array[pos_y:pos_y+h, pos_x:pos_x+w]
    return np.all(mask_zone[alpha] > 5)

def find_best_fit_mask_precise(bg_size, ov, mask_array):
    bw, bh = bg_size
    best_scale, best_pos, best_img = 0, (0, 0), None

    for scale in np.linspace(0.1, 2.0, 120):
        new_size = (int(ov.width * scale), int(ov.height * scale))
        if new_size[0] <= 2 or new_size[1] <= 2:
            continue
        ov_scaled = ov.resize(new_size, Image.LANCZOS)
        h, w = new_size[1], new_size[0]
        pos_x = (bw - w) // 2
        step = max(1, h // 64)
        for y in range(0, bh - h + 1, step):
            if fits_in_mask_pixelperfect(ov_scaled, mask_array, pos_x, y):
                best_scale = scale
                best_pos = (pos_x, y)
                best_img = ov_scaled
                break
    if best_img is None:
        new_size = (max(4, int(ov.width * 0.1)), max(4, int(ov.height * 0.1)))
        best_img = ov.resize(new_size, Image.LANCZOS)
        best_pos = ((bw - new_size[0]) // 2, (bh - new_size[1]) // 2)
        best_scale = 0.1
    return best_img, best_pos, best_scale

def overlay_image(ground_path, overlay_path, output_path, opacity=DEFAULT_OPACITY):
    bg = Image.open(ground_path).convert("RGBA")
    ov = Image.open(overlay_path).convert("RGBA")

    # ✅ Nouveau rognage intelligent
    ov = trim_smart(ov, tolerance=15)

    mask = Image.open(MASK_PATH).convert("L").resize(bg.size, Image.LANCZOS)
    mask_array = np.array(mask)

    ov_scaled, (pos_x, pos_y), scale = find_best_fit_mask_precise(bg.size, ov, mask_array)

    if opacity < 1.0:
        r, g, b, a = ov_scaled.split()
        a = a.point(lambda p: int(p * opacity))
        ov_scaled = Image.merge("RGBA", (r, g, b, a))

    result = bg.copy()
    result.paste(ov_scaled, (pos_x, pos_y), ov_scaled)
    result.save(output_path)
    print(f"✅ {output_path.name} créé — scale={scale:.2f}, pos=({pos_x},{pos_y})")

# ============================================================
# === MAIN (PIPELINE AUTOMATIQUE) ===
# ============================================================
def main():
    print("\n=== Étape 1 : Génération des tuiles hexagonales ===")
    generate_tiles()

    print("=== Étape 2 : Superposition des calques ===")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not MASK_PATH.exists():
        sys.exit(f"❌ Erreur : masque introuvable : {MASK_PATH}")

    grounds = sorted([f for f in GROUND_DIR.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])
    objects = sorted([f for f in OBJECT_DIR.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])

    if not grounds:
        print(f"⚠️ Aucun fichier trouvé dans {GROUND_DIR}.")
        return
    if not objects:
        print(f"⚠️ Aucun fichier trouvé dans {OBJECT_DIR}.")
        return

    total = 0

    # --- Étape 2A : copier chaque ground seul dans result ---
    print("\n--- Copie des grounds seuls ---")
    for g in grounds:
        dest = RESULT_DIR / g.name
        shutil.copy2(g, dest)
        print(f"✅ Copié {g.name} → {dest.name}")
        total += 1

    # --- Étape 2B : superpositions ground + object ---
    print("\n--- Superpositions ground + object ---")
    for g in grounds:
        for o in objects:
            out_name = f"{g.stem}_{o.stem}.png"
            out_path = RESULT_DIR / out_name
            overlay_image(g, o, out_path)
            total += 1

    print(f"\n✅ Étape 2 terminée : {total} images générées dans {RESULT_DIR.resolve()}")

if __name__ == "__main__":
    main()
