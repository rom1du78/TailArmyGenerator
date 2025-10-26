#!/usr/bin/env python3
# tail_army_generator.py — V7.4
# Pipeline automatique :
#   Étape 1 : Génération de tuiles hexagonales depuis input_ground/ vers data/temp/
#   Étape 1b : Pré-traitement objects (input_object → data/object_processed) par découpe alpha
#   Étape 2 : Copie de chaque terrain seul dans output_tails/
#              puis superposition (data/temp + data/object_processed → output_tails/)
#   Étape 3 : Nettoyage automatique du dossier data/temp/
# Usage :
#   python tail_army_generator.py

from PIL import Image
import numpy as np
import sys, os, glob, shutil
from pathlib import Path

# ============================================================
# === CONFIGURATION GÉNÉRALE ===
# ============================================================
DATA_DIR = Path("data/png_template")

TEMPLATE_BORDER = DATA_DIR / "hex_border_overlay.png"
INNER_MASK = DATA_DIR / "hex_inner_mask.png"
OUTER_MASK = DATA_DIR / "hex_outer_mask.png"
MASK_PATH = DATA_DIR / "mask.png"

# Dossiers de travail
INPUT_GROUND_DIR      = Path("input_ground")
TEMP_DIR              = Path("data/temp")
INPUT_OBJECT_DIR      = Path("input_object")
PROCESSED_OBJECT_DIR  = Path("data/object_processed")
OUTPUT_TAILS_DIR      = Path("output_tails")

DEFAULT_OPACITY = 0.85

# ============================================================
# === OUTILS COMMUNS ===
# ============================================================
def safe_open_rgba(path):
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def get_mask_info(mask_img_l):
    """Retourne (mask_array uint8, (cx,cy) centre de la zone blanche, (x0,y0,x1,y1) bbox blanche)."""
    m = np.array(mask_img_l)
    white = m > 5
    ys, xs = np.where(white)
    if len(xs) == 0 or len(ys) == 0:
        # masque vide : considérer tout comme blanc
        h, w = m.shape
        return m, (w // 2, h // 2), (0, 0, w - 1, h - 1)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return m, (int(cx), int(cy)), (int(x0), int(y0), int(x1), int(y1))

def fits_in_mask_pixelperfect(overlay_rgba, mask_array, pos_x, pos_y):
    """Vérifie que tous les pixels opaques de overlay tombent dans zone blanche du masque."""
    ov = np.array(overlay_rgba)
    alpha = ov[..., 3] > 0
    h, w = alpha.shape
    if pos_x < 0 or pos_y < 0 or pos_x + w > mask_array.shape[1] or pos_y + h > mask_array.shape[0]:
        return False
    zone = mask_array[pos_y:pos_y + h, pos_x:pos_x + w]
    return np.all(zone[alpha] > 5)

# ============================================================
# === ÉTAPE 1 — input_ground → data/temp ===
# ============================================================
def generate_tiles():
    os.makedirs(TEMP_DIR, exist_ok=True)

    border     = safe_open_rgba(TEMPLATE_BORDER)
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
    for p in glob.glob(str(INPUT_GROUND_DIR / "*")):
        if os.path.isdir(p): 
            continue
        try:
            art = safe_open_rgba(p)
        except Exception as e:
            print("Skip:", p, e)
            continue

        # Extension horizontale pour couvrir les pointes
        EXPAND_X = 16
        EXPAND_Y = 0
        target_w_expanded = target_w + EXPAND_X
        target_h_expanded = target_h + EXPAND_Y

        art_aspect    = art.width / art.height
        target_aspect = target_w_expanded / target_h_expanded
        if art_aspect > target_aspect:
            new_h = target_h_expanded
            new_w = round(new_h * art_aspect)
        else:
            new_w = target_w_expanded
            new_h = round(new_w / art_aspect)

        art_resized = art.resize((new_w, new_h), resample=Image.LANCZOS)

        # Centrage dans la zone intérieure
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ox = x0 + (target_w - new_w) // 2
        oy = y0 + (target_h - new_h) // 2
        canvas.alpha_composite(art_resized, (ox, oy))

        # Application masque intérieur
        canvas.putalpha(Image.composite(Image.new("L", (W, H), 255),
                                        Image.new("L", (W, H), 0),
                                        inner_mask))
        content = Image.composite(canvas, Image.new("RGBA", (W, H), (0, 0, 0, 0)), inner_mask)

        # Composition finale + bord
        final = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        final.alpha_composite(content)
        final.alpha_composite(safe_open_rgba(TEMPLATE_BORDER))
        final.putalpha(Image.composite(final.getchannel("A"), outer_mask, outer_mask))

        out_path = TEMP_DIR / os.path.basename(p)  # conserve le nom source
        final.save(out_path, optimize=True)
        count += 1
        print("Wrote", out_path)

    print(f"\n✅ Étape 1 terminée : {count} tuiles hexagonales générées dans {TEMP_DIR}\n")

# ============================================================
# === ÉTAPE 1b — input_object → data/object_processed (découpe alpha) ===
# ============================================================
def preprocess_objects(alpha_threshold=20):
    PROCESSED_OBJECT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in glob.glob(str(INPUT_OBJECT_DIR / "*")):
        if os.path.isdir(p):
            continue
        try:
            obj = safe_open_rgba(p)
        except Exception as e:
            print("Skip object:", p, e)
            continue

        data = np.array(obj)
        alpha = data[..., 3]
        ys, xs = np.where(alpha > alpha_threshold)
        if len(xs) == 0 or len(ys) == 0:
            print("Skip empty (fully transparent):", p)
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cropped = obj.crop((x1, y1, x2 + 1, y2 + 1))

        out_path = PROCESSED_OBJECT_DIR / os.path.basename(p)
        cropped.save(out_path, optimize=True)
        count += 1
        print("Processed object:", out_path)

    print(f"\n✅ Étape 1b terminée : {count} objets découpés dans {PROCESSED_OBJECT_DIR}\n")

# ============================================================
# === RECHERCHE DU SCALE MAXIMAL (centrage sur la zone blanche) ===
# ============================================================
def find_max_scale_centered(bg_size, ov_rgba, mask_array, mask_center):
    """
    Centre l'objet sur le centre de la zone blanche du masque.
    Cherche par dichotomie le scale maximal qui ne déborde pas.
    """
    bw, bh = bg_size
    cx, cy = mask_center

    # bornes de recherche
    s_low, s_high = 0.05, 4.0
    best_scale = s_low
    best_img = ov_rgba.resize((max(1, int(ov_rgba.width * s_low)), max(1, int(ov_rgba.height * s_low))), Image.LANCZOS)
    best_pos = ((bw - best_img.width) // 2, (bh - best_img.height) // 2)

    for _ in range(24):  # précision suffisante
        s_mid = 0.5 * (s_low + s_high)
        w = max(1, int(ov_rgba.width * s_mid))
        h = max(1, int(ov_rgba.height * s_mid))
        candidate = ov_rgba.resize((w, h), Image.LANCZOS)

        # position centrée sur le centre de la zone blanche
        pos_x = int(round(cx - w / 2))
        pos_y = int(round(cy - h / 2))

        if fits_in_mask_pixelperfect(candidate, mask_array, pos_x, pos_y):
            # ok, on peut agrandir encore
            best_scale, best_img, best_pos = s_mid, candidate, (pos_x, pos_y)
            s_low = s_mid
        else:
            # trop grand, on réduit
            s_high = s_mid

    return best_img, best_pos, best_scale

# ============================================================
# === SUPERPOSITION (temp + processed_object → output_tails) ===
# ============================================================
def overlay_image(ground_path, overlay_path, output_path, opacity=DEFAULT_OPACITY):
    bg = safe_open_rgba(ground_path)
    ov = safe_open_rgba(overlay_path)

    # masque (dimensionné au ground)
    mask_l = Image.open(MASK_PATH).convert("L").resize(bg.size, Image.LANCZOS)
    mask_array, (mcx, mcy), _ = get_mask_info(mask_l)

    # recherche du scale maximal au centre de la zone blanche
    ov_scaled, (pos_x, pos_y), scale = find_max_scale_centered(bg.size, ov, mask_array, (mcx, mcy))

    if opacity < 1.0:
        r, g, b, a = ov_scaled.split()
        a = a.point(lambda p: int(p * opacity))
        ov_scaled = Image.merge("RGBA", (r, g, b, a))

    result = bg.copy()
    result.paste(ov_scaled, (pos_x, pos_y), ov_scaled)
    result.save(output_path)
    print(f"✅ {output_path.name} créé — scale={scale:.3f}, pos=({pos_x},{pos_y})")

# ============================================================
# === MAIN PIPELINE ===
# ============================================================
def main():
    print("\n=== Étape 1 : Génération des tuiles hexagonales ===")
    generate_tiles()

    print("=== Étape 1b : Pré-traitement des objets (découpe alpha) ===")
    preprocess_objects(alpha_threshold=20)

    print("=== Étape 2 : Superposition des calques ===")
    OUTPUT_TAILS_DIR.mkdir(parents=True, exist_ok=True)

    if not MASK_PATH.exists():
        sys.exit(f"❌ Erreur : masque introuvable : {MASK_PATH}")

    grounds = sorted([f for f in TEMP_DIR.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])
    objects = sorted([f for f in PROCESSED_OBJECT_DIR.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])

    if not grounds:
        print(f"⚠️ Aucun fichier trouvé dans {TEMP_DIR}.")
        return
    if not objects:
        print(f"⚠️ Aucun fichier trouvé dans {PROCESSED_OBJECT_DIR}.")
        return

    total = 0

    # Copie des terrains "purs"
    print("\n--- Copie des terrains seuls ---")
    for g in grounds:
        dest = OUTPUT_TAILS_DIR / g.name
        shutil.copy2(g, dest)
        print(f"✅ Copié {g.name} → {dest.name}")
        total += 1

    # Superpositions (centrées + scale maximal)
    print("\n--- Superpositions (centrage + scale maximal) ---")
    for g in grounds:
        for o in objects:
            out_name = f"{g.stem}_{o.stem}.png"
            out_path = OUTPUT_TAILS_DIR / out_name
            overlay_image(g, o, out_path)
            total += 1

    print(f"\n✅ Étape 2 terminée : {total} images générées dans {OUTPUT_TAILS_DIR.resolve()}")

    # Nettoyage automatique du dossier data/temp
    print("\n🧹 Nettoyage du dossier data/temp/")
    for f in TEMP_DIR.glob("*"):
        try:
            f.unlink()
        except Exception as e:
            print(f"Impossible de supprimer {f}: {e}")
    print("✅ Dossier data/temp vidé.\n")

if __name__ == "__main__":
    main()
