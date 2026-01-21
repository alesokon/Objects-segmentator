"""
Objects Segmentator:
- Grounding DINO: detekuje objekty podle textového dotazu (open-vocabulary detection)
- SAM 2: z detekovaných boxů udělá přesné masky (instance segmentation)
- Aplikace:
  - upload obrázku
  - zadání textu "co hledám" (např. "person" / "car" / "dog" / "bottle")
  - nastavení prahů citlivosti
  - výstup: počet instancí + instance overlay + union overlay + tabulka
"""

import io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Processor,
    Sam2Model,
)

# ----------------------------
# Datová struktura pro jednu nalezenou instanci
# ----------------------------
@dataclass
class Instance:
    """
    Reprezentuje 1 konkrétní nalezený objekt (1 instanci):
    - idx: pořadové číslo (kvůli popiskům v obrázku a tabulce)
    - label: textový label (podle dotazu/labelů)
    - score: confidence z detektoru (GroundingDINO)
    - box_xyxy: bounding box v pixelech (x1,y1,x2,y2)
    - mask: binární maska HxW (True = pixel patří objektu)
    - area_px: plocha masky v pixelech (pro tabulku)
    """
    idx: int
    label: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    mask: np.ndarray  # HxW bool
    area_px: int


# ----------------------------
# Vizualizační pomocné funkce
# ----------------------------
def _safe_font(size: int = 16):
    """Fallback font."""
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def deterministic_color(i: int) -> Tuple[int, int, int]:
    """Deterministické výrazné barvy."""
    phi = 0.61803398875
    h = (i * phi) % 1.0
    s = 0.7
    v = 0.95
    k = int(h * 6)
    f = h * 6 - k
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    V = int(255 * v)
    k = k % 6
    if k == 0:
        return (V, t, p)
    if k == 1:
        return (q, V, p)
    if k == 2:
        return (p, V, t)
    if k == 3:
        return (p, q, V)
    if k == 4:
        return (t, p, V)
    return (V, p, q)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Překreslení (overlay) masky přes obrázek."""
    out = base_rgb.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = out[mask] * (1 - alpha) + col * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes_and_ids(img: Image.Image, insts: List[Instance]) -> Image.Image:
    """Vykreslí boxy + popisky (idx: label (score))."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _safe_font(16)

    for ins in insts:
        x1, y1, x2, y2 = ins.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)

        txt = f"{ins.idx}: {ins.label} ({ins.score:.2f})"
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y_top = max(0, y1 - th - 6)
        draw.rectangle([x1, y_top, x1 + tw + 8, y_top + th + 6], fill=(0, 0, 0))
        draw.text((x1 + 4, y_top + 3), txt, fill=(255, 255, 255), font=font)

    return out


def build_overlays(pil_img: Image.Image, insts: List[Instance], alpha: float, draw_boxes: bool):
    """
    Vytvoří 2 vizualizace:
    - instance_overlay: každá instance jinou barvou (po instancích)
    - union_overlay: všechny instance dohromady (sjednocení masek)
    """
    base = np.array(pil_img.convert("RGB"))
    inst_overlay = base.copy()
    union_overlay = base.copy()

    if insts:
        # union maska = OR přes všechny instanční masky
        union_mask = np.zeros((base.shape[0], base.shape[1]), dtype=bool)
        for ins in insts:
            union_mask |= ins.mask

        # Zelený překryv pro union
        union_overlay = overlay_mask(union_overlay, union_mask, (0, 255, 0), alpha)

        # Instance overlay = každá instance jinou barvou
        for i, ins in enumerate(insts, start=1):
            inst_overlay = overlay_mask(inst_overlay, ins.mask, deterministic_color(i), alpha)

    inst_img = Image.fromarray(inst_overlay)
    union_img = Image.fromarray(union_overlay)

    # Boxy + popisky
    if draw_boxes and insts:
        inst_img = draw_boxes_and_ids(inst_img, insts)
        union_img = draw_boxes_and_ids(union_img, insts)

    return inst_img, union_img


def table_rows(insts: List[Instance], img_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Tabulka s metrikami instancí."""
    W, H = img_size
    rows = []
    for ins in insts:
        x1, y1, x2, y2 = ins.box_xyxy
        rows.append(
            {
                "#": ins.idx,
                "label": ins.label,
                "score": round(ins.score, 4),
                "area_px": ins.area_px,
                "area_%": round(100.0 * ins.area_px / (W * H), 3),
                "box_xyxy": f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]",
            }
        )
    return rows


# ----------------------------
# Model loading (cache)
# ----------------------------
@st.cache_resource
def load_grounding_dino(model_id: str):
    """
    GroundingDINO:
    - AutoProcessor: připraví vstup pro model (image + text)
    - AutoModelForZeroShotObjectDetection: open-vocabulary detekce boxů
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()
    return processor, model


@st.cache_resource
def load_sam2(model_id: str):
    """
    SAM2 (Transformers):
    - Sam2Processor: připraví prompt (box) + obraz pro segmentaci
    - Sam2Model: vrací masky (může vracet více kandidátů + iou_scores)
    """
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id)
    model.eval()
    return processor, model


# ----------------------------
# Core pipeline
# ----------------------------
def parse_labels(query: str) -> List[str]:
    """Uživatel může napsat: 'person' nebo 'person, car'."""
    q = query.strip()
    if not q:
        return []
    parts = [p.strip() for p in q.replace(";", ",").split(",")]
    return [p for p in parts if p]


def run_pipeline(
        pil_img: Image.Image,
        text_query: str,
        box_threshold: float,
        text_threshold: float,
        max_det: int,
        sam_bin_thr: float,
        dino_processor,
        dino_model,
        sam_processor,
        sam_model,
        device: torch.device,
) -> List[Instance]:
    """
    Pipeline (Text → boxy → masky):
    1) GroundingDINO: najde bounding boxy všech objektů odpovídajících textu
    2) SAM2: pro každý box vytvoří masku instance
    """

    labels = parse_labels(text_query)
    if not labels:
        return []

    # 1) GroundingDINO: image + labels -> boxy
    inputs = dino_processor(images=pil_img, text=[labels], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    # postprocess: vrací boxy v pixelech xyxy
    res = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_img.size[::-1]],  # (H, W)
    )[0]

    boxes = res.get("boxes", None)
    scores = res.get("scores", None)
    det_labels = res.get("labels", None)

    if boxes is None or len(boxes) == 0:
        return []

    # Některé verze processors mohou vracet list / numpy / tensor.
    # Pro stabilitu převedeme vše na numpy.
    def to_numpy(x):
        """Převede Tensor / numpy / list na numpy array."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    boxes = to_numpy(boxes)
    scores = to_numpy(scores)
    det_labels = to_numpy(det_labels)

    # kdyby labels chyběly, vytvoříme dummy indexy (pak se label_str jen defaultuje)
    if det_labels is None:
        det_labels = np.zeros(len(scores), dtype=int)

    # omez počet detekcí (kvůli rychlosti a přehlednosti)
    order = np.argsort(-scores)[:max_det]
    boxes = boxes[order]
    scores = scores[order]
    det_labels = det_labels[order]

    # 2) SAM2: box -> maska
    sam_model = sam_model.to(device)
    raw = pil_img.convert("RGB")

    insts: List[Instance] = []
    with torch.no_grad():
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            score = float(scores[i])

            # mapování label indexu na textový label
            label_str = labels[0] if len(labels) == 1 else text_query
            try:
                li = int(det_labels[i])
                if 0 <= li < len(labels):
                    label_str = labels[li]
            except Exception:
                pass

            # SAM2 dostane box jako prompt (v pixelech)
            sam_inputs = sam_processor(
                images=raw,
                input_boxes=[[[x1, y1, x2, y2]]],  # batch=1, num_boxes=1
                return_tensors="pt",
            ).to(device)

            sam_out = sam_model(**sam_inputs)

            # přeškálování masek na původní velikost obrázku
            masks = sam_processor.post_process_masks(
                sam_out.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
            )[0]

            # masks: (1, K, H, W) nebo (K, H, W)
            if masks.ndim == 4:
                masks = masks[0]  # (K, H, W)

            # vyber nejlepší masku (pokud jsou iou_scores)
            best_k = 0
            if getattr(sam_out, "iou_scores", None) is not None:
                ious = sam_out.iou_scores.detach().cpu().numpy()
                ious = ious[0] if ious.ndim == 2 else ious
                best_k = int(np.argmax(ious))

            # binarizace masky
            mask = masks[best_k].numpy()
            mask_bool = mask > sam_bin_thr

            area = int(mask_bool.sum())
            if area == 0:
                continue

            insts.append(
                Instance(
                    idx=0,
                    label=label_str,
                    score=score,
                    box_xyxy=(x1, y1, x2, y2),
                    mask=mask_bool,
                    area_px=area,
                )
            )

    # seřadit a očíslovat
    insts.sort(key=lambda x: x.score, reverse=True)
    for j, ins in enumerate(insts, start=1):
        ins.idx = j
    return insts


# ----------------------------
# Streamlit GUI
# ----------------------------
st.set_page_config(page_title="Grounded SAM2: Text → Segment → Count", layout="wide")
st.title("Objects segmentator: Obrázek + Text → najdi → vysegmentuj → spočítej")

st.write(
    "Nahraj obrázek, napiš co hledáš (např. **person**, **car**, **dog**, **bottle**) "
    "a aplikace najde všechny objekty odpovídající textu, vysegmentuje je a spočítá."
)

with st.sidebar:
    st.header("Modely")

    dino_model_id = st.selectbox(
        "Grounding DINO",
        ["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"],
        index=0,
        help="Tiny je rychlejší, Base často přesnější.",
    )

    sam2_model_id = st.selectbox(
        "SAM2",
        [
            "facebook/sam2.1-hiera-base-plus",
            "facebook/sam2.1-hiera-small",
            "facebook/sam2.1-hiera-tiny",
            "facebook/sam2.1-hiera-large",
        ],
        index=0,
        help="Base-plus je dobrý kompromis. Small/Tiny rychlejší. Large nejkvalitnější, ale těžší.",
    )

    st.divider()
    device_choice = st.radio("Zařízení", ["Auto", "CPU", "GPU (CUDA)"], index=0)

    st.divider()
    st.subheader("Prahy (citlivost)")
    box_threshold = st.slider("Box threshold", 0.05, 0.95, 0.35, 0.05)
    text_threshold = st.slider("Text threshold", 0.05, 0.95, 0.25, 0.05)

    st.subheader("SAM2 maska")
    sam_bin_thr = st.slider("Binarizační práh masky", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    max_det = st.slider("Max. počet instancí", 1, 100, 25, 1)
    alpha = st.slider("Průhlednost overlay", 0.10, 0.90, 0.45, 0.05)
    draw_boxes = st.checkbox("Zobrazit boxy + popisky", value=True)

# --- Vstupy od uživatele (hlavní část UI) ---
uploaded = st.file_uploader("Obrázek (jpg/png)", type=["jpg", "jpeg", "png"])
text_query = st.text_input("Co hledám (text):", value="person")

if uploaded is None:
    st.info("Nahraj obrázek. Tip: běžné scény (lidi/auta/zvířata/předměty) fungují nejlépe.")
    st.stop()

# Načtení obrázku (PIL)
try:
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
except Exception as e:
    st.error(f"Nelze načíst obrázek: {e}")
    st.stop()

# Výběr zařízení (CPU/GPU)
if device_choice == "CPU":
    device = torch.device("cpu")
elif device_choice == "GPU (CUDA)":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Načtení modelů (cache - načítá se jen 1x, pak už je to rychlé)
dino_processor, dino_model = load_grounding_dino(dino_model_id)
sam_processor, sam_model = load_sam2(sam2_model_id)
dino_model = dino_model.to(device)

# ----------------------------
# OVLÁDÁNÍ RERUNŮ: výpočet až po kliknutí na tlačítko
# ----------------------------
# Streamlit defaultně rerunuje skript po každé změně widgetu.
# Aby se inference nepouštěla pořád dokola, držíme poslední výsledek v session_state
# a počítáme jen na explicitní kliknutí.
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None

col_run, col_clear = st.columns([1, 1])
with col_run:
    run_clicked = st.button("▶ Spustit detekci + segmentaci", type="primary")
with col_clear:
    clear_clicked = st.button("🧹 Vymazat výsledek")

if clear_clicked:
    st.session_state.last_result = None
    st.session_state.last_meta = None
    st.rerun()

# Parametry běhu – uložíme pro přehled (co přesně bylo nastavené při aktuálním výpočtu)
current_meta = {
    "text_query": text_query,
    "box_threshold": box_threshold,
    "text_threshold": text_threshold,
    "max_det": max_det,
    "sam_bin_thr": sam_bin_thr,
    "dino_model_id": dino_model_id,
    "sam2_model_id": sam2_model_id,
    "device": str(device),
}

if run_clicked:
    with st.spinner("Běžím: GroundingDINO (boxy) → SAM2 (masky)…"):
        insts = run_pipeline(
            pil_img=pil_img,
            text_query=text_query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_det=max_det,
            sam_bin_thr=sam_bin_thr,
            dino_processor=dino_processor,
            dino_model=dino_model,
            sam_processor=sam_processor,
            sam_model=sam_model,
            device=device,
        )
    st.session_state.last_result = insts
    st.session_state.last_meta = current_meta

# Pokud už máme výsledek z minulého kliknutí, zobrazíme ho.
insts: Optional[List[Instance]] = st.session_state.last_result

if insts is None:
    st.info("Nastav parametry a klikni na **Spustit detekci + segmentaci**.")
    # Volitelně můžeš zobrazit vstup i bez výsledku:
    st.image(pil_img, caption="Vstupní obrázek", use_container_width=True)
    st.stop()

# ----------------------------
# Zobrazení výsledků
# ----------------------------
st.success(f"Nalezeno **{len(insts)}** instancí pro dotaz: **{st.session_state.last_meta['text_query']}**")

# Overlays: instance (po instancích) + union (vše dohromady)
inst_overlay, union_overlay = build_overlays(pil_img, insts, alpha=alpha, draw_boxes=draw_boxes)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.subheader("Vstup")
    st.image(pil_img, use_container_width=True)
with c2:
    st.subheader("Každý objekt zvlášť (instance masky)")
    st.image(inst_overlay, use_container_width=True)
with c3:
    st.subheader("Vše dohromady (souhrnná maska)")
    st.image(union_overlay, use_container_width=True)

# Tabulka + tipy
st.subheader("Detaily detekcí")
if not insts:
    st.write("Nic nenalezeno. Zkus snížit box threshold nebo upravit text (např. 'person' vs 'a person').")
else:
    st.dataframe(table_rows(insts, pil_img.size), use_container_width=True)

# Zobrazit meta parametry posledního běhu
with st.expander("Parametry posledního běhu (pro přehled / debug)"):
    st.json(st.session_state.last_meta)
