import streamlit as st
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import pytesseract
import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
import json
import re

# Page configuration
st.set_page_config(
    page_title="Smart Label Analytics",
    layout="centered"
)


# Load model and processor once — cached so it does not reload on every interaction
@st.cache_resource
def load_model():
    processor = LayoutLMv3Processor.from_pretrained(
        "lavanyakapoor/smart-label-model",
        apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "lavanyakapoor/smart-label-model"
    )
    model.eval()
    return model, processor

@st.cache_resource
def load_labels():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="lavanyakapoor/smart-label-model",
        filename="label_names.json"
    )
    with open(path) as f:
        return json.load(f)

model, processor = load_model()
label_names      = load_labels()


def run_ocr(image):
    # Extract text tokens and their positions from the image using Tesseract
    W, H     = image.size
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    tokens, boxes = [], []
    for i, word in enumerate(ocr_data["text"]):
        word = word.strip()
        if not word:
            continue
        x, y = ocr_data["left"][i], ocr_data["top"][i]
        w, h = ocr_data["width"][i], ocr_data["height"][i]
        tokens.append(word)
        boxes.append([
            int(x/W*1000), int(y/H*1000),
            int((x+w)/W*1000), int((y+h)/H*1000)
        ])
    return tokens, boxes


def extract_nutrients(image, tokens, boxes):
    # Run LayoutLMv3 inference and decode predicted nutrient labels
    encoding = processor(
        images=image, text=tokens, boxes=boxes,
        truncation=True, padding="max_length",
        max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoding)

    predictions    = outputs.logits.argmax(dim=-1).squeeze().tolist()
    tokens_decoded = processor.tokenizer.convert_ids_to_tokens(
        encoding["input_ids"].squeeze().tolist()
    )

    # BIO span decoder — reconstructs full values from subword tokens
    nutrients     = {}
    current_label = None
    current_value = []

    for token_str, pred_id in zip(tokens_decoded, predictions):
        if token_str in ["<s>", "</s>", "<pad>"]:
            continue
        label     = label_names[pred_id]
        clean_tok = token_str.replace("\u2581","").replace("\u0120","").strip()
        if not clean_tok:
            continue
        if label == "O":
            if current_label and current_value:
                nutrients[current_label] = "".join(current_value)
            current_label = None
            current_value = []
        elif label.startswith("B-"):
            nutrient = label[2:]
            if nutrient == current_label:
                current_value.append(clean_tok)
            else:
                if current_label and current_value:
                    nutrients[current_label] = "".join(current_value)
                current_label = nutrient
                current_value = [clean_tok]
        elif label.startswith("I-") and current_label:
            current_value.append(clean_tok)

    if current_label and current_value:
        nutrients[current_label] = "".join(current_value)

    return {k: v.replace("\u010a","").strip() for k, v in nutrients.items()}


def parse_value(v):
    # Parse numeric value from strings like "3,6g" or "365kcal"
    v = v.encode("ascii","ignore").decode()
    v = v.replace(",",".").replace("<","").replace(">","")
    v = re.sub(r"^\.", "0.", v)
    match = re.search(r"[\d.]+", v)
    return float(match.group()) if match else 0.0


def score_product(nutrients):
    # Apply WHO/EU reference thresholds to compute a 0-100 health score
    score = 50

    def get(key):
        for suffix in ["_100G", "_SERVING"]:
            val = nutrients.get(key + suffix)
            if val:
                return parse_value(val)
        return None

    energy  = get("ENERGY_KCAL")
    sat_fat = get("SATURATED_FAT")
    sugars  = get("SUGARS")
    salt    = get("SALT")
    fibre   = get("FIBER")
    protein = get("PROTEINS")

    breakdown = {}

    if energy and energy > 200:
        p = min((energy-200)/200*10, 10)
        score -= p
        breakdown["Energy"]        = (f"{energy:.0f} kcal", f"-{p:.1f}")
    if sat_fat:
        p = min(sat_fat/5*8, 15)
        score -= p
        breakdown["Saturated fat"] = (f"{sat_fat:.1f}g", f"-{p:.1f}")
    if sugars:
        p = min(sugars/12.5*8, 15)
        score -= p
        breakdown["Sugars"]        = (f"{sugars:.1f}g", f"-{p:.1f}")
    if salt:
        p = min(salt/1.5*8, 10)
        score -= p
        breakdown["Salt"]          = (f"{salt:.2f}g", f"-{p:.1f}")
    if fibre:
        b = min(fibre/3*8, 10)
        score += b
        breakdown["Fibre"]         = (f"{fibre:.1f}g", f"+{b:.1f}")
    if protein:
        b = min(protein/10*6, 8)
        score += b
        breakdown["Protein"]       = (f"{protein:.1f}g", f"+{b:.1f}")

    score = max(0, min(100, round(score)))

    if score >= 75:   grade, advice, color = "A", "Excellent nutritional profile.", "#2ecc71"
    elif score >= 60: grade, advice, color = "B", "Good choice. Balanced nutritional profile.", "#a8d96a"
    elif score >= 45: grade, advice, color = "C", "Moderate. Fine occasionally, watch portions.", "#f1c40f"
    elif score >= 30: grade, advice, color = "D", "High in fat/sugar/salt. Limit frequency.", "#e67e22"
    else:             grade, advice, color = "E", "Poor nutritional profile. Consider alternatives.", "#e74c3c"

    return score, grade, advice, color, breakdown


# App layout
st.title("Smart Label Analytics")
st.caption("Upload a food packaging photo to receive an instant nutritional health assessment.")
st.divider()

uploaded = st.file_uploader(
    "Upload a nutrition label image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.divider()

    with st.spinner("Analysing nutrition label..."):
        tokens, boxes = run_ocr(image)

        if not tokens:
            st.error("No text detected. Please upload a clearer image.")
        else:
            nutrients = extract_nutrients(image, tokens, boxes)
            score, grade, advice, color, breakdown = score_product(nutrients)

    st.subheader("Health Assessment")

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{score} / 100")
    col2.metric("Grade", grade)
    col3.metric("Assessment", advice[:10] + "...")

    st.progress(score / 100)
    st.info(advice)
    st.divider()

    if breakdown:
        st.subheader("Nutrient Breakdown (per 100g)")
        cols = st.columns(3)
        for i, (nutrient, (value, impact)) in enumerate(breakdown.items()):
            with cols[i % 3]:
                delta_color = "inverse" if impact.startswith("-") else "normal"
                st.metric(nutrient, value, impact, delta_color=delta_color)

    st.divider()

    with st.expander("All extracted nutrient values"):
        for k, v in sorted(nutrients.items()):
            clean_key = (k.replace("_100G", " (100g)")
                          .replace("_SERVING", " (serving)")
                          .replace("_", " ")
                          .title())
            st.write(f"**{clean_key}** : {v}")
