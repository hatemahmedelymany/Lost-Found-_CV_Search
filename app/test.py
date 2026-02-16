# ============================================================
# LOSTFOUND AI – BACKEND (CARDS SECTION + NO OVERLAP)
# ============================================================

import os
import re
import json
import math
import time
import torch
import pymysql
import numpy as np

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image
from io import BytesIO

import open_clip as clip
from sklearn.cluster import KMeans
import cv2

from langdetect import detect
from deep_translator import GoogleTranslator

import difflib
import hashlib
import urllib.parse


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "lostfound")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


REF_X, REF_Y, REF_Z = 95.047, 100.000, 108.883

TEXT_COLOR_MAP = {
    "red": (220, 20, 60), "green": (34, 139, 34), "blue": (30, 144, 255),
    "black": (0, 0, 0), "white": (245, 245, 245), "brown": (139, 69, 19),
    "yellow": (255, 215, 0), "orange": (255, 140, 0), "pink": (255, 105, 180),
    "purple": (128, 0, 128), "grey": (128, 128, 128), "gray": (128, 128, 128),
}
AR_COLOR_MAP = {
    "اسود": "black", "أسود": "black", "ابيض": "white", "أبيض": "white",
    "احمر": "red", "أحمر": "red", "ازرق": "blue", "أزرق": "blue",
    "اخضر": "green", "أخضر": "green", "بني": "brown", "اصفر": "yellow",
    "أصفر": "yellow", "برتقالي": "orange", "زهري": "pink", "وردي": "pink",
}

# Thresholds
MIN_IMAGE_SIM_THRESHOLD = 0.60
MIN_TEXT_SIM_THRESHOLD = 0.50

TEXT_SEARCH_STATUSES = ("lost", "found")
IMAGE_SEARCH_STATUSES = ("lost", "found")

FUZZY_CUTOFF_AR = 0.82
FUZZY_CUTOFF_EN = 0.85

EXTRA_TERM_MIN_LEN = 3
EN_STOPWORDS = {"with", "and", "or", "the", "a", "an", "of", "to", "in", "on", "for"}
AR_STOPWORDS = {"مع", "و", "او", "في", "على", "من", "الى", "إلى", "عن", "ده", "دي", "دا", "هذا", "هذه"}

# Generic terms (avoid classifying on them)
GENERIC_TERMS_AR = {
    "شنطه", "شنطة", "حقيبه", "حقيبة", "كيس",
    "بطاقه", "بطاقة", "كارت",
    "ورق", "اوراق", "أوراق", "مستندات", "مستند",
    "حاجه", "حاجة", "شيء", "حاجات", "اغراض", "أغراض",
    "لقيت", "ضاع", "مفقود", "موجود"
}
GENERIC_TERMS_EN = {
    "bag", "card", "paper", "papers", "document", "documents",
    "thing", "item", "found", "lost"
}

# ============================================================
# INITIALIZATION
# ============================================================

app = FastAPI(title="Lost & Found AI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

try:
    connection = pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME,
        port=DB_PORT, autocommit=True, cursorclass=pymysql.cursors.DictCursor,
    )
    print("✅ DB connected")
except Exception as e:
    print(f"❌ DB FAILED: {e}")
    connection = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)

print(f"✅ CLIP loaded on {device}")

# ============================================================
# ARABIC NORMALIZATION + TOKENIZATION
# ============================================================

AR_RE = re.compile(r"[\u0600-\u06FF]")

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = text.replace("ـ", "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    return text

def strip_punct(text: str) -> str:
    return re.sub(r"[^\w\u0600-\u06FF]+", " ", (text or "")).strip()

def tokenize_mixed(text: str):
    text = strip_punct((text or "").lower())
    text = normalize_arabic(text)
    return [t for t in text.split() if t]

def is_arabic_token(token: str) -> bool:
    return AR_RE.search(token or "") is not None

def norm_kw(s: str) -> str:
    s = (s or "").strip().lower()
    s = normalize_arabic(s)
    s = strip_punct(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_generic_token(token: str) -> bool:
    token = norm_kw(token)
    if not token:
        return True
    if is_arabic_token(token):
        return token in {norm_kw(x) for x in GENERIC_TERMS_AR}
    return token in GENERIC_TERMS_EN

# ============================================================
# CATEGORY MAP (EXPANDED TO MATCH FLUTTER)
# ============================================================

CATEGORY_MAP = {
    "Cards": {
        "id card": [
            "بطاقة شخصية", "رقم قومي", "هوية", "اثبات شخصية", "كارنيه",
            "id card", "national id", "identity card"
        ],
        "license card": [
            "رخصة", "رخصة قيادة", "رخصة سواقة",
            "driver license", "driving license", "license card"
        ],
        "bank card": [
            "فيزا", "ماستر", "كارت بنك", "بطاقة بنكية", "بطاقة ائتمان", "كريدت", "ديبت",
            "credit card", "debit card", "bank card", "visa", "mastercard"
        ],
        "student card": [
            "كارنيه جامعة", "كارنيه طالب", "بطاقة طالب",
            "student card", "student id", "university id"
        ],
        "work id card": [
            "كارنيه شغل", "بطاقة عمل", "بطاقة موظف",
            "work id card", "employee id", "company id"
        ],
        "access card": [
            "كارت دخول", "كارت بوابة", "بطاقة دخول",
            "access card", "entry card", "gate card"
        ],
    },

    "Documents": {
        "passport": ["جواز سفر", "باسبور", "جواز", "passport"],
        "papers": ["اوراق رسمية", "مستند", "مستندات", "وثائق", "document", "documents", "papers"],
        "file": ["ملف", "دوسيه", "حافظة", "folder", "file", "document folder"],
        "certificate": ["شهادة", "certificate", "diploma"],
    },

    "Bags & Luggage": {
        "handbag": ["شنطة يد", "حقيبة يد", "شنطة حريمي", "handbag", "purse"],
        "backpack": ["شنطة ظهر", "حقيبة ظهر", "باك باك", "backpack", "rucksack"],
        "suitcase": ["شنطة سفر", "حقيبة سفر", "suitcase", "luggage"],
        "trolley bag": ["شنطة بعجل", "شنطة ترولي", "trolley bag", "rolling bag", "wheel bag"],
        "briefcase": ["شنطة اوراق", "حقيبة مستندات", "شنطة مكتب", "briefcase"],
        "laptop bag": ["شنطة لاب توب", "حقيبة لابتوب", "laptop bag", "computer bag"],
        "school bag": ["شنطة مدرسة", "شنطة مدرسية", "school bag", "school backpack"],
        "shopping bag": ["شنطة تسوق", "كيس تسوق", "shopping bag"],
        "gaming bag": ["شنطة جيمينج", "gaming bag", "console bag"],
        "baby bag": ["شنطة اطفال", "حقيبة مستلزمات اطفال", "baby bag", "diaper bag"],
    },

    "Electronics": {
        "phone": ["موبايل", "هاتف", "تليفون", "ايفون", "سامسونج", "اوبو", "شاومي",
                  "phone", "mobile", "iphone", "samsung", "oppo", "xiaomi"],
        "tablet": ["تابلت", "ايباد", "tablet", "ipad"],
        "laptop": ["لاب توب", "كمبيوتر", "حاسوب", "ماك بوك", "لينوفو", "ديل",
                   "laptop", "macbook", "lenovo", "dell"],
        "camera": ["كاميرا", "camera", "dslr", "canon", "nikon"],
        "headphones": ["سماعة", "سماعات", "ايربودز", "headphones", "earbuds", "airpods"],
        "bluetooth speaker": ["سبيكر", "سماعة بلوتوث", "bluetooth speaker", "speaker"],
        "smart watch": ["ساعة ذكية", "سمارت واتش", "ساعة ابل", "smart watch", "apple watch"],
        "router": ["راوتر", "واي فاي", "router", "wifi router"],
        "keyboard": ["كيبورد", "لوحة مفاتيح", "keyboard"],
        "mouse": ["ماوس", "فارة", "mouse"],
        "printer": ["طابعة", "printer"],
        "tablet/monitor": ["شاشة", "monitor", "screen", "display", "tv"],
        "charger cable": ["شاحن", "كابل", "وصلة", "سلك شاحن", "charger", "cable", "charging cable", "adapter"],
        "power bank": ["باور بانك", "شاحن متنقل", "power bank", "powerbank"],
        "usb drive": ["فلاشة", "يو اس بي", "usb drive", "flash drive", "thumb drive"],
        "sd card": ["كارت ميموري", "ميموري كارد", "sd card", "memory card"],
    },

    "Personal Items": {
        # ملاحظة: شيلنا "مفتاح بيت / house key" من هنا علشان نضيف "home key" في Home Items
        "keys": ["مفاتيح", "مفتاح", "key", "keys"],
        "remote": ["ريموت", "ريموت عربية", "key fob", "car remote", "remote"],
        "wallet": ["محفظة", "بوك", "wallet"],
        "sunglasses": ["نظارة شمسية", "نظارة", "sunglasses", "glasses"],
        "umbrella": ["شمسية", "مظلة", "umbrella"],
        "medical kit": ["شنطة اسعافات", "اسعافات اولية", "medical kit", "first aid kit"],
        "medicine": ["دواء", "علاج", "برشام", "medicine", "pills"],
        "cigarettes": ["سجاير", "دخان", "cigarettes", "smokes"],
    },

    "Jewelry & Accessories": {
        "ring": ["خاتم", "دبلة", "ring"],
        "watch": ["ساعة", "watch"],
        "bracelet": ["اسورة", "سوار", "انسيال", "bracelet", "bangle"],
        "necklace": ["عقد", "سلسلة", "قلادة", "necklace"],
        "earrings": ["حلق", "earrings"],
        "brooch": ["بروش", "brooch", "pin"],
        "cufflinks": ["كافلينكس", "أزرار كم", "cufflinks"],
    },

    "Clothes": {
        "t-shirt": ["تيشيرت", "قميص", "t-shirt", "tshirt", "shirt"],
        "jacket": ["جاكيت", "جاكت", "jacket", "coat"],
        "hoodie": ["هودي", "سويت شيرت", "hoodie", "sweatshirt"],
        "shoes": ["حذاء", "شوز", "كوتشي", "جزمة", "shoes", "sneakers"],
        "hat": ["قبعة", "طاقية", "كاب", "hat", "cap"],
        "scarf": ["وشاح", "كوفية", "سكارف", "شال", "scarf"],
        "gloves": ["جوانتي", "قفازات", "gloves"],
    },

    "Money": {
        "cash": ["فلوس", "نقد", "كاش", "cash", "money"],
        "coins": ["عملات", "فكة", "coins", "change"],
        "money pouch": ["جراب فلوس", "coin pouch", "money pouch", "coin purse"],
    },

    "Vehicles": {
        "car": ["سيارة", "عربية", "car", "auto"],
        "bike": ["عجلة", "دراجة", "bike", "bicycle"],
        "scooter": ["سكوتر", "scooter"],
        "motorcycle": ["موتوسيكل", "دراجة نارية", "motorcycle", "motorbike"],
        "boat": ["مركب", "boat"],
        "airplane item": ["طيران", "airplane item", "plane item"],
    },

    "Sports Equipment": {
        "football": ["كورة", "كرة قدم", "football", "soccer ball"],
        "basketball": ["كرة سلة", "basketball"],
        "tennis racket": ["مضرب تنس", "tennis racket"],
        "golf club": ["عصاية جولف", "golf club"],
        "boxing gloves": ["جوانتي ملاكمة", "boxing gloves"],
    },

    "Kids Items": {
        "toy": ["لعبة", "دمية", "عروسة", "toy", "doll"],
        "baby bag": ["شنطة اطفال", "حقيبة اطفال", "baby bag", "diaper bag"],
        "stroller": ["عربية اطفال", "عربية بيبي", "stroller", "pram"],
        "kids tablet": ["تابلت اطفال", "kids tablet", "children tablet"],
    },

    "Pets": {
        "dog": ["كلب", "dog"],
        "cat": ["قطة", "cat"],
        "bird": ["عصفور", "طائر", "bird"],
        "other pet": ["حيوان", "حيوان اليف", "other pet", "pet"],
    },

    "Home Items": {
        "home key": ["مفتاح بيت", "مفتاح شقة", "house key", "home key", "apartment key"],
        "lamp": ["لمبة", "مصباح", "lamp"],
        "kitchen tool": ["اداة مطبخ", "ادوات مطبخ", "kitchen tool", "kitchen utensil"],
        "cleaning item": ["منظف", "مستلزمات تنظيف", "cleaning item", "cleaning supplies"],
    },

    "Stationery": {
        "pen": ["قلم", "قلم حبر", "pen"],
        "pencil": ["قلم رصاص", "pencil"],
        "marker": ["ماركر", "قلم فلوماستر", "marker"],
        "notebook": ["كراسة", "كشكول", "دفتر", "notebook"],
        "diary": ["مذكرة", "يوميات", "diary", "journal"],
        "book": ["كتاب", "رواية", "book"],
        "paper file": ["ملف ورق", "حافظة ورق", "paper file", "paper holder"],
        "note paper": ["ورق", "ملاحظات", "note paper", "sticky notes"],
    },

    "Tools": {
        "toolbox": ["شنطة عدة", "صندوق عدة", "toolbox"],
        "hand tool": ["مفك", "مطرقة", "كماشة", "hand tool", "screwdriver", "hammer", "pliers"],
        "plumbing tool": ["ادوات سباكة", "plumbing tool", "wrench", "pipe wrench"],
        "electric tool": ["ادوات كهرباء", "electric tool", "drill", "power tool"],
    },

    "Others": {
        "other": ["اخرى", "اخر", "شيء اخر", "other"],
        "unknown item": ["مجهول", "unknown", "unknown item"],
    },
}



# ============================================================
# BUILD KEYWORD MAPS + VALIDATION
# ============================================================

def validate_no_keyword_overlap(category_map):
    seen = {}
    overlaps = []

    for section, subs in category_map.items():
        for subcat, keywords in subs.items():
            for kw in [subcat] + keywords:
                k = norm_kw(kw)
                if not k:
                    continue
                if k in seen and seen[k] != subcat:
                    overlaps.append((k, seen[k], subcat))
                else:
                    seen[k] = subcat

    if overlaps:
        msg = "\n".join([f"Keyword '{k}' mapped to BOTH '{a}' and '{b}'" for k, a, b in overlaps[:80]])
        raise ValueError("❌ Overlapping keywords detected:\n" + msg)

validate_no_keyword_overlap(CATEGORY_MAP)

KEYWORD_TO_CATEGORY = {}
CATEGORY_TO_SECTION = {}

for section, sub_categories in CATEGORY_MAP.items():
    for sub_cat, keywords in sub_categories.items():
        CATEGORY_TO_SECTION[sub_cat] = section
        for keyword in [sub_cat] + keywords:
            k = norm_kw(keyword)
            if not k:
                continue
            KEYWORD_TO_CATEGORY[k] = sub_cat

print("✅ Category mappings generated successfully (Cards separated).")

def expand_category_scope(scope_list):
    """Expand scope to match BOTH storage styles:
    - New items: category stored as SECTION (e.g., 'Cards')
    - Old items: category stored as SUBCATEGORY (e.g., 'id card')

    For each input token, we include:
      - the token itself
      - if token is a section: all its subcategories
      - if token is a subcategory: its parent section
    """
    if not scope_list:
        return []

    out = set()
    for x in scope_list:
        x = (x or "").strip()
        if not x:
            continue

        if x in CATEGORY_MAP:  # section
            out.add(x)
            for subcat in CATEGORY_MAP[x].keys():
                out.add(subcat)
            continue

        # subcategory or unknown
        out.add(x)
        sec = CATEGORY_TO_SECTION.get(x)
        if sec:
            out.add(sec)

    return list(out)


AR_KEYWORDS, EN_KEYWORDS = [], []
for kw in KEYWORD_TO_CATEGORY.keys():
    (AR_KEYWORDS if AR_RE.search(kw) else EN_KEYWORDS).append(kw)

# ============================================================
# FUZZY MATCH
# ============================================================

def fuzzy_match_token(token: str):
    token = norm_kw(token)
    if not token or len(token) < 2:
        return None
    if is_generic_token(token):
        return None

    if token in KEYWORD_TO_CATEGORY:
        return token

    if is_arabic_token(token):
        matches = difflib.get_close_matches(token, AR_KEYWORDS, n=1, cutoff=FUZZY_CUTOFF_AR)
        return matches[0] if matches else None
    else:
        matches = difflib.get_close_matches(token, EN_KEYWORDS, n=1, cutoff=FUZZY_CUTOFF_EN)
        return matches[0] if matches else None

def infer_detected_subcats_and_terms(query: str):
    tokens = tokenize_mixed(query)

    detected_subcats, matched_keywords, extra_terms = [], [], []

    for t in tokens:
        if is_generic_token(t):
            continue

        kw = fuzzy_match_token(t)
        if kw:
            subcat = KEYWORD_TO_CATEGORY.get(kw)
            if subcat:
                detected_subcats.append(subcat)
                matched_keywords.append(kw)
            continue

        if len(t) >= EXTRA_TERM_MIN_LEN:
            if is_arabic_token(t):
                if t not in {norm_kw(x) for x in AR_STOPWORDS}:
                    extra_terms.append(t)
            else:
                if t not in EN_STOPWORDS:
                    extra_terms.append(t)

    detected_subcats = list(dict.fromkeys(detected_subcats))
    matched_keywords = list(dict.fromkeys(matched_keywords))
    extra_terms = list(dict.fromkeys(extra_terms))

    sections = []
    for sc in detected_subcats:
        sec = CATEGORY_TO_SECTION.get(sc)
        if sec and sec not in sections:
            sections.append(sec)

    print(f"[DETECT] query='{query}' tokens={tokens} -> subcats={detected_subcats} sections={sections} extra={extra_terms} matched_kw={matched_keywords}")
    return detected_subcats, matched_keywords, extra_terms, sections

def subcats_for_sections(sections):
    return [subcat for subcat, sec in CATEGORY_TO_SECTION.items() if sec in sections]

# ============================================================
# CLIP EMBEDDINGS
# ============================================================

def encode_image(img: Image.Image):
    img = img.resize((224, 224))
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten().tolist()


def _center_crop_pil(img: Image.Image, scale: float = 0.75) -> Image.Image:
    """Center-crop to reduce background / folio cover dominance."""
    if img is None:
        return img
    w, h = img.size
    scale = max(0.2, min(1.0, float(scale)))
    nw, nh = int(w * scale), int(h * scale)
    left = max(0, (w - nw) // 2)
    top = max(0, (h - nh) // 2)
    return img.crop((left, top, left + nw, top + nh))


def encode_image_multi(img: Image.Image):
    """CLIP image embedding using full + center crop (helps with screen-off reflections/folio cases)."""
    try:
        embs = []
        embs.append(_np_emb(encode_image(img)))
        embs.append(_np_emb(encode_image(_center_crop_pil(img, 0.75))))
        emb = np.mean(embs, axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb.tolist()
    except Exception:
        return encode_image(img)

def encode_text(text: str):
    text = (text or "").strip()
    if not text:
        return [0.0] * 512
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text_input)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten().tolist()

def normalize_and_translate(text: str):
    text = (text or "").strip()
    if not text:
        return "", "unknown"

    if re.search(r"[\u0600-\u06FF]", text):
        return normalize_arabic(text).lower(), "ar"

    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"

    if lang != "en":
        try:
            translated = GoogleTranslator(source=lang, target="en").translate(text)
            return (translated or text).lower(), lang
        except Exception:
            return text.lower(), lang

    return text.lower(), lang

# ============================================================
# COLOR + SHAPE
# ============================================================

def extract_dominant_color(image: Image.Image, k: int = 4):
    image = image.resize((150, 150))
    data = np.array(image.convert("RGB"))
    pixels = data.reshape(-1, 3)
    mask = np.sum(pixels, axis=1) < 700
    pixels = pixels[mask]
    if len(pixels) < k:
        return (200, 200, 200)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return tuple(int(c) for c in dominant_color)

def extract_color_from_text(text: str):
    clean_text = (text or "").lower().replace(",", " ").replace(".", " ")
    words = clean_text.split()
    for w in words:
        if w in TEXT_COLOR_MAP:
            return list(TEXT_COLOR_MAP[w])
        if w in AR_COLOR_MAP:
            eng_color = AR_COLOR_MAP[w]
            if eng_color in TEXT_COLOR_MAP:
                return list(TEXT_COLOR_MAP[eng_color])
    return None

def extract_shape_vector(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(th)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.tolist()

def ciede2000(Lab_1, Lab_2):
    L1, a1, b1 = Lab_1
    L2, a2, b2 = Lab_2
    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = math.degrees(math.atan2(b1, a1p))
    h1p += 360 if h1p < 0 else 0
    h2p = math.degrees(math.atan2(b2, a2p))
    h2p += 360 if h2p < 0 else 0
    avg_hp = (h1p + h2p + 360) / 2.0 if abs(h1p - h2p) > 180 else (h1p + h2p) / 2.0
    T = (1 - 0.17 * math.cos(math.radians(avg_hp - 30)) + 0.24 * math.cos(math.radians(2 * avg_hp))
         + 0.32 * math.cos(math.radians(3 * avg_hp + 6)) - 0.20 * math.cos(math.radians(4 * avg_hp - 63)))
    dhp = h2p - h1p
    dhp -= 360 if dhp > 180 else (-360 if dhp < -180 else 0)
    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))
    Sl = 1 + (0.015 * (avg_L - 50)**2) / math.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    delta_ro = 30 * math.exp(-(((avg_hp - 275) / 25)**2))
    Rc = 2 * math.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
    Rt = -Rc * math.sin(math.radians(2 * delta_ro))
    dE = math.sqrt((dLp / Sl)**2 + (dCp / Sc)**2 + (dHp / Sh)**2 + Rt * (dCp / Sc) * (dHp / Sh))
    return dE

def rgb_to_lab(rgb):
    r, g, b = [c/255.0 for c in rgb]
    r, g, b = [c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4 for c in (r,g,b)]
    x = r*0.4124 + g*0.3576 + b*0.1805
    y = r*0.2126 + g*0.7152 + b*0.0722
    z = r*0.0193 + g*0.1192 + b*0.9505
    x, y, z = x*100, y*100, z*100
    xr, yr, zr = x/REF_X, y/REF_Y, z/REF_Z
    fx, fy, fz = [t**(1/3) if t > (6/29)**3 else t/(3*(6/29)**2)+4/29 for t in (xr,yr,zr)]
    L = 116*fy - 16
    a = 500*(fx-fy)
    b2 = 200*(fy-fz)
    return (L, a, b2)

def color_similarity_lab(color_a_rgb, color_b_rgb):
    if color_a_rgb is None or color_b_rgb is None:
        return 0.5
    lab1 = rgb_to_lab(color_a_rgb)
    lab2 = rgb_to_lab(color_b_rgb)
    delta_e = ciede2000(lab1, lab2)
    return max(0.0, 1.0 - (delta_e / 100.0))

def shape_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.5
    dist = np.linalg.norm(np.array(v1, dtype=float) - np.array(v2, dtype=float))
    return 1.0 / (1.0 + dist)

def safe_rgb(r, g, b):
    if r is None or g is None or b is None:
        return None
    return (float(r), float(g), float(b))

# ============================================================
# TEXT SEARCH
# ============================================================

def build_embedding_query(core_labels: str, translated_en: str, raw_query: str, matched_keywords: list):
    parts = []
    if core_labels:
        parts.append(f"a photo of {core_labels}")
    if matched_keywords:
        parts.append(" ".join(matched_keywords))
    if translated_en:
        parts.append(translated_en)
    if raw_query:
        parts.append(raw_query.lower())
    return ", ".join([p for p in parts if p]).strip()

def run_text_search(categories_scope, embedding_query, extra_terms, top_k, original_query_for_color):
    categories_scope = expand_category_scope(categories_scope)
    q_emb = np.array(encode_text(embedding_query), dtype=np.float32)
    q_color = extract_color_from_text(original_query_for_color)

    w_text, w_color = (0.80, 0.20) if q_color else (0.95, 0.05)

    sql_params = []
    cat_placeholders = ", ".join(["%s"] * len(categories_scope))
    sql_params.extend(categories_scope)

    st_placeholders = ", ".join(["%s"] * len(TEXT_SEARCH_STATUSES))
    sql_params.extend(TEXT_SEARCH_STATUSES)

    extra_filter_sql = ""
    for term in extra_terms:
        extra_filter_sql += " AND (LOWER(i.title) LIKE %s OR LOWER(i.description) LIKE %s)"
        like = f"%{term.lower()}%"
        sql_params.extend([like, like])

    sql_query = f"""
        SELECT
            i.id, i.title, i.description, i.category, i.location, i.image_url, i.date_reported, i.status,
            t.embedding AS text_embedding,
            c.color_r, c.color_g, c.color_b
        FROM items i
        JOIN item_embeddings t
            ON i.id = t.item_id AND t.embedding_model = 'clip_text'
        LEFT JOIN item_embeddings c
            ON i.id = c.item_id AND c.embedding_model = 'clip_image'
        WHERE i.matched = 0
          AND i.category IN ({cat_placeholders})
          AND i.status IN ({st_placeholders})
          {extra_filter_sql}
    """

    with connection.cursor() as cursor:
        cursor.execute(sql_query, sql_params)
        rows = cursor.fetchall()

    results = []
    for row in rows:
        try:
            emb_text = np.array(json.loads(row["text_embedding"]), dtype=np.float32)
        except Exception:
            continue

        sim_text = float(np.clip(np.dot(q_emb, emb_text), 0.0, 1.0))
        if sim_text < MIN_TEXT_SIM_THRESHOLD:
            continue

        db_color = safe_rgb(row.get("color_r"), row.get("color_g"), row.get("color_b"))
        sim_color = color_similarity_lab(q_color, db_color)

        final_sim = (w_text * sim_text) + (w_color * sim_color)
        row["similarity"] = float(final_sim)
        results.append(row)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# ============================================================
# IMAGE SEARCH – BETTER FOR CARDS
# ============================================================

SUBCAT_TEXT_EMBEDS = {}
SECTION_TEXT_EMBEDS = {}

def _np_emb(x):
    return np.array(x, dtype=np.float32)

SUBCAT_PROMPTS = {
    "id card": [
        "a photo of a national id card",
        "an identity card with portrait photo and name",
        "government issued id card"
    ],
    "license card": [
        "a photo of a driver license card",
        "driving license with portrait photo",
        "driver's licence card"
    ],
    "bank card": [
        "a photo of a credit card",
        "a debit card with numbers",
        "bank card"
    ],
    "student card": [
        "a student id card",
        "university id card"
    ],
    "work id card": [
        "employee id card",
        "company id badge"
    ],
    "access card": [
        "access card",
        "entry card"
    ],
    "passport": [
        "a photo of a passport booklet",
        "passport photo page"
    ],
    
    # ✅ Electronics (important for tablets / screens-off / folio cases)
    "tablet": [
        "a photo of a tablet computer",
        "a photo of an iPad tablet",
        "a photo of an android tablet",
        "a photo of a tablet with a black screen",
        "a photo of a tablet in a folio case",
        "a photo of a tablet in a leather cover case",
        "a photo of a touchscreen tablet device",
        "a photo of a tablet with front camera on the top bezel",
    ],
}

SUBCAT_PROMPTS.update({
    "cash": ["cash money", "banknotes", "paper money"],
    "coins": ["coins", "metal coins"],
    "car": ["a photo of a car", "car vehicle"],
    "bike": ["a bicycle", "bike vehicle"],
    "scooter": ["an electric scooter", "scooter vehicle"],
    "motorcycle": ["a motorcycle", "motorbike"],
    "dog": ["a dog pet", "a photo of a dog"],
    "cat": ["a cat pet", "a photo of a cat"],
    "toolbox": ["a toolbox", "tools in a box"],
    "kitchen tool": ["kitchen utensil", "kitchen tool"],
    "trolley bag": ["a rolling suitcase", "trolley bag with wheels"],
    "bluetooth speaker": ["bluetooth speaker", "portable speaker"],
    "camera": ["a camera", "dslr camera"],
})


# Better section prompts ("a photo of Cards" is too generic)
SECTION_PROMPTS = {
    "Cards": [
        "a photo of an id card",
        "a photo of a bank card or credit card",
        "a photo of a driver license card",
        "a photo of a student id card",
    ],
    "Documents": [
        "a photo of a passport",
        "a photo of documents or papers",
        "a photo of a book or notebook",
        "a photo of a certificate",
    ],
    "Electronics": [
        "a photo of an electronic device",
        "a photo of a phone or tablet",
        "a photo of a laptop or tablet",
        "a photo of a tablet device",
    ],
    "Personal Items": [
        "a photo of keys",
        "a photo of a wallet",
        "a photo of sunglasses",
        "a photo of a pen",
    ],
    "Bags & Luggage": [
        "a photo of a backpack",
        "a photo of a suitcase",
        "a photo of a handbag",
    ],
    "Jewelry & Accessories": [
        "a photo of a watch",
        "a photo of a ring",
        "a photo of a necklace",
    ],
    "Clothes": [
        "a photo of shoes",
        "a photo of a jacket",
        "a photo of a hat",
    ],
    "Kids Items": [
        "a photo of a stroller",
        "a photo of a toy",
    ],
    "Others": [
        "a photo of a random object",
        "a photo of an unknown item",
    ],
}

def build_category_text_embeddings():
    global SUBCAT_TEXT_EMBEDS, SECTION_TEXT_EMBEDS
    SUBCAT_TEXT_EMBEDS = {}
    SECTION_TEXT_EMBEDS = {}

    for subcat in CATEGORY_TO_SECTION.keys():
        prompts = SUBCAT_PROMPTS.get(subcat, [f"a photo of {subcat}"])
        embs = [_np_emb(encode_text(p)) for p in prompts]
        emb = np.mean(embs, axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        SUBCAT_TEXT_EMBEDS[subcat] = emb

    for section in CATEGORY_MAP.keys():
        prompts = SECTION_PROMPTS.get(section, [f"a photo of {section}"])
        embs = [_np_emb(encode_text(p)) for p in prompts]
        emb = np.mean(embs, axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        SECTION_TEXT_EMBEDS[section] = emb

    print("✅ Precomputed text embeddings for subcategories & sections.")

build_category_text_embeddings()

def infer_best_section_from_image(q_img_emb: np.ndarray, top_n: int = 2):
    scores = []
    for section, t_emb in SECTION_TEXT_EMBEDS.items():
        sim = float(np.clip(np.dot(q_img_emb, t_emb), 0.0, 1.0))
        scores.append((section, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def infer_best_subcat_from_image(q_img_emb: np.ndarray, top_n: int = 10, allowed_subcats=None):
    scores = []
    for subcat, t_emb in SUBCAT_TEXT_EMBEDS.items():
        if allowed_subcats is not None and subcat not in allowed_subcats:
            continue
        sim = float(np.clip(np.dot(q_img_emb, t_emb), 0.0, 1.0))
        scores.append((subcat, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def run_image_search(categories_scope, q_embs, q_color, q_shape, top_k):
    if categories_scope:
        categories_scope = expand_category_scope(categories_scope)
    """Search DB images within category scope.
    categories_scope: list[str] OR None (None = search across ALL categories)
    q_embs: np.ndarray or list[np.ndarray] (we'll take the max CLIP similarity across them)
    """
    # Accept a single embedding or a list of embeddings
    if isinstance(q_embs, np.ndarray):
        q_list = [q_embs]
    else:
        q_list = list(q_embs) if q_embs is not None else []

    # Dynamic weights (keep as-is)
    try:
        lab = rgb_to_lab(q_color)
        chroma = math.sqrt(lab[1]**2 + lab[2]**2)
        t = max(0.0, min(1.0, chroma / 70.0))
        w_color = 0.15 + 0.20 * t
        w_shape = 0.10 + 0.05 * t
        w_clip = 1.0 - w_color - w_shape
    except Exception:
        w_clip, w_color, w_shape = 0.85, 0.10, 0.05

    st_placeholders = ", ".join(["%s"] * len(IMAGE_SEARCH_STATUSES))

    sql_params = []
    where_clauses = [
        "e.embedding_model = 'clip_image'",
        "i.matched = 0",
        f"i.status IN ({st_placeholders})"
    ]

    # Category filter (optional)
    if categories_scope:
        cat_placeholders = ", ".join(["%s"] * len(categories_scope))
        where_clauses.append(f"i.category IN ({cat_placeholders})")
        # NOTE: category placeholders come AFTER statuses in the WHERE string, so we must append in same order.
        # We'll build query accordingly (category params first, then statuses) by placing category clause before status if needed.
        # Easiest: rebuild where & params with correct order:
        where_clauses = [
            "e.embedding_model = 'clip_image'",
            "i.matched = 0",
            f"i.category IN ({cat_placeholders})",
            f"i.status IN ({st_placeholders})"
        ]
        sql_params.extend(categories_scope)
        sql_params.extend(IMAGE_SEARCH_STATUSES)
    else:
        sql_params.extend(IMAGE_SEARCH_STATUSES)

    sql_query = f"""
        SELECT
            i.id, i.title, i.description, i.category, i.location, i.image_url, i.date_reported, i.status,
            e.embedding, e.color_r, e.color_g, e.color_b, e.shape_vector
        FROM items i
        JOIN item_embeddings e ON i.id = e.item_id
        WHERE {' AND '.join(where_clauses)}
    """

    with connection.cursor() as cursor:
        cursor.execute(sql_query, sql_params)
        rows = cursor.fetchall()

    results = []
    for row in rows:
        try:
            db_emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
        except Exception:
            continue

        # Take the best similarity across our query embeddings (full vs multi-crop)
        if q_list:
            sim_clip = max(float(np.clip(np.dot(q, db_emb), 0.0, 1.0)) for q in q_list)
        else:
            sim_clip = 0.0

        db_color = safe_rgb(row.get("color_r"), row.get("color_g"), row.get("color_b"))
        sim_color = color_similarity_lab(q_color, db_color)
        db_shape = json.loads(row["shape_vector"]) if row.get("shape_vector") else None
        sim_shape = shape_similarity(q_shape, db_shape)

        final_sim = (w_clip * sim_clip) + (w_color * sim_color) + (w_shape * sim_shape)

        if final_sim < MIN_IMAGE_SIM_THRESHOLD:
            continue

        row["similarity"] = float(final_sim)
        results.append(row)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "db_connected": connection is not None}

@app.post("/api/items/add")
async def add_item(
    title: str = Form(...),
    description: str = Form(""),
    category: str = Form(...),
    location: str = Form(...),
    status: str = Form(...),
    image: UploadFile = None
):
    if not connection:
        return {"error": "DB Not Connected"}

    img_url, file_path = "", None
    if image:
        fname = f"{int(time.time() * 1000)}_{image.filename.replace(' ', '_')}"
        file_path = os.path.join(UPLOAD_DIR, fname)
        with open(file_path, "wb") as f:
            f.write(await image.read())
        img_url = f"{BASE_URL}/uploads/{fname}"

    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO items(title, description, category, location, status, image_url) VALUES (%s, %s, %s, %s, %s, %s)",
            (title, description, category, location, status, img_url)
        )
        item_id = cursor.lastrowid

        if file_path:
            try:
                img = Image.open(file_path).convert("RGB")
                emb_img = encode_image(img)
                color_r, color_g, color_b = extract_dominant_color(img)
                shape_vec = extract_shape_vector(img)
                cursor.execute(
                    "INSERT INTO item_embeddings (item_id, embedding, embedding_model, color_r, color_g, color_b, shape_vector) VALUES (%s, %s, 'clip_image', %s, %s, %s, %s)",
                    (item_id, json.dumps(emb_img), color_r, color_g, color_b, json.dumps(shape_vec))
                )
            except Exception as e:
                print(f"Error processing image for item {item_id}: {e}")

        full_text = f"{title} | {description} | {category}"
        emb_text = encode_text(full_text)
        cursor.execute(
            "INSERT INTO item_embeddings(item_id, embedding, embedding_model) VALUES (%s, %s, 'clip_text')",
            (item_id, json.dumps(emb_text))
        )

    connection.commit()
    return {"message": "Item added successfully", "item_id": item_id, "image_url": img_url}

@app.get("/api/items/list")
def list_items():
    if not connection:
        return []
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, title, description, category, location, image_url, date_reported, status, matched FROM items ORDER BY date_reported DESC")
        return cursor.fetchall()

@app.post("/api/items/mark_matched")
def mark_matched(item_id: int = Form(...)):
    if not connection:
        return {"error": "DB Not Connected"}
    with connection.cursor() as cursor:
        cursor.execute("UPDATE items SET matched = 1 WHERE id = %s", (item_id,))
    connection.commit()
    return {"success": True, "message": "Item marked as matched"}

# ============================================================
# TEXT SEARCH ENDPOINT (TWO STAGE)
# ============================================================

@app.post("/api/search/text")
async def search_text(query: str = Form(...), top_k: int = 10, section: str = Form(None)):
    if not connection:
        return {"error": "DB Not Connected"}

    detected_subcats, matched_keywords, extra_terms, sections = infer_detected_subcats_and_terms(query)

    # ✅ If user selected a MAIN SECTION in the frontend, search ONLY inside it
    if section:
        section = section.strip()
        if section in CATEGORY_MAP:
            translated_en, lang = normalize_and_translate(query)
            emb_q = build_embedding_query(section, translated_en, query, matched_keywords)
            results = run_text_search(
                categories_scope=[section],
                embedding_query=emb_q,
                extra_terms=extra_terms,
                top_k=top_k,
                original_query_for_color=query,
            )
            return {"results": results, "scope": "section_selected", "section": section}

    if not detected_subcats:
        return {"results": [], "scope": "none", "hint": "اكتب كلمة محددة مثل: 'بطاقة', 'فيزا', 'رخصة', 'id'"}

    translated_en, lang = normalize_and_translate(query)

    # STEP 1
    core_part_1 = ", ".join(detected_subcats)
    emb_q1 = build_embedding_query(core_part_1, translated_en, query, matched_keywords)

    results_1 = run_text_search(
        categories_scope=detected_subcats,
        embedding_query=emb_q1,
        extra_terms=extra_terms,
        top_k=top_k,
        original_query_for_color=query
    )

    if results_1:
        return {"results": results_1, "scope": "subcategory", "detected_subcats": detected_subcats, "sections": sections}

    # ✅ Relaxed fallback: if extra terms (like names) filtered everything out,
    # retry WITHOUT extra_terms and with a cleaner embedding query (category-focused).
    if extra_terms:
        emb_q1_relaxed = build_embedding_query(core_part_1, "", "", matched_keywords)
        results_1b = run_text_search(
            categories_scope=detected_subcats,
            embedding_query=emb_q1_relaxed,
            extra_terms=[],
            top_k=top_k,
            original_query_for_color=query
        )
        if results_1b:
            return {
                "results": results_1b,
                "scope": "subcategory_relaxed",
                "detected_subcats": detected_subcats,
                "sections": sections,
                "ignored_extra_terms": extra_terms,
            }

    # STEP 2
    if not sections:
        return {"results": [], "scope": "subcategory_empty_no_section", "detected_subcats": detected_subcats}

    expanded_subcats = subcats_for_sections(sections)
    core_part_2 = ", ".join(sections)
    emb_q2 = build_embedding_query(core_part_2, translated_en, query, matched_keywords)

    results_2 = run_text_search(
        categories_scope=expanded_subcats,
        embedding_query=emb_q2,
        extra_terms=extra_terms,
        top_k=top_k,
        original_query_for_color=query
    )

    if results_2:
        return {"results": results_2, "scope": "section_fallback", "detected_subcats": detected_subcats, "fallback_sections": sections}

    # ✅ Relaxed fallback on section scope too
    if extra_terms:
        emb_q2_relaxed = build_embedding_query(core_part_2, "", "", matched_keywords)
        results_2b = run_text_search(
            categories_scope=expanded_subcats,
            embedding_query=emb_q2_relaxed,
            extra_terms=[],
            top_k=top_k,
            original_query_for_color=query
        )
        return {
            "results": results_2b,
            "scope": "section_fallback_relaxed",
            "detected_subcats": detected_subcats,
            "fallback_sections": sections,
            "ignored_extra_terms": extra_terms,
        }

    return {"results": [], "scope": "section_fallback_empty", "detected_subcats": detected_subcats, "fallback_sections": sections}

# ============================================================
# EXACT IMAGE MATCH (NO OCR) — SHA256 HASH ON BYTES
# ============================================================

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _image_url_to_local_path(image_url: str):
    if not image_url:
        return None
    try:
        # Works for both: http://127.0.0.1:8000/uploads/xxx.jpg  OR  /uploads/xxx.jpg
        if "/uploads/" in image_url:
            fname = image_url.split("/uploads/")[-1]
            fname = urllib.parse.unquote(fname)
            return os.path.join(UPLOAD_DIR, fname)
    except Exception:
        return None
    return None

def try_exact_image_hash_match(upload_bytes: bytes):
    """If the user uploads the *same file bytes* that already exist in DB, return it immediately.
    This avoids wrong category gating and guarantees exact matches without OCR.
    """
    if not connection or not upload_bytes:
        return None

    qh = _sha256_hex(upload_bytes)
    st_placeholders = ", ".join(["%s"] * len(IMAGE_SEARCH_STATUSES))
    sql = f"""
        SELECT id, title, description, category, location, image_url, date_reported, status
        FROM items
        WHERE matched = 0
          AND status IN ({st_placeholders})
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, list(IMAGE_SEARCH_STATUSES))
        rows = cursor.fetchall()

    for row in rows:
        p = _image_url_to_local_path(row.get("image_url"))
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                bh = f.read()
        except Exception:
            continue
        if _sha256_hex(bh) == qh:
            row["similarity"] = 1.0
            return row
    return None

# ============================================================
# IMAGE SEARCH ENDPOINT (SECTION-FIRST + TRY TOP-K)
# ============================================================

@app.post("/api/search/image")
async def search_image(image: UploadFile, top_k: int = 10):
    if not connection:
        return {"error": "DB Not Connected"}

    try:
        raw_bytes = await image.read()
        exact = try_exact_image_hash_match(raw_bytes)
        if exact:
            return {"results": [exact], "scope": "exact_hash"}

        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        # Compute BOTH embeddings:
        # - full image (best for exact/near-exact matches)
        # - multi (full + center-crop avg) for robust category inference under reflections/folio covers
        q_emb_full = np.array(encode_image(img), dtype=np.float32)
        q_emb_multi = np.array(encode_image_multi(img), dtype=np.float32)
        q_embs = [q_emb_full, q_emb_multi]
        q_emb_infer = q_emb_multi

        q_color = extract_dominant_color(img)
        q_shape = extract_shape_vector(img)
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    # 1) Don't lock on a single section: take Top-3 sections then search within their subcats
    top_sections = infer_best_section_from_image(q_emb_infer, top_n=3)
    best_section, best_section_sim = top_sections[0]

    candidate_sections = [s for s, sc in top_sections if sc >= 0.20]
    if not candidate_sections:
        candidate_sections = [s for s, _ in top_sections[:2]]

    allowed_subcats = subcats_for_sections(candidate_sections)
    top_subcats = infer_best_subcat_from_image(q_emb_infer, top_n=12, allowed_subcats=allowed_subcats)

    # 2) If confidence is low, expand to all subcats (prevents 'book/pen' bias)
    if top_subcats and top_subcats[0][1] < 0.23:
        top_subcats = infer_best_subcat_from_image(q_emb_infer, top_n=12, allowed_subcats=None)

    print(f"[IMG-DETECT] best_section={best_section}({best_section_sim:.3f}) candidate_sections={candidate_sections} top_subcats={top_subcats[:6]}")

    # Similarity thresholds (tune if needed)
    MIN_SIM_SUBCAT = 0.60   # require a solid match within a single subcategory
    MIN_SIM_SECTION = 0.55  # allow slightly broader match within the detected section

    strict_tried = []
    best_results = None
    best_subcat = None
    best_subcat_score = None
    best_sim = -1.0

    # Try top subcategories, but DON'T return the first non-empty list.
    # Pick the subcategory that yields the highest similarity result.
    for subcat, sc in top_subcats[:6]:
        strict_tried.append((subcat, sc))
        cand = run_image_search(
            categories_scope=[subcat],
            q_embs=q_embs,
            q_color=q_color,
            q_shape=q_shape,
            top_k=top_k
        )
        if cand:
            top_sim = float(cand[0].get("similarity", 0.0))
            if top_sim > best_sim:
                best_sim = top_sim
                best_results = cand
                best_subcat = subcat
                best_subcat_score = sc

    if best_results and best_sim >= MIN_SIM_SUBCAT:
        return {
            "results": best_results,
            "scope": "subcategory",
            "best_section": best_section,
            "best_section_score": best_section_sim,
            "best_subcategory": best_subcat,
            "best_subcategory_score": best_subcat_score,
            "strict_tried": strict_tried,
        }
    expanded_subcats = subcats_for_sections(candidate_sections)
    results_2 = run_image_search(
        categories_scope=expanded_subcats,
        q_embs=q_embs,
        q_color=q_color,
        q_shape=q_shape,
        top_k=top_k
    )

    # 3) Section fallback ONLY (NO global fallback).
    # If nothing matches well within the same detected section, return empty results.
    if not results_2:
        return {
            "results": [],
            "scope": "no_match",
            "best_section": best_section,
            "best_section_score": best_section_sim,
            "top_subcats": top_subcats[:10],
            "strict_tried": strict_tried,
        }

    top_sim2 = float(results_2[0].get("similarity", 0.0))
    if top_sim2 < MIN_SIM_SECTION:
        return {
            "results": [],
            "scope": "no_match",
            "reason": f"top_similarity_below_threshold<{MIN_SIM_SECTION}",
            "best_section": best_section,
            "best_section_score": best_section_sim,
            "top_subcats": top_subcats[:10],
            "strict_tried": strict_tried,
        }

    return {
        "results": results_2,
        "scope": "section_fallback",
        "best_section": best_section,
        "best_section_score": best_section_sim,
        "top_subcats": top_subcats[:10],
        "strict_tried": strict_tried,
    }
    
    
@app.get("/api/items/lost")
def list_lost_items():
    if not connection:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, title, description, category, location, image_url, date_reported, status, matched
            FROM items
            WHERE status='lost'
            ORDER BY date_reported DESC
            """
        )
        return cursor.fetchall()


@app.get("/api/items/found")
def list_found_items():
    if not connection:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, title, description, category, location, image_url, date_reported, status, matched
            FROM items
            WHERE status='found'
            ORDER BY date_reported DESC
            """
        )
        return cursor.fetchall()