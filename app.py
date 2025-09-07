import os
import json
from datetime import datetime
from flask import Flask, request, render_template
import torch
import torchvision.transforms as T
import timm
from PIL import Image
import numpy as np

# ===================== CONFIG =====================
NUM_CLASSES = 3
CLASS_NAMES = ["Stage_A", "Stage_B", "Stage_C"]   # <-- ใส่ชื่อคลาสจริงของคุณ
INPUT_SIZE = (240, 240)                           # EfficientNet-B1 มาตรฐานใน timm
UPLOAD_DIR = "static/uploads"
WEIGHTS_PATH = "weights/efficientnet_b1_fold1_state_dict.pth"  # state_dict ที่แปลงแล้ว
HISTORY_PATH = "predictions.jsonl"                # เก็บประวัติการทำนาย
MAX_HISTORY_TO_SHOW = 12                          # แสดงล่าสุดกี่รายการ

# สร้างโฟลเดอร์อัปโหลดถ้ายังไม่มี (และต้องเป็นโฟลเดอร์จริง ๆ)
if not os.path.isdir(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===================== MODEL ======================
def build_model(num_classes: int = NUM_CLASSES):
    model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=num_classes)
    return model

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"ไม่พบไฟล์น้ำหนัก: {WEIGHTS_PATH}\n"
        f"โปรดตรวจชื่อไฟล์และตำแหน่งในโฟลเดอร์ weights/"
    )

device = torch.device("cpu")
model = build_model()
state = torch.load(WEIGHTS_PATH, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=True)
model.eval()

# ===================== PREPROCESS =================
transform = T.Compose([
    T.Resize(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ===================== HISTORY HELPERS ============
def append_history(record: dict):
    """บันทึกผลทำนายลงบรรทัดสุดท้ายของ predictions.jsonl"""
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_history(limit: int = MAX_HISTORY_TO_SHOW):
    """โหลดประวัติล่าสุด (เรียงจากใหม่ไปเก่า)"""
    if not os.path.exists(HISTORY_PATH):
        return []
    items = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    # เรียงตามเวลา (ใหม่ก่อน) แล้วตัดจำนวน
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return items[:limit]

# ===================== FLASK APP ==================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    history = load_history()
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return "No file uploaded", 400

    f = request.files["file"]
    # กันชื่อไฟล์ซ้ำเล็กน้อย: เติม timestamp หน้าไฟล์
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(f.filename)
    safe_name = f"{base}_{ts}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(save_path)

    # เปิดภาพ + แปลงเป็นเทนเซอร์
    img = Image.open(save_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = int(np.argmax(probs))

    # ตารางความน่าจะเป็นต่อคลาส
    prob_table = [(CLASS_NAMES[i], float(probs[i])) for i in range(NUM_CLASSES)]

    # บันทึกประวัติ
    record = {
        "image_path": save_path.replace("\\", "/"),
        "pred_label": CLASS_NAMES[pred_idx],
        "pred_conf": float(probs[pred_idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": safe_name,
    }
    append_history(record)

    disclaimer = "For research/education only. Not for clinical use."

    # โหลดประวัติล่าสุดมาโชว์ด้วย
    history = load_history()

    return render_template(
        "index.html",
        image_path=record["image_path"],
        prediction_label=record["pred_label"],
        prediction_confidence=f"{record['pred_conf']:.3f}",
        prob_table=prob_table,
        disclaimer=disclaimer,
        history=history
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
