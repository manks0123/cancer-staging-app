# convert_checkpoint.py
import torch
from pathlib import Path

# พาธไฟล์โมเดลเดิม (ไฟล์ที่คุณได้จากตอนเทรน)
src = Path("weights/efficientnet_b1_checkpoint_fold1.pt")

# พาธไฟล์ใหม่ที่จะสร้าง (state_dict เท่านั้น)
dst = Path("weights/efficientnet_b1_fold1_state_dict.pth")

# โหลดไฟล์เดิม (เราเชื่อว่าไฟล์นี้มาจากตัวเองเทรน)
ckpt = torch.load(src, map_location="cpu", weights_only=False)

# พยายามดึง state_dict ออกมา
if hasattr(ckpt, "state_dict"):  
    # กรณี ckpt เป็นโมเดลเต็ม
    sd = ckpt.state_dict()
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    # กรณี ckpt เป็น dict ที่ห่อ state_dict
    sd = ckpt["state_dict"]
else:
    raise RuntimeError("ไม่พบ state_dict ใน checkpoint ที่ให้มา")

# บันทึกไฟล์ใหม่
torch.save(sd, dst)
print(f"✅ Saved state_dict to {dst}")
