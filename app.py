import streamlit as st
import torch
import clip
from PIL import Image
import io
import base64

# --- Cấu hình trang ---
st.set_page_config(page_title="Phân loại xe dự án OD", page_icon="🚗", layout="centered")

# --- CSS tuỳ chỉnh ---
st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .stApp { background-color: #0E1117; }
    footer { visibility: hidden; }
    .copyright {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tiêu đề chính ---
st.title("🚗 Phân loại loại xe dự án OD")
st.caption("Nhận dạng các loại xe thông dụng bằng mô hình AI CLIP của OpenAI")

# --- Tải mô hình CLIP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Danh sách các loại xe ---
labels = ["SUV", "HATCHBACK", "MINIVAN", "VAN", "PICKUP TRUCK", "SEDAN", "TRUCK", "BUS", "WAGON"]

# --- Hướng dẫn ---
st.markdown("🖼️ **Bạn có thể chọn ảnh hoặc dán trực tiếp (Ctrl + V) vào đây:**")

# --- Upload hoặc dán ảnh ---
uploaded_file = st.file_uploader("📁 Chọn ảnh xe", type=["jpg", "jpeg", "png"])

# --- Dán ảnh bằng clipboard (nếu có) ---
pasted_image_data = st.query_params.get("pasted_image", [None])[0]

if pasted_image_data:
    image_bytes = base64.b64decode(pasted_image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes))
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    st.info("📋 Dán ảnh (Ctrl + V) hoặc tải ảnh để bắt đầu.")
    image = None

# --- Nếu có ảnh ---
if image is not None:
    st.image(image, caption="Ảnh xe được tải lên", use_column_width=True)

    # Tiền xử lý ảnh và dự đoán
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        logits_per_image, _ = model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # --- Hiển thị kết quả ---
    st.success("✅ Kết quả phân loại:")
    for label, prob in zip(labels, probs):
        st.write(f"**{label}**: {prob * 100:.2f}%")

# --- Thêm JavaScript hỗ trợ dán ảnh ---
st.markdown("""
<script>
document.addEventListener('paste', async (event) => {
    const items = event.clipboardData.items;
    for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
            const blob = item.getAsFile();
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64Image = e.target.result;
                const queryParams = new URLSearchParams(window.location.search);
                queryParams.set("pasted_image", base64Image);
                window.location.search = queryParams.toString();
            };
            reader.readAsDataURL(blob);
        }
    }
});
</script>
""", unsafe_allow_html=True)

# --- Bản quyền ---
st.markdown("""
<div class="copyright">© Bản quyền bởi <b>Dino (Thien)</b> – Công ty <b>AIWORX</b></div>
""", unsafe_allow_html=True)
