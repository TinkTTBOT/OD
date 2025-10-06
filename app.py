import streamlit as st
import torch
import open_clip
from PIL import Image
import io
import base64

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Phân loại xe dự án OD",
    page_icon="🚗",
    layout="centered",
)

# --- Hàm tải mô hình ---
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

model, preprocess, tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# --- Giao diện ---
st.title("🚗 Phân loại loại xe dự án OD")
st.caption("Nhận dạng các loại xe thông dụng bằng mô hình AI CLIP của OpenAI")

st.write("📸 Bạn có thể **chọn ảnh hoặc dán trực tiếp (Ctrl + V)** vào đây:")

# --- Chức năng tải hoặc dán ảnh ---
uploaded_file = st.file_uploader("📂 Chọn ảnh xe", type=["jpg", "jpeg", "png"])

# Dán ảnh từ clipboard (base64)
pasted_image_data = st.experimental_get_query_params().get("pasted_image", [None])[0]

if pasted_image_data:
    image_bytes = base64.b64decode(pasted_image_data)
    uploaded_file = io.BytesIO(image_bytes)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã chọn", use_container_width=True)

    if st.button("🔍 Phân loại"):
        with st.spinner("Đang xử lý bằng AI..."):
            image_input = preprocess(image).unsqueeze(0).to(device)

            labels = [
                "sedan", "suv", "van", "minivan", "truck",
                "pickup truck", "bus", "hatchback", "wagon", "coupe"
            ]
            text_tokens = tokenizer(labels).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(labels))

            st.success("✅ Kết quả phân loại:")
            for idx, val in zip(indices, values):
                st.write(f"**{labels[idx].upper()}**: {val.item() * 100:.2f}%")

else:
    st.info("👉 Dán ảnh (Ctrl + V) hoặc tải ảnh để bắt đầu.")

# --- Footer bản quyền ---
st.markdown("""
<style>
footer {visibility: hidden;}
.footer-text {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f3f6;
    color: #333;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #ddd;
}
</style>

<div class="footer-text">
    🄫 2025 Bản quyền thuộc về <b>Dino (Thien)</b> · Công ty <b>AIWORX</b>
</div>
""", unsafe_allow_html=True)
