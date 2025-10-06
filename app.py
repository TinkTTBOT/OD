import streamlit as st
import torch
import open_clip
from PIL import Image
import io, base64

# --- Cấu hình trang ---
st.set_page_config(page_title="Phân loại xe dự án OD", page_icon="🚗", layout="centered")

# --- CSS ---
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

st.title("🚗 Phân loại loại xe dự án OD")
st.caption("Nhận dạng các loại xe thông dụng bằng mô hình AI CLIP (OpenCLIP)")

# --- Load model ---
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = load_model()
model.to(device)

labels = ["SUV", "HATCHBACK", "MINIVAN", "VAN", "PICKUP TRUCK", "SEDAN", "TRUCK", "BUS", "WAGON"]

# --- Session state để lưu ảnh ---
if "image" not in st.session_state:
    st.session_state.image = None

# --- Upload hoặc paste base64 ---
uploaded_file = st.file_uploader("📁 Chọn ảnh xe", type=["jpg", "jpeg", "png"])
paste_base64 = st.text_area("📋 Dán ảnh dưới dạng base64 ở đây (Ctrl+V) hoặc bỏ trống", height=50)

if uploaded_file:
    st.session_state.image = Image.open(uploaded_file)
elif paste_base64:
    try:
        image_bytes = base64.b64decode(paste_base64.split(",")[-1])
        st.session_state.image = Image.open(io.BytesIO(image_bytes))
    except:
        st.warning("❌ Base64 không hợp lệ.")

image = st.session_state.image

# --- Hiển thị và phân loại ---
if image:
    st.image(image, caption="Ảnh xe", use_column_width=True)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer(labels).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity[0].cpu().numpy()

    st.success("✅ Kết quả phân loại:")
    for label, prob in zip(labels, probs):
        st.write(f"**{label}**: {prob * 100:.2f}%")

# --- Bản quyền ---
st.markdown("""
<div class="copyright">© Bản quyền bởi <b>Dino (Thien)</b> – Công ty <b>AIWORX</b></div>
""", unsafe_allow_html=True)
