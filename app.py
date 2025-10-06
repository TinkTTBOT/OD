import streamlit as st
import torch
import open_clip
from PIL import Image
import io, base64

# --- Cấu hình trang ---
st.set_page_config(page_title="Phân loại xe dự án OD", page_icon="🚗", layout="centered")

# --- CSS giao diện ---
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

# --- Tải mô hình OpenCLIP ---
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = load_model()
model.to(device)

# --- Danh sách nhãn ---
labels = ["SUV", "HATCHBACK", "MINIVAN", "VAN", "PICKUP TRUCK", "SEDAN", "TRUCK", "BUS", "WAGON"]

# --- Lưu ảnh dán trong session_state ---
if "pasted_image" not in st.session_state:
    st.session_state.pasted_image = None

# --- Giao diện upload/dán ảnh ---
st.markdown("🖼️ **Bạn có thể chọn ảnh hoặc dán trực tiếp (Ctrl + V):**")
uploaded_file = st.file_uploader("📁 Chọn ảnh xe", type=["jpg", "jpeg", "png"])

# --- Lấy ảnh ---
if uploaded_file:
    image = Image.open(uploaded_file)
    st.session_state.pasted_image = image
elif st.session_state.pasted_image:
    image = st.session_state.pasted_image
else:
    st.info("📋 Dán ảnh (Ctrl + V) hoặc tải ảnh để bắt đầu.")
    image = None

# --- Xử lý ảnh ---
if image is not None:
    st.image(image, caption="Ảnh xe được tải lên", use_column_width=True)
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

# --- Script hỗ trợ dán ảnh (lưu vào session_state, không reload) ---
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
                // Gửi base64 lên Streamlit qua streamlit.setComponentValue
                const input = document.createElement('input');
                input.type = 'text';
                input.id = 'pasted_image_input';
                input.value = base64Image;
                input.style.display = 'none';
                document.body.appendChild(input);
                input.dispatchEvent(new Event('change'));
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
