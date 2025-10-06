import streamlit as st
from PIL import Image
import torch
import open_clip

# --- Tải model ---
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

model, preprocess, tokenizer = load_model()

# --- Danh sách nhãn ---
labels = [
    "SUV", "Sedan", "Truck", "Van", "Bus", "Pickup Truck",
    "Hatchback", "Minivan", "Wagon", "Coupe", "Convertible"
]

st.title("🚗 Phân loại xe bằng AI (CLIP)")
st.write("Tải lên hoặc dán ảnh xe để hệ thống xác định loại xe chính xác nhất.")

# --- Upload hoặc dán ảnh ---
uploaded_file = st.file_uploader("📁 Chọn ảnh xe", type=["png", "jpg", "jpeg"])
image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
else:
    st.info("👉 Hãy tải ảnh xe hoặc dán ảnh vào đây.")

# --- Nút phân loại ---
if st.button("🔍 Phân loại"):
    if image is not None:
        with st.spinner("⏳ Đang phân tích ảnh..."):
            image_input = preprocess(image).unsqueeze(0)

            text_inputs = tokenizer([f"a photo of a {label}" for label in labels])
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                # Chuẩn hóa vector
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Tính độ tương đồng
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = probs[0].tolist()

            # --- Kết quả chính ---
            top_idx = torch.argmax(torch.tensor(probs)).item()
            top_label = labels[top_idx]
            top_prob = probs[top_idx] * 100

        st.success(f"✅ **Kết quả:** {top_label} ({top_prob:.2f}%)")
    else:
        st.warning("⚠️ Bạn cần chọn ảnh trước khi phân loại.")

