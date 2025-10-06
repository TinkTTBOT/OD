import streamlit as st
from streamlit_paste_image import image_paste
import torch, open_clip

# --- Cấu hình ---
st.set_page_config(page_title="Phân loại xe dự án OD", page_icon="🚗", layout="centered")
st.title("🚗 Phân loại loại xe dự án OD")

# --- Paste ảnh từ clipboard ---
image = image_paste("📋 Dán ảnh vào đây (Ctrl+V) hoặc upload")

# --- Upload file dự phòng ---
uploaded_file = st.file_uploader("📁 Chọn ảnh xe", type=["jpg","jpeg","png"])
if uploaded_file:
    from PIL import Image
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Ảnh xe", use_column_width=True)

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

    # --- Phân loại ---
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer(labels).to(device)
    import torch
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
