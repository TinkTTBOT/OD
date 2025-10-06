import streamlit as st
from PIL import Image
import torch
import open_clip
import numpy as np
import io
import base64

# =============================
# 🏁 CẤU HÌNH CƠ BẢN
# =============================
st.set_page_config(page_title="Phân loại xe bằng AI (CLIP)", page_icon="🚗", layout="centered")

st.title("🚗 Phân loại xe bằng AI (CLIP)")
st.write("Tải ảnh hoặc **dán ảnh (Ctrl + V)** để AI phân loại loại xe.")

# =============================
# 📦 TẢI MODEL
# =============================
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = load_model()
model.to(device).eval()

# =============================
# 📷 GIAO DIỆN TẢI ẢNH
# =============================

st.markdown("""
<style>
div[data-testid="stFileUploader"] > div {
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Chọn ảnh xe**", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- Dán ảnh bằng clipboard ---
st.markdown("""
<hr>
<p style='text-align:center;'>Hoặc dán ảnh bằng <b>Ctrl + V</b> (hỗ trợ trình duyệt Chrome / Edge)</p>
<input type="file" accept="image/*" id="paster" style="display:none">
<script>
document.addEventListener('paste', function(event) {
  const items = (event.clipboardData || event.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.type.indexOf('image') !== -1) {
      const file = item.getAsFile();
      const reader = new FileReader();
      reader.onload = function(e) {
        const data = e.target.result.split(',')[1];
        const url = window.location.href.split('?')[0] + "?pasted=" + encodeURIComponent(data);
        window.location.href = url;
      };
      reader.readAsDataURL(file);
      break;
    }
  }
});
</script>
""", unsafe_allow_html=True)

# --- Kiểm tra ảnh đã chọn hoặc dán ---
paste_img_data = st.experimental_get_query_params().get("pasted", [None])[0]
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
elif paste_img_data:
    img_data = base64.b64decode(paste_img_data)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
else:
    img = None

# =============================
# 🧠 PHÂN LOẠI ẢNH
# =============================
if img:
    st.image(img, caption="Ảnh đã chọn", width=400)

    if st.button("🔍 Phân loại"):
        with st.spinner("AI đang phân tích..."):
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            # Các nhóm xe
            labels = [
                "sedan", "hatchback", "wagon", "minivan",
                "SUV", "pickup truck", "truck", "van", "bus"
            ]
            text = [f"a photo of a {l}" for l in labels]
            text_tokens = tokenizer(text).to(device)

            with torch.no_grad():
                image_features = model.encode_image(input_tensor)
                text_features = model.encode_text(text_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0).softmax(dim=0)

            probs = similarity.cpu().numpy()
            results = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

            st.success("✅ Kết quả phân loại:")
            for label, prob in results:
                st.write(f"**{label.upper()}**: {prob * 100:.2f}%")

else:
    st.info("⬆️ Hãy chọn hoặc dán ảnh để bắt đầu.")
