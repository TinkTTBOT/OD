import streamlit as st
from PIL import Image
import torch
import open_clip
import gc # Import thư viện Garbage Collector

# --- Cấu hình bộ nhớ và PyTorch ---
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
gc.collect()

st.set_page_config(
    # Đổi tiêu đề trang web để chuyên nghiệp hơn
    page_title="Phân loại Xe hơi bằng AI (CLIP)",
    page_icon="🚗",
    layout="wide"
)

# --- Tải model ---
@st.cache_resource
def load_model():
    """Tải mô hình CLIP và các thành phần liên quan, ưu tiên CPU."""
    try:
        device = "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        return model, preprocess, tokenizer, device
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: Vui lòng kiểm tra kết nối mạng hoặc thư viện đã cài đặt. Chi tiết: {e}")
        st.stop()

model, preprocess, tokenizer, device = load_model()

# --- Danh sách nhãn (Labels) ---
labels = [
    "SUV", "Sedan", "Truck", "Van", "Bus", "Pickup Truck",
    "Hatchback", "Minivan", "Wagon", "Coupe", "Convertible"
]
prompts = [f"A photo of a {label} car" for label in labels]


# ===================================================================
# --- GIAO DIỆN CHÍNH (MAIN UI) ---
# ===================================================================

st.title("🚗 Hệ thống Phân loại Loại Xe Tự động (CLIP)")
st.markdown("Bạn có thể nhấn **Enter** sau khi tải ảnh để phân loại.")

# SỬA LỖI GIAO DIỆN: Dùng tỷ lệ [1, 2] hoặc [1, 2.5] để cột 2 có nhiều không gian hơn
col1, col2 = st.columns([1, 2]) # Tỉ lệ 1:2 giúp ảnh và biểu đồ hiển thị đẹp hơn
image = None 
submitted = False 

# --- Bắt đầu Form (Widget tải ảnh và nút) ---
with col1:
    with st.form("classification_form"):
        st.subheader("1. Tải lên Hình ảnh Xe 📸")
        
        uploaded_file = st.file_uploader(
            "📁 Chọn ảnh xe (.png, .jpg, .jpeg) hoặc Kéo và Thả vào đây:", 
            type=["png", "jpg", "jpeg"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.info("💡 Ảnh đã sẵn sàng. Nhấn nút **'Bắt đầu Phân loại'** hoặc nhấn phím **Enter**.")
            except Exception as e:
                st.error(f"❌ Không thể xử lý tệp ảnh. Lỗi: {e}")
        else:
            st.warning("👉 Vui lòng tải lên một ảnh xe (hoặc kéo thả) để bắt đầu.")

        submitted = st.form_submit_button("🔍 Bắt đầu Phân loại", use_container_width=True, type="primary")

# --- Logic Hiển thị Ảnh và Kết quả (Tất cả ở Cột 2) ---
with col2:
    st.subheader("2. Kết quả Phân loại & Ảnh 📊")
    
    # --- Xử lý logic Phân loại sau khi form được gửi (bằng nút hoặc Enter) ---
    if submitted:
        if image is not None:
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

            with st.spinner("⏳ Đang phân tích ảnh và tính độ tương đồng..."):
                try:
                    # Tiền xử lý ảnh và chuyển sang device
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    text_inputs = tokenizer(prompts).to(device)
                    
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_inputs)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        probs = probs[0].tolist()

                        top_idx = torch.argmax(torch.tensor(probs)).item()
                        top_label = labels[top_idx]
                        top_prob = probs[top_idx] * 100

                    st.success(f"🎉 **Kết quả Chính:** Xe thuộc loại **{top_label}**")
                    st.metric(label="Độ Tự Tin (Confidence)", value=f"{top_prob:.2f}%")

                    st.subheader("Chi tiết Độ Tương Đồng:")
                    results_data = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
                    chart_labels = [r[0] for r in results_data]
                    chart_probs = [r[1] for r in results_data]
                    st.bar_chart({"Loại Xe": chart_labels, "Xác Suất": chart_probs}, x="Loại Xe", y="Xác Suất")
                    
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi trong quá trình phân tích: {e}")
                
                finally:
                    # Dọn dẹp bộ nhớ sau khi tính toán
                    gc.collect() 
                    if 'image_input' in locals(): del image_input
                    if 'text_inputs' in locals(): del text_inputs
                    if 'image_features' in locals(): del image_features
                    if 'text_features' in locals(): del text_features
                    if device == "cuda": torch.cuda.empty_cache()

        else:
            st.warning("⚠️ Vui lòng tải lên ảnh trước khi nhấn Phân loại (hoặc Enter).")

    # Hiển thị placeholder nếu chưa có ảnh và chưa nhấn submit lần nào
    elif image is None and not submitted:
        st.info("Ảnh xe của bạn và kết quả phân loại sẽ hiển thị tại đây.")
    
    # Nếu ảnh đã tải lên nhưng chưa nhấn submit
    elif image is not None and not submitted:
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)


# ===================================================================
# --- FOOTER & BẢN QUYỀN ---
# ===================================================================

st.markdown("---")
st.caption("""
    © Bản quyền thuộc về **Thiện, Công ty AIWORKX**. 
    Ứng dụng được xây dựng trên nền tảng Streamlit và mô hình thị giác-ngôn ngữ **CLIP** (OpenAI/OpenCLIP).
""")
