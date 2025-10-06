import streamlit as st
from PIL import Image
import torch
import open_clip

# Thiết lập cấu hình trang cho chuyên nghiệp hơn
st.set_page_config(
    page_title="Phân loại Xe hơi bằng AI (CLIP)",
    page_icon="🚗",
    layout="wide" # Sử dụng layout rộng để tận dụng không gian màn hình
)

# --- Tải model ---
@st.cache_resource
def load_model():
    # Sử dụng 'ViT-B-32' như bạn đã làm, đây là một lựa chọn tốt cho tốc độ
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

try:
    model, preprocess, tokenizer = load_model()
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình: Vui lòng kiểm tra kết nối mạng hoặc thư viện đã cài đặt. Chi tiết: {e}")
    st.stop() # Dừng ứng dụng nếu tải model thất bại

# --- Danh sách nhãn (Labels) ---
labels = [
    "SUV", "Sedan", "Truck", "Van", "Bus", "Pickup Truck",
    "Hatchback", "Minivan", "Wagon", "Coupe", "Convertible"
]
# Tạo prompt chuyên nghiệp hơn
prompts = [f"A photo of a {label} car" for label in labels]


# ===================================================================
# --- GIAO DIỆN CHÍNH (MAIN UI) ---
# ===================================================================

st.title("🚗 Phân loại Loại Xe Tự động bằng AI")
st.markdown("Sử dụng mô hình **CLIP (Contrastive Language–Image Pre-training)** để xác định loại xe dựa trên hình ảnh. Mô hình này rất mạnh mẽ trong việc hiểu cả hình ảnh và văn bản.")

# Tạo hai cột để bố cục đẹp hơn
col1, col2 = st.columns([1, 1.5]) 

with col1:
    st.subheader("1. Tải lên Hình ảnh Xe 📸")
    uploaded_file = st.file_uploader(
        "📁 Chọn ảnh xe (.png, .jpg, .jpeg) từ thiết bị của bạn", 
        type=["png", "jpg", "jpeg"]
    )
    image = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.info("💡 Nhấn nút 'Phân loại' ở cột bên cạnh để xem kết quả.")
        except Exception as e:
            st.error(f"❌ Không thể xử lý tệp ảnh. Lỗi: {e}")
    else:
        st.warning("👉 Vui lòng tải lên một ảnh xe để bắt đầu.")


with col2:
    st.subheader("2. Kết quả Phân loại & Ảnh 📊")
    
    if image is not None:
        # Hiển thị ảnh trong cột 2, dùng tham số mới thay thế 'use_column_width'
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        # --- Nút phân loại ---
        if st.button("🔍 Bắt đầu Phân loại", use_container_width=True, type="primary"):
            
            with st.spinner("⏳ Đang phân tích ảnh và tìm độ tương đồng với các loại xe..."):
                try:
                    # Tiền xử lý ảnh
                    image_input = preprocess(image).unsqueeze(0)

                    # Mã hóa văn bản (labels/prompts)
                    text_inputs = tokenizer(prompts)
                    
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_inputs)

                        # Chuẩn hóa vector
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                        # Tính độ tương đồng cosine (logits)
                        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        probs = probs[0].tolist()

                        # --- Xử lý Kết quả ---
                        top_idx = torch.argmax(torch.tensor(probs)).item()
                        top_label = labels[top_idx]
                        top_prob = probs[top_idx] * 100

                    st.success(f"🎉 **Kết quả Chính:** Xe thuộc loại **{top_label}**")
                    st.metric(label="Độ Tự Tin", value=f"{top_prob:.2f}%")

                    # Hiển thị tất cả kết quả dưới dạng biểu đồ
                    st.subheader("Chi tiết Độ tương đồng")
                    # Gộp kết quả vào DataFrame (cho việc hiển thị biểu đồ tốt hơn)
                    results = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
                    
                    # Chuẩn bị dữ liệu cho bar chart
                    chart_labels = [r[0] for r in results]
                    chart_probs = [r[1] for r in results]
                    
                    st.bar_chart({"Loại Xe": chart_labels, "Xác Suất": chart_probs}, x="Loại Xe", y="Xác Suất")
                    
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi trong quá trình phân tích: {e}")
                    
    else:
        st.info("Ảnh sẽ hiển thị ở đây sau khi bạn tải lên.")
        
# ===================================================================
# --- FOOTER & BẢN QUYỀN ---
# ===================================================================

st.markdown("---")
st.caption("""
    © Bản quyền thuộc về **Thiện, Công ty AIWORKX**. 
    Ứng dụng được xây dựng trên nền tảng Streamlit và mô hình thị giác-ngôn ngữ **CLIP** (OpenAI/OpenCLIP).
""")
