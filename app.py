import streamlit as st
import torch
import open_clip
from PIL import Image

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Phân loại xe dự án OD",
    page_icon="🚗",
    layout="centered"
)

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

# --- Load model ---
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = load_m_
