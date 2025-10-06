import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import torch
import clip
from torchvision import transforms
import io

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
CAR_LABELS = ["Sedan", "SUV", "Truck", "Van"]

# --- Hàm nhận diện ---
def predict_image(image_pil):
    image = preprocess(image_pil).unsqueeze(0).to(DEVICE)
    text = clip.tokenize([f"a photo of a {label}" for label in CAR_LABELS]).to(DEVICE)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    results = list(zip(CAR_LABELS, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Ứng dụng chính ---
class CarClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Phân loại xe dự án OD")
        self.root.geometry("720x680")
        self.root.configure(bg="#f2f2f2")

        self.image_path = None
        self.image_pil = None
        self.tk_image = None

        # --- Nút chọn ảnh ---
        self.btn_select = tk.Button(
            root, text="📁 Chọn ảnh", command=self.choose_image,
            bg="#dbeafe", fg="black", font=("Arial", 11, "bold"),
            relief="groove", width=15
        )
        self.btn_select.pack(pady=8)

        # --- Nút dán ảnh ---
        self.btn_paste = tk.Button(
            root, text="📋 Dán ảnh (Ctrl+V)", command=self.paste_image,
            bg="#fde68a", fg="black", font=("Arial", 11, "bold"),
            relief="groove", width=15
        )
        self.btn_paste.pack(pady=4)

        # --- Nút phân loại ---
        self.btn_predict = tk.Button(
            root, text="🔍 Phân loại", command=self.classify_image,
            bg="#bbf7d0", fg="black", font=("Arial", 11, "bold"),
            relief="groove", width=15
        )
        self.btn_predict.pack(pady=4)

        # --- Khung ảnh ---
        self.canvas_frame = tk.Frame(root, bg="white", bd=1, relief="solid")
        self.canvas_frame.pack(padx=20, pady=15)
        self.canvas = tk.Canvas(self.canvas_frame, width=560, height=330, bg="white", highlightthickness=0)
        self.canvas.pack()

        # --- Kết quả ---
        self.result_label = tk.Label(root, text="", bg="#f2f2f2", font=("Consolas", 11))
        self.result_label.pack(pady=10)

        # --- Phím tắt Ctrl+V ---
        root.bind("<Control-v>", lambda e: self.paste_image())

    # --- Chọn ảnh từ file ---
    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        img = Image.open(file_path).convert("RGB")
        self.image_path = file_path
        self.image_pil = img
        self.display_image(img)

    # --- Dán ảnh từ clipboard ---
    def paste_image(self):
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showwarning("Thông báo", "Không có ảnh trong clipboard!")
                return
            img = img.convert("RGB")
            self.image_pil = img
            self.image_path = None
            self.display_image(img)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể dán ảnh!\n{e}")

    # --- Hiển thị ảnh ---
    def display_image(self, img):
        max_w, max_h = 560, 330
        ratio = min(max_w / img.width, max_h / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img_resized = img.resize(new_size, Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        x = (560 - new_size[0]) // 2
        y = (330 - new_size[1]) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)

    # --- Phân loại ---
    def classify_image(self):
        if self.image_pil is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn hoặc dán ảnh trước!")
            return
        results = predict_image(self.image_pil)
        text = "📊 Xác suất:\n"
        for name, prob in results:
            text += f"{name:<10}: {prob*100:5.2f}%\n"
        self.result_label.config(text=text)

# --- Khởi động ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CarClassifierApp(root)
    root.mainloop()
