# Hướng dẫn cài đặt môi trường và chạy chương trình trên macOS
<br>

**1. Tạo và active môi trường ảo:**  
python -m venv .venv  
source .venv/bin/activate

**2. Cài đặt torch, torchvision sử dụng CPU**  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

**3. Cài đặt các thư viện còn lại**  
pip install -r requirements.txt --ignore-installed torch torchvision

**4. Train các mô hình Softmax, MLP và CNN**  
python train.py

**5. Đánh giá các mô hình Softmax, MLP và CNN**  
python evaluate.py