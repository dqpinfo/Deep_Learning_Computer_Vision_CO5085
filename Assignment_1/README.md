# Bài tập lớn A1 — CNN vs Transformer trên CIFAR-100

## Tóm tắt dễ hiểu

Bài A1 trả lời câu hỏi: **giữa hai kiểu "bộ não" nhận dạng ảnh phổ biến nhất hiện nay, kiểu nào học tốt hơn, và cách dạy nào hiệu quả hơn?**

- **CNN (ResNet-18)**: nhìn ảnh qua các "cửa sổ" nhỏ trượt khắp ảnh, giỏi bắt chi tiết cục bộ như cạnh, góc, hoa văn.
- **Transformer (DeiT-Tiny)**: cắt ảnh thành các mảnh 16×16 rồi cho các mảnh "nói chuyện" với nhau (self-attention), giỏi nắm quan hệ toàn ảnh nhưng thường cần nhiều dữ liệu hoặc được học trước.

Hai mô hình được dạy theo bốn cách:

| Cách dạy | Giải thích đời thường |
|---|---|
| **From-scratch** | Học từ con số 0, chỉ dùng ảnh CIFAR-100 |
| **Pretrained + chỉ train head** | Lấy mô hình đã học trên 1,2 triệu ảnh ImageNet, giữ nguyên toàn bộ "kiến thức cũ", chỉ dạy lại lớp ra quyết định cuối |
| **Pretrained + đóng băng một phần** | Giữ nguyên nửa đầu (kiến thức cơ bản: cạnh, màu, kết cấu), cho nửa sau học lại |
| **Pretrained + fine-tune toàn bộ** | Cho toàn bộ mô hình điều chỉnh theo dữ liệu mới |

Tổng cộng: 2 mô hình × 4 cách dạy = **8 thí nghiệm**, chạy trên **cùng một cách chia dữ liệu** để so sánh công bằng.

## Mốc A1.1 (26/10/2026) — Dataset + protocol

A1.1 chỉ yêu cầu **chốt dữ liệu và kế hoạch thí nghiệm**, chưa cần kết quả cuối. Nhóm đã làm:

1. **Chọn tập dữ liệu CIFAR-100** và nêu lý do, license, cách chia split.
2. **Khám phá dữ liệu (EDA)**: số ảnh mỗi lớp, ảnh mẫu, thống kê màu, 20 nhóm lớp lớn (`eda.py`).
3. **Mô tả DataLoader và augmentation** (`data.py`).
4. **Liệt kê mô hình CNN và Transformer**, cùng kế hoạch from-scratch / pretrained / đóng băng (`config.py`, `models.py`).
5. **Kiểm tra pipeline**: tạo đủ 8 cấu hình, đếm tham số, chạy thử một bước huấn luyện, kiểm tra phần đóng băng đúng (`smoke_test.py`), và chạy thử vòng huấn luyện trên tập con nhỏ (`train.py --subset`).
6. **Báo cáo A1.1**: [`Reports/A1_report.pdf`](../Reports/A1_report.pdf).

## Mốc A1.2 (09/11/2026) — So sánh cuối

Chạy đủ 8 thí nghiệm bằng `train.py`, sau đó lập bảng số liệu (accuracy, macro-F1, thời gian, số tham số trainable), biểu đồ, confusion matrix và phần thảo luận. Phần huấn luyện thật nên chạy trên GPU, ví dụ Google Colab T4, vì máy Mac M1 8 GB quá chậm cho 8 lần huấn luyện ở độ phân giải 224×224.

## Cấu trúc thư mục

```text
Assignment_1/
├── config.py          # Cấu hình chung: seed, split, kích thước ảnh, 8 thí nghiệm, siêu tham số
├── data.py            # Tải CIFAR-100, chia train/val/test, augmentation, DataLoader
├── models.py          # Tạo ResNet-18 / DeiT-Tiny, các kiểu đóng băng
├── train.py           # Vòng lặp huấn luyện + đánh giá tự viết cho một cấu hình
├── eda.py             # Khám phá dữ liệu, xuất hình vào output/eda/
├── smoke_test.py      # Kiểm tra nhanh 8 cấu hình, xuất output/protocol_table.md
├── splits/            # File split cố định (seed 42) dùng chung cho mọi thí nghiệm
├── output/            # Hình EDA, bảng protocol, log huấn luyện
└── requirements.txt
```

Thư mục `data/` (CIFAR-100, khoảng 140 MB) được tải tự động khi chạy lần đầu và **không commit** lên GitHub. Dữ liệu lấy từ bản sao chính thức trên Hugging Face ([`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100)), giống hệt bản gốc của Đại học Toronto (cùng ảnh, nhãn và thứ tự). Nhóm dùng bản này vì server gốc tải rất chậm.

## Cách chạy (macOS / Linux)

```bash
cd Assignment_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python eda.py                                         # EDA -> output/eda/
python smoke_test.py                                  # Kiểm tra 8 cấu hình -> output/protocol_table.md
python train.py --exp resnet18_head --subset 512 --epochs 1   # Chạy thử nhanh pipeline
python train.py --exp deit_tiny_partial               # Huấn luyện thật một cấu hình (A1.2)
```

Tên 8 cấu hình: `resnet18_scratch`, `resnet18_head`, `resnet18_partial`, `resnet18_full`, `deit_tiny_scratch`, `deit_tiny_head`, `deit_tiny_partial`, `deit_tiny_full`.
