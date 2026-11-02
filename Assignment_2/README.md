# Bài tập lớn A2 — Phân đoạn ảnh thú cưng 38 lớp: U-Net (CNN) vs SegFormer (Transformer)

## Tóm tắt dễ hiểu

Bài A2 dạy máy **tô màu từng điểm ảnh** trong ảnh chó mèo: điểm nào là nền, điểm nào thuộc con vật, và con vật đó là **giống nào** trong 37 giống (12 giống mèo, 25 giống chó). Vì vậy mỗi điểm ảnh có 38 khả năng: nền + 37 giống. Mô hình phải vừa khoanh đúng hình dạng con vật, vừa nhận ra đúng giống, nên khó hơn nhiều so với chỉ tách con vật khỏi nền.

Hai "bộ não" được so sánh, đều đã được học trước trên ImageNet và có số tham số tương đương:

| Mô hình | Họ | Cách hoạt động |
|---|---|---|
| **U-Net + ResNet-50** | CNN | Thu nhỏ ảnh dần để hiểu nội dung, rồi phóng to lại từng bước, ghép với chi tiết của tầng trước để vẽ biên chính xác |
| **SegFormer-B2** | Transformer | Cho các vùng ảnh "nói chuyện" với nhau ở nhiều tỉ lệ (self-attention), rồi một bộ giải mã toàn MLP gộp lại thành bản đồ nhãn |

Tập dữ liệu: **Oxford-IIIT Pet** (Parkhi et al., CVPR 2012), 7.349 ảnh có mặt nạ trimap. [Trang chính thức](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Paper nhóm chọn để phân tích

> E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

| Tài nguyên | Link |
|---|---|
| Trang arXiv | [arXiv:2105.15203](https://arxiv.org/abs/2105.15203) |
| File PDF | [arxiv.org/pdf/2105.15203](https://arxiv.org/pdf/2105.15203) (18 trang) |
| Mã nguồn chính thức | [github.com/NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) |
| Trọng số dùng trong bài | [`nvidia/mit-b2`](https://huggingface.co/nvidia/mit-b2) trên Hugging Face (encoder MiT-B2 pretrained ImageNet-1k) |

**Vì sao chọn bài này**

- **Liên quan trực tiếp đến phần cài đặt:** SegFormer-B2 chính là một trong hai mô hình nhóm huấn luyện, nên mọi thí nghiệm của nhóm đều đối chiếu được với paper.
- **Hội nghị hàng đầu, được dùng rộng rãi:** NeurIPS 2021; SegFormer là mô hình Transformer chuẩn cho phân đoạn ảnh, có cài đặt chính thức trong thư viện Hugging Face Transformers.
- **Vừa sức tính toán:** họ mô hình có nhiều cỡ (B0–B5); bản B2 (27,5 triệu tham số theo paper) huấn luyện được trên GPU T4 của Colab.

**Nhóm dựa vào paper như thế nào**

| Từ paper | Trong bài của nhóm |
|---|---|
| Kiến trúc: encoder Mix Transformer phân cấp + bộ giải mã toàn MLP | Biến thể Transformer `segformer_b2` (`models.py`) |
| Cách huấn luyện: AdamW, lr ban đầu 6e-5, lịch "poly" hệ số 1.0, batch 16 (mục 4 của paper); bộ giải mã học nhanh gấp 10 lần encoder (file cấu hình trong mã nguồn chính thức) | Dùng chung cho cả hai mô hình (`config.TRAIN_HPARAMS`) để so sánh công bằng |
| Metric chính: mIoU | Metric chính của nhóm: mIoU 38 lớp (`metrics.py`) |
| Tuyên bố: bộ giải mã MLP gọn mà vẫn đủ tốt; mạnh hơn CNN cùng cỡ | Kiểm chứng ở A2.2 bằng cách so với U-Net + ResNet-50 trên tập nhỏ hơn nhiều so với tập của paper, đặc biệt ở chất lượng biên con vật |

**Bản PDF trên máy:** paper được phát hành trên arXiv theo giấy phép chỉ cho arXiv quyền phân phối, nên nhóm không đưa file PDF lên GitHub. Để đọc offline, tải về thư mục `paper/` (đã nằm trong `.gitignore`):

```bash
mkdir -p paper && curl -L -o paper/SegFormer_NeurIPS2021_arXiv2105.15203.pdf https://arxiv.org/pdf/2105.15203
```

## Mốc A2.1 (02/11/2026) — Chủ đề, paper, dữ liệu, kế hoạch

1. **Chốt chủ đề, paper, tập dữ liệu và metric** (mIoU 38 lớp là chỉ số chính; thêm mean Dice, pixel accuracy, IoU thú/nền, thời gian và tốc độ).
2. **Tải và kiểm tra dữ liệu** (`data.py`): tải hai file gốc, kiểm tra MD5 trùng bản chính thức, tạo mặt nạ 38 lớp từ trimap (vùng viền = 255, bỏ qua khi tính loss/metric).
3. **Chia tập cố định** (`splits/pet_split.json`, seed 42): 3.303 train / 370 validation (10 ảnh mỗi giống, lấy từ trainval chính thức) / 3.662 test chính thức. 14 ảnh có trimap lỗi (toàn nền dù có con vật) bị loại khỏi mọi split.
4. **Khám phá dữ liệu (EDA)** (`eda.py`, kết quả ở `output/eda/`).
5. **DataLoader và augmentation** dùng chung cho cả hai mô hình (`data.py`).
6. **Hai mô hình và kiểm tra pipeline** (`models.py`, `metrics.py`, `smoke_test.py`): dựng được cả hai mô hình, forward/backward chạy đúng, loss giảm khi học thử trên một batch.
7. **Báo cáo A2.1**: [`Reports/A2_1_report.pdf`](../Reports/A2_1_report.pdf).

## Mốc A2.2 (16/11/2026) — Huấn luyện, so sánh, phân tích paper

- 8 lần huấn luyện trên Google Colab (GPU T4): 2 mô hình × 4 lượng dữ liệu train (100%, 50%, 25%, 10%) để xem mô hình nào cần ít nhãn hơn.
- Bảng số liệu, đường học, IoU từng giống, nhầm giống cùng loài so với nhầm chó ↔ mèo, ảnh lỗi điển hình.
- Phân tích đầy đủ paper SegFormer và đối chiếu với kết quả của nhóm.
- Phần mở rộng dự kiến: thử chéo trên ảnh chó/mèo của Pascal VOC, demo.

## Cách chạy

```bash
pip install -r requirements.txt
python eda.py          # tải dữ liệu (lần đầu), tạo split, vẽ hình EDA vào output/eda/
python smoke_test.py   # kiểm tra hai mô hình + DataLoader + metric, ghi output/protocol_table.md
```

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `config.py` | Đường dẫn, seed, định nghĩa 38 lớp, kích thước ảnh, ma trận thí nghiệm, siêu tham số |
| `data.py` | Tải + kiểm tra MD5, đọc danh sách chính thức, tạo mặt nạ 38 lớp, chia split, augmentation, DataLoader |
| `eda.py` | Thống kê và hình EDA |
| `models.py` | U-Net + ResNet-50 (timm) và SegFormer-B2 (Hugging Face) |
| `metrics.py` | mIoU, Dice, pixel accuracy, IoU thú/nền từ ma trận nhầm lẫn |
| `smoke_test.py` | Kiểm tra pipeline mốc A2.1 |
| `splits/pet_split.json` | Danh sách ảnh của từng split (cố định) |
| `report/` | Mã nguồn LaTeX báo cáo |
| `paper/` | Bản PDF paper SegFormer để đọc offline (tải bằng lệnh ở trên, không commit) |
