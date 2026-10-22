############################
#   Đánh giá các mô hình   #
############################

import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataset import get_data_loaders
from models import PatchTokenizer, RowTokenizer, MyMSA, PyTorchMSA, ImageTransformerClassifier

# Đếm tổng số tham số cần huấn luyện của mô hình (Model Capacity)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Tính độ chính xác của mô hình trên tập test
def evaluate_model(model, test_loader):
    model.eval()                                            # Chuyển mô hình sang chế độ đánh giá (evaluation mode) để tắt dropout, batch normalization, v.v.

    correct = 0
    total = 0
    
    with torch.no_grad():                                   # Tắt tính toán gradient để tiết kiệm bộ nhớ
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)       # Lấy nhãn dự đoán bằng cách chọn lớp có điểm số cao nhất (argmax) từ logits
            total += labels.size(0)                         # Cập nhật tổng số mẫu đã xử lý trong tập test
            correct += (predicted == labels).sum().item()   # Tính số lượng dự đoán đúng bằng cách so sánh nhãn dự đoán với nhãn thực tế
            
    accuracy = correct / total                              # Tính độ chính xác bằng cách chia số lượng dự đoán đúng cho tổng số mẫu
    return accuracy

# Hàm vẽ biểu đồ Loss và Accuracy so sánh giữa các mô hình
def plot_learning_curves(output_dir = Path("./output")):
    history_file = output_dir / "training_history.json"
    
    if not history_file.exists():
        print(f"File lịch sử huấn luyện không tồn tại: {history_file}")
        return
    
    # Đọc dữ liệu lịch sử huấn luyện từ file JSON
    with open(history_file, "r") as f:
        history_dict = json.load(f)

    _, axes = plt.subplots(1, 2, figsize = (14, 5))
    
    # Biểu đồ Loss
    for name, history in history_dict.items():
        axes[0].plot(history['train_loss'], label = name, linewidth = 2)
    axes[0].set_title('So sánh Training Loss qua các Epochs', fontsize = 12, fontweight = 'bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle = '--', alpha = 0.7)
    
    # Biểu đồ Accuracy
    for name, history in history_dict.items():
        axes[1].plot(history['train_acc'], label = name, linewidth = 2)
    axes[1].set_title('So sánh Training Accuracy qua các Epochs', fontsize = 12, fontweight = 'bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle = '--', alpha = 0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_curves.png', dpi = 300)
    print("-> Đã lưu biểu đồ so sánh Loss và Accuracy: comparison_curves.png")
    plt.show()

# Vẽ bảng tổng hợp số liệu (tham số, accuracy) và lưu thành file ảnh
def plot_summary_table(table_data, output_dir = Path("./output")):
    
    fig, ax = plt.subplots(figsize = (7, 3))
    ax.axis('off')
    
    table = ax.table(cellText = table_data, loc = 'center', cellLoc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Định dạng màu cho hàng tiêu đề
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color = 'white', weight = 'bold')
            
    ax.set_title('Bảng tổng hợp chỉ số so sánh Bài tập E2', y = 0.85, weight = 'bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'summary_table.png', dpi = 300)
    plt.show()
    print(f"Đã lưu bảng tổng hợp tại: '{output_dir / f'summary_table.png'}'")

# Vẽ biểu đồ cột so sánh Accuracy giữa các mô hình và lưu thành file ảnh
def plot_accuracy_comparison(models_name, accuracies, output_dir = Path("./output")):

    fig, ax = plt.subplots(figsize = (6, 5))
    bars = ax.bar(models_name, accuracies, color = ['skyblue', 'salmon'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_title('So sánh độ chính xác (Accuracy) các mô hình')
    
    # Ghi giá trị trực tiếp lên đầu cột
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.2f}%', ha = 'center', va = 'bottom')
        
    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_comparison.png', dpi = 300)
    plt.show()
    print(f"Đã lưu biểu đồ Accuracy: '{output_dir / f'accuracy_comparison.png'}'")


def main():
    output_dir = Path("./output")

    if output_dir.is_dir() == False:
        print(f"Folder not found: {output_dir}")
        return
    else:
        # Tải dữ liệu test
        _, test_loader = get_data_loaders(batch_size = 64, subset_test = 1000)

        # Khởi tạo và nạp trọng số đã huấn luyện cho các mô hình
        model_patch_my_msa = ImageTransformerClassifier(
            tokenizer = PatchTokenizer(img_size = 28, patch_size = 7, in_channels = 1, embed_dim = 64),
            attention = MyMSA(embed_dim = 64, num_heads = 4),
            embed_dim = 64,
            num_classes = 10
        )
        model_patch_my_msa.load_state_dict(torch.load(output_dir / "patchtokenizer_mymsa_weights.pth"))

        model_patch_pytorch_msa = ImageTransformerClassifier(
            tokenizer = PatchTokenizer(img_size = 28, patch_size = 7, in_channels = 1, embed_dim = 64),
            attention = PyTorchMSA(embed_dim = 64, num_heads = 4),
            embed_dim = 64,
            num_classes = 10
        )
        model_patch_pytorch_msa.load_state_dict(torch.load(output_dir / "patchtokenizer_pytorchmsa_weights.pth"))

        model_row_my_msa = ImageTransformerClassifier(
            tokenizer = RowTokenizer(img_size = 28, in_channels = 1, embed_dim = 64),
            attention = MyMSA(embed_dim = 64, num_heads = 4),
            embed_dim = 64,
            num_classes = 10
        )
        model_row_my_msa.load_state_dict(torch.load(output_dir / "rowtokenizer_mymsa_weights.pth"))

        model_row_pytorch_msa = ImageTransformerClassifier(
            tokenizer = RowTokenizer(img_size = 28, in_channels = 1, embed_dim = 64),
            attention = PyTorchMSA(embed_dim = 64, num_heads = 4),
            embed_dim = 64,
            num_classes = 10
        )
        model_row_pytorch_msa.load_state_dict(torch.load(output_dir / "rowtokenizer_pytorchmsa_weights.pth"))

        # So sánh số lượng tham số
        params_patch_my_msa = count_parameters(model_patch_my_msa)
        params_patch_pytorch_msa = count_parameters(model_patch_pytorch_msa)
        params_row_my_msa = count_parameters(model_row_my_msa)
        params_row_pytorch_msa = count_parameters(model_row_pytorch_msa)

        print("\n===================================================================")
        print("Bảng so sánh số lượng tham số (Model Capacity)")
        print("===================================================================")
        print(f"Patch + My MSA      : {params_patch_my_msa:,} tham số")
        print(f"Patch + PyTorch MSA : {params_patch_pytorch_msa:,} tham số")
        print(f"Row + My MSA        : {params_row_my_msa:,} tham số")
        print(f"Row + PyTorch MSA   : {params_row_pytorch_msa:,} tham số")
        print("===================================================================")

        # Đánh giá độ chính xác trên tập Test
        acc_patch_my_msa = evaluate_model(model_patch_my_msa, test_loader)
        acc_patch_pytorch_msa = evaluate_model(model_patch_pytorch_msa, test_loader)
        acc_row_my_msa = evaluate_model(model_row_my_msa, test_loader)
        acc_row_pytorch_msa = evaluate_model(model_row_pytorch_msa, test_loader)

        print("\n===================================================================")
        print("Bảng đánh giá độ chính xác (Accuracy) trên tập Test")
        print("===================================================================")
        print(f"Patch + My MSA      : {acc_patch_my_msa * 100:.2f}%")
        print(f"Patch + PyTorch MSA : {acc_patch_pytorch_msa * 100:.2f}%")
        print(f"Row + My MSA        : {acc_row_my_msa * 100:.2f}%")
        print(f"Row + PyTorch MSA   : {acc_row_pytorch_msa * 100:.2f}%")
        print("===================================================================")

        # Vẽ và lưu bảng tổng hợp số liệu
        table_data = [
            ["Mô hình", "Số lượng tham số", "Độ chính xác (Accuracy)"],
            ["Patch + My MSA", f"{params_patch_my_msa:,}", f"{acc_patch_my_msa * 100:.2f}%"],
            ["Patch + PyTorch MSA", f"{params_patch_pytorch_msa:,}", f"{acc_patch_pytorch_msa * 100:.2f}%"],
            ["Row + My MSA", f"{params_row_my_msa:,}", f"{acc_row_my_msa * 100:.2f}%"],
            ["Row + PyTorch MSA", f"{params_row_pytorch_msa:,}", f"{acc_row_pytorch_msa * 100:.2f}%"]
        ]
        
        plot_summary_table(table_data)

        # Tạo biểu đồ so sánh đường học (Learning Curves) giữa các mô hình
        print("\nĐang vẽ biểu đồ so sánh Loss và Accuracy giữa các mô hình")
        plot_learning_curves()

        # Vẽ và lưu biểu đồ cột Accuracy
        models_name = ['Patch MyMSA', 'Patch PyTorchMSA', 'Row MyMSA', 'Row PyTorchMSA']
        accuracies = [acc_patch_my_msa * 100, acc_patch_pytorch_msa * 100, acc_row_my_msa * 100, acc_row_pytorch_msa * 100]

        plot_accuracy_comparison(models_name, accuracies)

if __name__ == "__main__":
    main()