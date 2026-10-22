############################
#   Đánh giá các mô hình   #
############################

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from dataset import get_data_loaders
from models import SoftmaxClassifier, MLPClassifier, CNNClassifier

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

# Hàm vẽ Confusion Matrix và tìm các mẫu lỗi (Error Analysis)
def analyze_errors_and_confusion_matrix(model, test_loader, model_name = "Model", output_dir = Path("./output")):

    model.eval()

    all_preds = []
    all_labels = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Thu thập toàn bộ dự đoán và nhãn thực tế
            for img, true_lbl, pred_lbl in zip(images, labels, predicted):
                all_preds.append(pred_lbl.item())
                all_labels.append(true_lbl.item())
                
                # Lưu lại một số mẫu dự đoán sai để phân tích lỗi
                if true_lbl.item() != pred_lbl.item() and len(incorrect_samples) < 5:
                    incorrect_samples.append((img, true_lbl.item(), pred_lbl.item()))

    # Tính Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Vẽ biểu đồ Heatmap cho Confusion Matrix
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', cbar = False)
    plt.title(f'Confusion Matrix - {model_name}', fontsize = 14, fontweight='bold')
    plt.xlabel('Dự đoán (Predicted)', fontsize = 11)
    plt.ylabel('Thực tế (Actual)', fontsize = 11)
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name.lower()}.png', dpi = 300)
    print(f"-> Đã lưu Confusion Matrix: confusion_matrix_{model_name.lower()}.png")
    plt.show()
    
    # Vẽ trực quan hóa một số mẫu dự đoán sai
    if incorrect_samples:
        fig, axes = plt.subplots(1, len(incorrect_samples), figsize=(12, 3))
        if len(incorrect_samples) == 1:
            axes = [axes]
        for i, (img, true_l, pred_l) in enumerate(incorrect_samples):
            # Chuyển tensor ảnh về numpy để vẽ (bỏ chiều kênh 1)
            img_np = img.squeeze().numpy()
            axes[i].imshow(img_np, cmap = 'gray')
            axes[i].set_title(f"Thật: {true_l}\nĐoán: {pred_l}", color = 'red', fontsize = 10)
            axes[i].axis('off')
        plt.suptitle(f'Các mẫu dự đoán sai tiêu biểu - {model_name}', fontsize = 12, fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'error_samples_{model_name.lower()}.png', dpi = 300)
        print(f"-> Đã lưu các mẫu lỗi: error_samples_{model_name.lower()}.png")
        plt.show()


def main():
    output_dir = Path("./output")

    if output_dir.is_dir() == False:
        print(f"Folder not found: {output_dir}")
        return
    else:
        # Tải dữ liệu test
        _, test_loader = get_data_loaders(batch_size = 64)
        
        # Khởi tạo và nạp trọng số đã huấn luyện cho các mô hình
        softmax_model = SoftmaxClassifier()
        softmax_model.load_state_dict(torch.load(output_dir / "softmax_weights.pth"))
        
        mlp_model = MLPClassifier()
        mlp_model.load_state_dict(torch.load(output_dir / "mlp_weights.pth"))
        
        cnn_model = CNNClassifier()
        cnn_model.load_state_dict(torch.load(output_dir / "cnn_weights.pth"))
        
        # So sánh số lượng tham số
        params_softmax_model = count_parameters(softmax_model)
        params_mlp_model = count_parameters(mlp_model)
        params_cnn_model = count_parameters(cnn_model)

        print("\n===================================================================")
        print("Bảng so sánh số lượng tham số (Model Capacity)")
        print("===================================================================")
        print(f"Softmax Classifier : {params_softmax_model:,} tham số")
        print(f"MLP Classifier     : {params_mlp_model:,} tham số")
        print(f"CNN Classifier     : {params_cnn_model:,} tham số")
        print("===================================================================")
        
        # Đánh giá độ chính xác trên tập Test
        acc_softmax = evaluate_model(softmax_model, test_loader)
        acc_mlp = evaluate_model(mlp_model, test_loader)
        acc_cnn = evaluate_model(cnn_model, test_loader)

        print("\n===================================================================")
        print("Bảng đánh giá độ chính xác (Accuracy) trên tập Test")
        print("===================================================================")
        print(f"Softmax Test Accuracy : {acc_softmax * 100:.2f}%")
        print(f"MLP Test Accuracy     : {acc_mlp * 100:.2f}%")
        print(f"CNN Test Accuracy     : {acc_cnn * 100:.2f}%")
        print("===================================================================")

        # Vẽ và lưu bảng tổng hợp số liệu
        table_data = [
            ["Mô hình", "Số lượng tham số", "Độ chính xác (Accuracy)"],
            ["Softmax", f"{params_softmax_model:,}", f"{acc_softmax * 100:.2f}%"],
            ["MLP", f"{params_mlp_model:,}", f"{acc_mlp * 100:.2f}%"],
            ["CNN", f"{params_cnn_model:,}", f"{acc_cnn * 100:.2f}%"]
        ]
        
        plot_summary_table(table_data)

        # Tạo biểu đồ so sánh đường học (Learning Curves) giữa các mô hình
        print("\nĐang vẽ biểu đồ so sánh Loss và Accuracy giữa các mô hình")
        plot_learning_curves()
        
        # Phân tích lỗi và Confusion Matrix cho các mô hình Softmax, MLP và CNN
        print("\nTạo Confusion Matrix và biểu đồ phân tích mẫu lỗi cho Softmax")
        analyze_errors_and_confusion_matrix(softmax_model, test_loader, model_name = "Softmax")
        
        print("\nTạo Confusion Matrix và biểu đồ phân tích mẫu lỗi cho MLP")
        analyze_errors_and_confusion_matrix(mlp_model, test_loader, model_name = "MLP")
        
        print("\nTạo Confusion Matrix và biểu đồ phân tích mẫu lỗi cho CNN")
        analyze_errors_and_confusion_matrix(cnn_model, test_loader, model_name = "CNN")

if __name__ == "__main__":
    main()