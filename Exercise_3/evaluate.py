import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_data_loaders
from models import ImageRNNClassifier

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tải dữ liệu test
        train_loader, test_loader = get_data_loaders(batch_size = 64)

        # Khởi tạo và nạp trọng số đã huấn luyện cho các mô hình
        lstm_model = ImageRNNClassifier(rnn_type = 'LSTM', input_size = 28, hidden_size = 128, num_classes = 10).to(device)
        lstm_model.load_state_dict(torch.load(output_dir / "lstm_weights.pth"))

        gru_model = ImageRNNClassifier(rnn_type = 'GRU', input_size = 28, hidden_size = 128, num_classes = 10).to(device)
        gru_model.load_state_dict(torch.load(output_dir / "gru_weights.pth"))

        # So sánh số lượng tham số
        params_mlp = 109386
        params_cnn = 105962
        params_lstm = count_parameters(lstm_model)
        params_gru = count_parameters(gru_model)

        print("\n===================================================================")
        print("Bảng so sánh số lượng tham số (Model Capacity)")
        print("===================================================================")
        print(f"MLP Classifier  (E1)    : {params_mlp:,} tham số")
        print(f"CNN Classifier  (E1)    : {params_cnn:,} tham số")
        print(f"LSTM Classifier (E3)    : {params_lstm:,} tham số")
        print(f"GRU Classifier  (E3)    : {params_gru:,} tham số")
        print("===================================================================")

        # Đánh giá độ chính xác trên tập Test
        acc_mlp = 0.9697
        acc_cnn = 0.9891
        acc_lstm = evaluate_model(lstm_model, test_loader)
        acc_gru = evaluate_model(gru_model, test_loader)

        print("\n===================================================================")
        print("Bảng đánh giá độ chính xác (Accuracy) trên tập Test")
        print("===================================================================")
        print(f"MLP Test Accuracy  (E1)  : {acc_mlp * 100:.2f}%")
        print(f"CNN Test Accuracy  (E1)  : {acc_cnn * 100:.2f}%")
        print(f"LSTM Test Accuracy (E3)  : {acc_lstm * 100:.2f}%")
        print(f"GRU Test Accuracy  (E3)  : {acc_gru * 100:.2f}%")
        print("===================================================================")

        # Vẽ và lưu bảng tổng hợp số liệu
        table_data = [
            ["Mô hình", "Số lượng tham số", "Độ chính xác (Accuracy)"],
            ["MLP  (E1)", f"{params_mlp:,}", f"{acc_mlp * 100:.2f}%"],
            ["CNN  (E1)", f"{params_cnn:,}", f"{acc_cnn * 100:.2f}%"],
            ["LSTM (E3)", f"{params_lstm:,}", f"{acc_lstm * 100:.2f}%"],
            ["GRU  (E3)", f"{params_gru:,}", f"{acc_gru * 100:.2f}%"]
        ]
        
        plot_summary_table(table_data)

        # Tạo biểu đồ so sánh đường học (Learning Curves) giữa các mô hình
        print("\nĐang vẽ biểu đồ so sánh Loss và Accuracy giữa các mô hình")
        plot_learning_curves()
        
        # Phân tích lỗi và Confusion Matrix cho các mô hình LSTM và GRU
        print("\nTạo Confusion Matrix và biểu đồ phân tích mẫu lỗi cho LSTM")
        analyze_errors_and_confusion_matrix(lstm_model, test_loader, model_name = "LSTM")

        print("\nTạo Confusion Matrix và biểu đồ phân tích mẫu lỗi cho GRU")
        analyze_errors_and_confusion_matrix(gru_model, test_loader, model_name = "GRU")

if __name__ == '__main__':
    main()