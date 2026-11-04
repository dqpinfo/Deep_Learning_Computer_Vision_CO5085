import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_data_loaders
from models import ImageRNNClassifier

SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)

DIGIT_CLASSES = [str(i) for i in range(10)]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_sequence_representation(image_tensor, label_name="Sample"):
    img = image_tensor.squeeze().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"MNIST Digit ({label_name})")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="magma")
    axes[1].axhline(y=14, color="cyan", linestyle="--", linewidth=2)
    axes[1].set_title("Row-wise Sequence\n(Time-step = Row)")
    axes[1].set_ylabel("Time Steps (0-27)")
    axes[1].set_xlabel("Features / Pixels (28)")

    axes[2].imshow(img.T, cmap="magma")
    axes[2].axvline(x=14, color="cyan", linestyle="--", linewidth=2)
    axes[2].set_title("Column-wise Sequence\n(Time-step = Column)")
    axes[2].set_xlabel("Features / Pixels (28)")
    axes[2].set_ylabel("Time Steps (0-27)")

    plt.tight_layout()
    
    # Show ảnh ra cửa sổ popup trước
    plt.show()
    
    # Sau đó lưu file vào thư mục output
    output_path = os.path.join(SAVE_DIR, "e3_mnist_sequence_representation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ mô tả chuỗi tại: {output_path}")

def plot_confusion_matrix(y_true, y_pred, model_name="LSTM"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES)
    plt.title(f"MNIST Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.tight_layout()
    
    # Show cửa sổ popup chứa ma trận nhầm lẫn
    plt.show()
    
    # Lưu file vào thư mục output
    output_path = os.path.join(SAVE_DIR, f"e3_mnist_confusion_matrix_{model_name.lower()}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Confusion Matrix của {model_name} vào thư mục '{SAVE_DIR}' thành công!")

def plot_baseline_comparison(results_dict):
    models = list(results_dict.keys())
    accuracies = list(results_dict.values())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, accuracies, color=["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e"])
    plt.title("MNIST Performance Comparison (Baseline E1 vs RNN E3)")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(80, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Show cửa sổ popup biểu đồ so sánh
    plt.show()
    
    # Lưu file vào thư mục output
    output_path = os.path.join(SAVE_DIR, "e3_mnist_baseline_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ so sánh Baseline vào thư mục '{SAVE_DIR}' thành công!")

def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()
    
    sample_img, sample_label = next(iter(train_loader))
    plot_sequence_representation(sample_img[0], label_name=DIGIT_CLASSES[sample_label[0]])
    
    test_acc_results = {}
    
    for rnn_type in ['LSTM', 'GRU']:
        model = ImageRNNClassifier(rnn_type=rnn_type, input_size=28, hidden_size=128, num_classes=10).to(device)
        
        weight_path = os.path.join("output", f"{rnn_type.lower()}_model.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            print(f"[!] Cảnh báo: Không tìm thấy file {weight_path}. Đang đánh giá với mô hình chưa train.")
            
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = (correct / total) * 100
        params = count_parameters(model)
        test_acc_results[f"{rnn_type} (E3)"] = acc
        
        print(f"\n========================================")
        print(f" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH: {rnn_type}")
        print(f"========================================")
        print(f" - Tổng số lượng tham số: {params:,}")
        print(f" - Độ chính xác Test Accuracy: {acc:.2f}%")
        print("\nBáo cáo phân loại chi tiết (Classification Report):")
        print(classification_report(all_labels, all_preds, target_names=DIGIT_CLASSES))
        
        plot_confusion_matrix(all_labels, all_preds, model_name=rnn_type)
        
    test_acc_results['MLP (E1)'] = 97.80
    test_acc_results['CNN (E1)'] = 99.10
    
    plot_baseline_comparison(test_acc_results)

if __name__ == '__main__':
    evaluate_models()