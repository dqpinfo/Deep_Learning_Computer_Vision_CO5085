import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataset import get_data_loaders
from models import ImageRNNClassifier

def train_model(rnn_type = 'LSTM', batch_size = 64, epochs = 5, lr = 0.001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Đang huấn luyện mô hình: {rnn_type} trên thiết bị: {device} ---")
    
    train_loader, _ = get_data_loaders(batch_size = batch_size)
    
    model = ImageRNNClassifier(rnn_type = rnn_type, input_size = 28, hidden_size = 128, num_layers = 1, num_classes = 10).to(device)
    
    criterion = nn.CrossEntropyLoss()                       # Sử dụng hàm mất mát CrossEntropyLoss, phù hợp cho bài toán phân loại đa lớp
    optimizer = optim.Adam(model.parameters(), lr = lr)     # Sử dụng Adam Optimizer để cập nhật trọng số của mô hình, với learning rate là lr

    history = {'train_loss': [], 'train_acc': []}           # Lưu trữ lịch sử huấn luyện: loss và accuracy qua các epoch

    for epoch in range(epochs):
        model.train()                                       # Chuyển mô hình sang chế độ huấn luyện (training mode) để bật dropout, batch normalization, v.v.

        running_loss = 0.0                                  # Lưu trữ tổng loss trong epoch
        correct = 0                                         # Lưu trữ số lượng dự đoán đúng trong epoch
        total = 0                                           # Lưu trữ tổng số mẫu trong epoch
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 1. Reset gradient (Đặt gradient về 0)
            optimizer.zero_grad()                           # Đặt gradient về 0 trước khi tính toán gradient mới, tránh cộng dồn gradient từ các batch trước đó
            
            # 2. Forward pass (Lan truyền tiến)
            outputs = model(images)                         # Tính toán đầu ra của mô hình dựa trên ảnh đầu vào, trả về logits cho mỗi lớp nhãn

            # 3. Loss function (Hàm mất mát)
            loss = criterion(outputs, labels)               # Tính toán loss giữa đầu ra dự đoán (logits) và nhãn thực tế
            
            # 4. Backward pass (Lan truyền ngược)
            loss.backward()                                 # Tính toán gradient của loss đối với các tham số của mô hình
            
            # 5. Update weights (Cập nhật trọng số)
            optimizer.step()                                # Cập nhật trọng số của mô hình dựa trên gradient đã tính toán
            
            running_loss += loss.item() * images.size(0)    # Cộng dồn loss của batch hiện tại vào tổng loss của epoch, nhân với số lượng mẫu trong batch để tính trung bình sau này
            _, predicted = torch.max(outputs.data, 1)       # Lấy nhãn dự đoán bằng cách chọn lớp có điểm số cao nhất (argmax) từ logits, trả về chỉ số của lớp dự đoán
            total += labels.size(0)                         # Cập nhật tổng số mẫu đã xử lý trong epoch
            correct += (predicted == labels).sum().item()   # Tính số lượng dự đoán đúng bằng cách so sánh nhãn dự đoán với nhãn thực tế, sau đó cộng dồn vào biến correct
            
        epoch_loss = running_loss / total                   # Tính loss trung bình của epoch bằng cách chia tổng loss cho tổng số mẫu
        epoch_acc = 100.0 * correct / total                 # Tính accuracy trung bình của epoch bằng cách chia số lượng dự đoán đúng cho tổng số mẫu
        
        history['train_loss'].append(epoch_loss)            # Lưu trữ loss trung bình của epoch vào lịch sử huấn luyện
        history['train_acc'].append(epoch_acc)              # Lưu trữ accuracy trung bình của epoch vào lịch sử huấn luyện
        
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    
    return model, history

if __name__ == '__main__':
    output_dir = Path("./output")
    output_dir.mkdir(parents = True, exist_ok = True)

    all_histories = {}

    # Huấn luyện LSTM và lưu vào output/lstm_model.pth
    rrn_type = 'LSTM'
    trained_model, history = train_model(rnn_type = rrn_type, batch_size = 64, epochs = 5, lr = 0.001)
    all_histories[rrn_type] = history

    # Lưu trọng số của mô hình đã huấn luyện
    torch.save(trained_model.state_dict(), output_dir / f"{rrn_type.lower()}_weights.pth")
    print(f"\nĐã lưu file trọng số: {output_dir / f'{rrn_type.lower()}_weights.pth'}")
    
    # Huấn luyện GRU và lưu vào output/gru_model.pth
    rrn_type = 'GRU'
    trained_model, history = train_model(rnn_type = rrn_type, batch_size = 64, epochs = 5, lr = 0.001)
    all_histories[rrn_type] = history

    # Lưu trọng số của mô hình đã huấn luyện
    torch.save(trained_model.state_dict(), output_dir / f"{rrn_type.lower()}_weights.pth")
    print(f"\nĐã lưu file trọng số: {output_dir / f'{rrn_type.lower()}_weights.pth'}")

    # Lưu toàn bộ lịch sử huấn luyện ra file JSON
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(all_histories, f)
    print("\nĐã lưu lịch sử huấn luyện vào file: training_history.json")