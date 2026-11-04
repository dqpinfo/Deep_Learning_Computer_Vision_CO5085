import torch
import torch.nn as nn

class ImageRNNClassifier(nn.Module):
    def __init__(self, rnn_type='LSTM', input_size=28, hidden_size=128, num_layers=1, num_classes=10, sequence_mode='row'):
        super(ImageRNNClassifier, self).__init__()
        self.rnn_type = rnn_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_mode = sequence_mode # 'row' (theo hàng) hoặc 'col' (theo cột)
        
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type phải là 'LSTM' hoặc 'GRU'")
            
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x ban đầu có kích thước: (Batch, Channels, Height, Width) -> (Batch, 1, 28, 28)
        batch_size = x.size(0)
        
        if self.sequence_mode == 'col':
            # Quét theo cột: Hoán đổi chiều Height và Width (dim 2 và dim 3)
            # Sau khi permute: (Batch, 1, Width, Height), lúc này mỗi cột trở thành 1 time-step
            x = x.permute(0, 1, 3, 2)
        
        # Loại bỏ kênh màu (channel = 1) để đưa về tensor 3D: (Batch, Seq_Len, Input_Dim)
        x = x.squeeze(1) 
        
        # Khởi tạo hidden state
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            out, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            out, hn = self.rnn(x, h0)
            
        # Lấy trạng thái ẩn ở bước thời gian cuối cùng để phân loại
        out = self.fc(hn[-1])
        return out