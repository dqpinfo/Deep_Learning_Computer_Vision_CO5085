| Cấu hình | Chế độ | Trainable | Val acc | Test acc | Macro-F1 | Train (phút) | GPU (GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `resnet18_scratch` | Học từ đầu | 11.227.812 | 68,54 | 67,46 | 67,50 | 37,0 | 1,69 |
| `resnet18_head` | Chỉ train head | 51.300 | 50,82 | 51,02 | 49,24 | 3,9 | 0,60 |
| `resnet18_partial` | Đóng băng một phần | 10.544.740 | 74,42 | 73,67 | 73,42 | 6,0 | 0,72 |
| `resnet18_full` | Fine-tune toàn bộ | 11.227.812 | 74,92 | 74,71 | 74,48 | 12,3 | 1,69 |
| `deit_tiny_scratch` | Học từ đầu | 5.543.716 | 47,86 | 48,48 | 48,01 | 32,9 | 2,23 |
| `deit_tiny_head` | Chỉ train head | 19.300 | 65,62 | 65,11 | 64,72 | 3,9 | 0,25 |
| `deit_tiny_partial` | Đóng băng một phần | 2.688.868 | 80,76 | 81,31 | 81,30 | 7,2 | 1,19 |
| `deit_tiny_full` | Fine-tune toàn bộ | 5.543.716 | 84,94 | 85,02 | 85,04 | 11,0 | 2,23 |
