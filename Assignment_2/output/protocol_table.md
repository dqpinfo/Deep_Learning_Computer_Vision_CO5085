| Mô hình | Encoder | Tổng tham số | Encoder | Decoder | Gradient | Loss đầu -> cuối |
|---|---|---:|---:|---:|---|---|
| unet_r50 | resnet50.a1_in1k | 32,521,606 | 23,508,032 | 9,013,574 | OK | 3.811 -> 3.080 |
| segformer_b2 | nvidia/mit-b2 | 27,375,846 | 24,196,288 | 3,179,558 | OK | 3.686 -> 0.048 |
