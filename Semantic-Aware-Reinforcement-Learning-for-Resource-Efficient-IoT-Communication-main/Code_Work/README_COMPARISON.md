# 🔬 Baseline vs VAE Comparison

## 📋 Tổng Quan

Dự án này so sánh hai phương pháp xử lý ngữ nghĩa cho IoT communication:

1. **BASELINE**: Sử dụng features gốc (5 chiều)
2. **VAE**: Sử dụng Variational Autoencoder để nén features (5D → 3D)

## 🏗️ Cấu Trúc Thư Mục

```
Code_Work/
├── main.py                    # Pipeline baseline (5D features)
├── main_vae.py                # Pipeline với VAE compression (3D)
├── vae_model.py               # Implementation VAE
├── compare_results.py         # So sánh kết quả
├── run_all.py                 # Chạy cả 2 pipelines + so sánh
└── results/
    ├── baseline/              # Kết quả baseline
    ├── vae/                   # Kết quả VAE
    └── comparison/            # So sánh trực quan
```

## 🚀 Cách Chạy

### Option 1: Chạy Tất Cả (Khuyến Nghị)

```bash
python run_all.py
```

Script này sẽ:
1. Chạy baseline pipeline
2. Chạy VAE pipeline  
3. So sánh và tạo visualization

### Option 2: Chạy Từng Bước

```bash
# Bước 1: Chạy baseline
python main.py

# Bước 2: Chạy VAE
python main_vae.py

# Bước 3: So sánh kết quả
python compare_results.py
```

## 🔬 VAE - Variational Autoencoder

### Tại Sao Dùng VAE?

- **Nén dữ liệu**: Giảm từ 5D xuống 3D (compression ratio 1.67x)
- **Denoising**: Loại bỏ noise trong dữ liệu
- **Learned representation**: Học được latent space có ý nghĩa
- **Tăng tốc training**: Ít features hơn = training nhanh hơn
- **Regularization**: Giảm overfitting

### Kiến Trúc VAE

```
Input (5D) → Encoder (64→32) → Latent Space (3D)
                                      ↓
Output (5D) ← Decoder (64→32) ← Latent Space (3D)
```

### Hyperparameters

- **Latent dimension**: 3D
- **Epochs**: 50
- **Batch size**: 64
- **Learning rate**: 0.001
- **Loss**: Reconstruction MSE + KL Divergence

## 📊 So Sánh

### Baseline (5D Features)

```python
Features: [mean_rssi, num_active_bs, Latitude, Longitude, hour]
Dimension: 5D
Preprocessing: StandardScaler normalization
```

### VAE (3D Latent)

```python
Original: [mean_rssi, num_active_bs, Latitude, Longitude, hour]
Compressed: [latent_dim_1, latent_dim_2, latent_dim_3]
Dimension: 3D
Compression: 5D → 3D (1.67x)
```

## 📈 Metrics Để So Sánh

1. **Classification Accuracy**: Độ chính xác của Random Forest
2. **ROC-AUC Score**: Khả năng phân biệt của model
3. **Anomaly Detection**: Hiệu quả phát hiện bất thường
4. **Training Time**: Thời gian training DQN
5. **Feature Quality**: Quality của latent representations

## 🎯 Kết Quả Mong Đợi

### VAE có thể:
- ✅ Giảm dimensionality mà vẫn giữ được performance
- ✅ Tăng tốc độ training DQN
- ✅ Cải thiện generalization
- ✅ Tạo ra meaningful latent space

### Tradeoffs:
- ⚠️ Có thể mất một ít thông tin trong quá trình nén
- ⚠️ Cần thêm bước training VAE
- ⚠️ Phức tạp hơn về implementation

## 📁 Output Files

### Baseline (`results/baseline/`)
- `confusion_matrix.pdf`
- `feature_importance.pdf`
- `anomaly_map.pdf`
- `RFroc_curve.pdf`
- `IFroc_curve.pdf`
- `combined_ROC_curves_with_labels.pdf`

### VAE (`results/vae/`)
- Tất cả files như baseline, plus:
- `vae_training_loss.pdf` - VAE training progress
- `vae_latent_space.pdf` - Visualization của latent space

### Comparison (`results/comparison/`)
- `comparison_*.png` - Side-by-side comparisons

## 🔧 Requirements

Tất cả dependencies giống nhau:
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn shap lime
```

## 💡 Tips

1. **Điều chỉnh latent dimension**: Thử nghiệm với `latent_dim=2` hoặc `latent_dim=4`
2. **VAE epochs**: Tăng epochs nếu muốn VAE học tốt hơn
3. **Batch size**: Điều chỉnh dựa trên RAM available
4. **Device**: Code tự động dùng GPU nếu có, CPU nếu không

## 🎓 Kiến Thức Thêm

### VAE là gì?
Variational Autoencoder là một generative model học cách:
- Encode dữ liệu vào latent space
- Decode từ latent space về original space
- Regularize latent space bằng KL divergence

### Khi nào nên dùng VAE?
- High-dimensional data cần compression
- Muốn learn meaningful representations
- Cần denoising
- Muốn generate new samples

## 📚 References

- VAE Paper: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)
- DQN Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

## 🐛 Troubleshooting

### Lỗi: Out of Memory
```bash
# Giảm batch size trong vae_model.py
train_vae(..., batch_size=32)
```

### Lỗi: VAE không converge
```bash
# Tăng số epochs
train_vae(..., epochs=100)
```

### Lỗi: Poor reconstruction
```bash
# Tăng latent dimension
VAE(input_dim=5, latent_dim=4)
```

## ✅ Checklist

- [x] Tạo cấu trúc thư mục results/
- [x] Implement VAE model
- [x] Tích hợp VAE vào pipeline
- [x] Tạo comparison script
- [x] Tạo run_all script
- [x] Documentation

---

**Tác giả**: Updated for VAE comparison experiment  
**Ngày**: 2026  
**Version**: 2.0 (VAE-enhanced)
