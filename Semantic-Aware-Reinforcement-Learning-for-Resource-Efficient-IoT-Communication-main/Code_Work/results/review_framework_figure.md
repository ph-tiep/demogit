# Nhận xét: Kiểm tra sự khớp giữa hình Framework và Paper

**Tệp paper:** `results/paper.tex`  
**Hình ảnh:** Proposed Framework Overview (3 stage)  
**Ngày kiểm tra:** 2026-06-10

---

## Tổng quan

Luồng tổng thể 3 stage trong hình khớp đúng với mô tả trong paper. Tuy nhiên có **3 điểm không khớp** cần sửa, trong đó 1 điểm nghiêm trọng ảnh hưởng đến tính chính xác kỹ thuật.

---

## Lỗi 1 — Ký hiệu sai trong hộp EKF Predict (NGHIÊM TRỌNG)

**Vị trí trong hình:** Hộp "EKF Predict & Update", dòng Predict

| | Nội dung |
|---|---|
| **Hình hiện tại** | $\hat{x}_{t\|t-1} = F\hat{x}_{t-1}$ |
| **Paper (eq. 6)** | $\hat{\mathbf{s}}_{t\|t-1} = f(\hat{\mathbf{s}}_{t-1\|t-1})$ |

**Hai lỗi trong một dòng:**

1. **Sai ký hiệu biến:** Hình dùng `x̂` (thường ký hiệu cho input/feature), trong khi paper dùng `ŝ` để chỉ *state estimate* của EKF. Cần thống nhất với eq. (6) trong paper.

2. **Sai loại mô hình:** Hình viết `F·` (nhân ma trận tuyến tính), ngụ ý đây là **Kalman Filter tuyến tính thông thường**. Trong khi paper định nghĩa hàm chuyển trạng thái **phi tuyến** với tanh (eq. 3–4):

$$s^{(1)}_i(t+1) = s^{(1)}_i(t) + \alpha\tanh\!\left(\frac{\mu_i - s^{(1)}_i(t)}{\sigma_i}\right) + w^{(1)}_t$$

   Đây là lý do mô hình được gọi là **Extended** Kalman Filter (EKF) — phải linearize qua Jacobian $\mathbf{F}_t$ (eq. 5). Viết `F·` (tuyến tính) phủ nhận tính chất EKF cốt lõi này.

**Cách sửa:**
```
Predict: ŝ_{t|t-1} = f(ŝ_{t-1|t-1})
```

---

## Lỗi 2 — Xung đột ký hiệu `z` giữa Stage 1 và Stage 2

**Vị trí trong hình:** Box "Latent Code" (Stage 1) và box "Observation seq." (Stage 2)

| Stage | Ký hiệu trong hình | Ý nghĩa |
|---|---|---|
| Stage 1 | $\mathbf{z} \in \mathbb{R}^{d_z}$ | Latent code của VAE |
| Stage 2 | $\mathbf{z}_t = [\text{RSSI}_t, n_t^{BS}]$ | Chuỗi quan sát EKF |

**Vấn đề:** Cùng ký hiệu `z` được dùng cho hai đại lượng hoàn toàn khác nhau trong cùng một hình, gây nhầm lẫn cho người đọc. Lỗi này tồn tại cả trong paper (section III.A dùng `z` cho VAE, section III.B dùng `z_i(t)` cho EKF observation).

**Cách sửa đề xuất cho hình:**  
Đổi ký hiệu observation ở Stage 2 thành một trong các lựa chọn sau:
- `o_t = [RSSI_t, n_t^BS]` (observation)
- `y_t = [RSSI_t, n_t^BS]` (measurement, phổ biến trong tài liệu KF)
- `s_t = [RSSI_t, n_t^BS]` (nhất quán với eq. 1 trong paper)

> **Lưu ý:** Nếu sửa trong hình, nên sửa đồng thời trong paper để tránh mâu thuẫn ký hiệu.

---

## Lỗi 3 — Ký hiệu Update thiếu chỉ số thời gian prior (NHỎ)

**Vị trí trong hình:** Hộp "EKF Predict & Update", dòng Update

| | Nội dung |
|---|---|
| **Hình hiện tại** | $K_t = P_t H^T S_t^{-1}$ |
| **Paper (eq. 8)** | $\mathbf{K}_t = \mathbf{P}_{t\|t-1}\mathbf{H}^\top\mathbf{S}_t^{-1}$ |

**Vấn đề:** `P_t` trong hình không phân biệt được là:
- `P_{t|t-1}`: ma trận hiệp phương sai **prior** (trước khi update) — đúng trong công thức Kalman gain
- `P_{t|t}`: ma trận hiệp phương sai **posterior** (sau khi update)

Việc dùng đúng `P_{t|t-1}` là quan trọng về mặt ký hiệu EKF chuẩn.

**Cách sửa:**
```
Update: K_t = P_{t|t-1} H^T S_t^{-1}
```

---

## Các phần khớp đúng

| Thành phần | Hình | Paper | Trạng thái |
|---|---|---|---|
| VAE compression ratio Sigfox | 84 → 16 | Section III.A, Table I | Khớp |
| VAE compression ratio LoRaWAN/WiFi | 5 → 3 | Section III.A, Table I | Khớp |
| Quality Label | Youden's J threshold τ* | eq. (11) | Khớp |
| Anomaly Flag | NIS > χ²_{0.95} | eq. trong Section III.B | Khớp |
| GAT attention coefficient | α_{ij} | eq. (10) | Khớp |
| Resource Actions | a_t ∈ {1...5} | eq. (C.4), Table I (A=5) | Khớp |
| Reward | r_i(t) | eq. (12) | Khớp |
| Agent Graph | G = (V, E) | Section III.C | Khớp |
| Semantic State feedback | Dashed line z → Agent Graph | Section III.C ("local state is VAE latent code") | Khớp |
| Luồng tổng thể 3 stage | VAE → EKF → MADRL-GAT | Section III, Algorithm 1 | Khớp |

---

## Tóm tắt các thay đổi cần thực hiện

| Ưu tiên | Vị trí | Sửa từ | Sửa thành |
|---|---|---|---|
| Cao | EKF Predict, biến state | `x̂_{t\|t-1} = Fx̂_{t-1}` | `ŝ_{t\|t-1} = f(ŝ_{t-1\|t-1})` |
| Trung bình | Observation seq. Stage 2 | `z_t = [RSSI_t, n_t^BS]` | `o_t = [RSSI_t, n_t^BS]` (hoặc `y_t`, `s_t`) |
| Thấp | EKF Update, Kalman gain | `K_t = P_t H^T S_t^{-1}` | `K_t = P_{t\|t-1} H^T S_t^{-1}` |
