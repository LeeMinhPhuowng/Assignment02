# Ba hệ thống thông minh ứng dụng Machine Learning

Dự án này hiện thực hóa toàn diện quy trình phát triển và triển khai ba hệ thống học máy (Machine Learning) thông minh độc lập, giải quyết ba bài toán thực tế thuộc các lĩnh vực: **Y tế (Healthcare)**, **Bất động sản (Real Estate)** và **Thương mại điện tử (E-Commerce)**.

Mỗi ứng dụng tuân thủ nghiêm ngặt chu trình kỹ thuật chuẩn từ dữ liệu thô đến dịch vụ hoàn chỉnh:
$$\text{Raw Data} \longrightarrow \text{Clean} \longrightarrow \text{Represent} \longrightarrow \text{Learn} \longrightarrow \text{Evaluate} \longrightarrow \text{Persist} \longrightarrow \text{Deploy (Web API \& Mobile)}$$

---

## 1. Tổng quan 3 ứng dụng

| Ứng dụng | Bài toán | Tập dữ liệu | Mô hình tối ưu được chọn | Các mô hình được thử nghiệm khác | REST API Endpoint |
|---|---|---|---|---|---|
| **Diabetes** | Phân loại nhị phân nguy cơ đái tháo đường | Pima Indians Diabetes ($N=768$) | **Decision Tree Classifier** | Random Forest, SVM (Linear, RBF), KNN, Logistic Regression | `POST /diabetes/v1/predict` |
| **House Price** | Hồi quy định giá bất động sản (tỷ VNĐ) | Vietnam Housing 2024 ($N=30,229$) | **Random Forest Regressor** | Gradient Boosting, Decision Tree, Ridge, Linear Regression | `POST /house-price/v1/predict` |
| **Customer Behavior** | Xử lý ngôn ngữ tự nhiên (NLP) phân loại review | Women's Clothing Reviews ($N=23,486$) | **Logistic Regression (Balanced)** | Multinomial Naive Bayes, Linear SVM, SGD, Random Forest, Decision Tree | `POST /recommendation/v1/predict` |

### Lý do lựa chọn các mô hình trên
Mô hình triển khai chính thức phải được lựa chọn và biện luận dựa trên sự cân bằng giữa 5 tiêu chí:
1. **Diabetes $\rightarrow$ Chọn Decision Tree Classifier:**  
   Trong y tế, chỉ số **Recall** (độ nhạy) là tối quan trọng để giảm thiểu ca bỏ sót bệnh (False Negative). Decision Tree đạt Recall ($0.722$) và F1-score ($0.716$) cao nhất trên tập kiểm thử, đồng thời sở hữu **tính minh bạch (Interpretability)** vượt trội giúp bác sĩ giải thích rõ ràng căn nguyên quyết định.
2. **House Price $\rightarrow$ Chọn Random Forest Regressor:**  
   Đạt sai số toàn phương trung bình thấp nhất (**RMSE = 1.8478 tỷ VNĐ**) và hệ số xác định cao nhất (**$R^2 = 0.2997$**). Kỹ thuật Ensemble Bagging giúp mô hình bền vững trước các ngoại lai và có tốc độ suy luận thực tế nhanh hơn Gradient Boosting ($0.4$s vs $1.3$s).
3. **Customer Behavior $\rightarrow$ Chọn Logistic Regression (Balanced):**  
   Đạt diện tích dưới đường cong ROC cao nhất (**ROC-AUC = 0.940**). Nhờ tham số `class_weight='balanced'`, mô hình nhận diện chính xác $84.6\%$ số review tiêu cực (lớp thiểu số), kết hợp tốc độ suy luận ma trận thưa TF-IDF siêu tốc ($< 1$ms) và trích xuất trực tiếp được trọng số của từng từ vựng.

---

## 2. Cấu trúc dự án

```text
├── diabetes/
│   ├── diabetes.csv
│   ├── Diabets_as2.ipynb
│   ├── REST_API.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── preprocessor.sav
│   │   └── decision_tree.sav, random_forest.sav, ...
│   ├── web/
│   │   └── index.html
│   └── mobile/
│       ├── pubspec.yaml
│       └── lib/main.dart
│
├── house_price/
│   ├── house_prices.csv
│   ├── House_prices.ipynb
│   ├── REST_API.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── preprocessor.sav
│   │   └── random_forest_regressor.sav, gradient_boosting_regressor.sav, ...
│   ├── web/
│   │   └── index.html
│   └── mobile/
│       ├── pubspec.yaml
│       └── lib/main.dart
│
├── customer_behavior/
│   ├── Womens Clothing E-Commerce Reviews.csv
│   ├── Customer_behavior.ipynb
│   ├── REST_API.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── tfidf_vectorizer.sav
│   │   └── logistic_regression.sav, multinomial_naive_bayes.sav, ...
│   ├── web/
│   │   └── index.html
│   └── mobile/
│       ├── pubspec.yaml
│       └── lib/main.dart
│
└── README.md
```

---

## 3. Nguồn dữ liệu & biểu diễn 

### 3.1. Diabetes Prediction
- **Nguồn:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- **Quy mô:** $768$ quan sát lâm sàng, $9$ cột dữ liệu gốc.
- **Danh sách cột trong tệp `diabetes.csv`:**
  1. `Pregnancies`: Số lần mang thai
  2. `Glucose`: Nồng độ đường huyết sau nghiệm pháp dung nạp glucose 2 giờ
  3. `BloodPressure`: Huyết áp tâm trương (mm Hg)
  4. `SkinThickness`: Độ dày nếp gấp da cơ tam đầu (mm)
  5. `Insulin`: Nồng độ insulin huyết thanh sau 2 giờ (mu U/ml)
  6. `BMI`: Chỉ số khối cơ thể (cân nặng tính bằng kg / (chiều cao tính bằng m)²)
  7. `DiabetesPedigreeFunction`: Điểm số phả hệ đái tháo đường di truyền gia đình
  8. `Age`: Tuổi (năm)
  9. `Outcome`: Biến nhãn mục tiêu ($1$: mắc đái tháo đường, $0$: không mắc)
- **Thuộc tính tuyển chọn vào mô hình:** 5 biến lâm sàng có tính khả dụng cao (`Glucose`, `BMI`, `Age`, `Pregnancies`, `DiabetesPedigreeFunction`).
- **Xử lý giá trị vô lý:** Các giá trị `0` ở `Glucose` và `BMI` là phi lý về mặt sinh học $\to$ thay bằng `NaN` và điền giá trị trung vị (`SimpleImputer(strategy='median')`), sau đó chuẩn hóa bằng `StandardScaler()`.
- **Biểu diễn:** Vector số thực $x_i \in \mathbb{R}^5$, ma trận đầu vào mô hình $X \in \mathbb{R}^{B \times 5}$.

### 3.2. House Price Prediction
- **Nguồn:** [Kaggle - Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024).
- **Quy mô:** $30,229$ tin đăng giao dịch bất động sản nhà ở tại Việt Nam, $12$ cột dữ liệu gốc.
- **Danh sách cột trong tệp `house_prices.csv`:**
  1. `Address`: Địa chỉ căn nhà (Đường, Phường/Xã, Quận/Huyện, Tỉnh/Thành phố)
  2. `Area`: Diện tích mặt sàn đất ($m^2$)
  3. `Frontage`: Mặt tiền nhà ($m$)
  4. `Access Road`: Độ rộng đường/ngõ vào nhà ($m$)
  5. `House direction`: Hướng nhà chính
  6. `Balcony direction`: Hướng ban công
  7. `Floors`: Số tầng của căn nhà
  8. `Bedrooms`: Số phòng ngủ
  9. `Bathrooms`: Số phòng tắm/vệ sinh
  10. `Legal status`: Tình trạng pháp lý (Sổ đỏ, sổ hồng...)
  11. `Furniture state`: Tình trạng nội thất (`Basic` hoặc `Full`)
  12. `Price`: Giá bán bất động sản (Biến mục tiêu, đơn vị: tỷ VNĐ)
- **Thuộc tính tuyển chọn vào mô hình:** 5 đặc trưng số (`Area`, `Access Road`, `Floors`, `Bedrooms`, `Bathrooms`) và 1 đặc trưng phân loại (`Furniture state`).
- **Tiền xử lý:** Nhánh số xử lý qua `SimpleImputer(strategy='median')` $\to$ `StandardScaler()`; Nhánh phân loại xử lý qua `SimpleImputer(strategy='most_frequent')` $\to$ `OneHotEncoder()`.
- **Biểu diễn:** Vector số thực $x_i \in \mathbb{R}^7$, ma trận đầu vào $X \in \mathbb{R}^{B \times 7}$.

### 3.3. Customer Behavior & NLP
- **Nguồn:** [Kaggle - Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).
- **Quy mô:** $23,486$ bài đánh giá khách hàng, $11$ cột dữ liệu gốc.
- **Danh sách cột trong tệp `Womens Clothing E-Commerce Reviews.csv`:**
  1. `Unnamed: 0`: Chỉ số dòng gốc (Index)
  2. `Clothing ID`: Mã định danh sản phẩm may mặc
  3. `Age`: Độ tuổi của khách hàng
  4. `Title`: Tiêu đề bài đánh giá
  5. `Review Text`: Nội dung văn bản nhận xét chi tiết của khách hàng
  6. `Rating`: Số sao đánh giá từ khách hàng (1 đến 5 sao)
  7. `Recommended IND`: Biến nhãn mục tiêu ($1$: Khách hàng khuyến nghị mua, $0$: Không khuyến nghị)
  8. `Positive Feedback Count`: Số lượt người dùng khác bấm bình chọn tích cực cho review này
  9. `Division Name`: Nhóm phân phối sản phẩm
  10. `Department Name`: Ngành hàng (Tops, Dresses, Bottoms, Intimate, Jackets...)
  11. `Class Name`: Phân loại chi tiết (Blouses, Knits, Pants, Dresses...)
- **Thuộc tính tuyển chọn vào mô hình NLP:** Ghép nối `Title` và `Review Text`.
- **Đường ống xử lý NLP:** Làm sạch chữ thường, chuẩn hóa regex loại bỏ ký tự lạ, lọc stopwords tiếng Anh, trích xuất unigram và bigram `(1, 2)` với `max_features=3500`.
- **Biểu diễn:** $\text{Raw Text} \to \text{Tokens} \to \text{Token IDs} \to \text{TF-IDF Matrix} \in \mathbb{R}^{B \times 3500}$.
- **Xử lý mất cân bằng nhãn:** Sử dụng trọng số `class_weight='balanced'` để đảm bảo nhận diện chính xác $84.6\%$ review tiêu cực (lớp thiểu số $17.8\%$).

---

## 4. REST API & Web Dashboard

Mỗi ứng dụng tích hợp một Web Server Flask độc lập (mặc định cổng `5000`), cung cấp endpoint kiểm tra liveness `GET /health`, giao diện người dùng trực quan `GET /web` và endpoint suy luận dự đoán:

### 4.1. Diabetes Prediction Service
- **Endpoint:** `POST /diabetes/v1/predict`
- **Request Body mẫu:**
  ```json
  {
    "Glucose": 145.0,
    "BMI": 33.2,
    "Age": 42,
    "Pregnancies": 2,
    "DiabetesPedigreeFunction": 0.55,
    "model": "decision_tree"
  }
  ```
- **Response Body mẫu:**
  ```json
  {
    "status": "success",
    "prediction": "Nguy cơ cao tiểu đường theo mô hình",
    "prediction_class": 1,
    "confidence": 68.5,
    "model": "decision_tree",
    "model_label": "Decision Tree (khuyến nghị)",
    "knowledge": []
  }
  ```
- *(Tùy chọn: Tích hợp Neo4j Knowledge Graph bằng cách khai báo các biến môi trường `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` để trả về lời khuyên dinh dưỡng, lối sống và biến chứng).*

### 4.2. House Price Prediction Service
- **Endpoint:** `POST /house-price/v1/predict`
- **Request Body mẫu:**
  ```json
  {
    "Area": 85.0,
    "Access Road": 4.5,
    "Floors": 3,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Furniture state": "Full",
    "model": "random_forest_regressor"
  }
  ```
- **Response Body mẫu:**
  ```json
  {
    "status": "success",
    "predicted_price": 5.82,
    "predicted_price_display": "5.82 tỷ VNĐ",
    "model": "random_forest_regressor",
    "model_label": "Random Forest Regressor (khuyến nghị)",
    "message": "Giá dự đoán chỉ mang tính tham khảo, không phải giá thẩm định chính thức."
  }
  ```

### 4.3. Customer Behavior Recommendation Service
- **Endpoint:** `POST /recommendation/v1/predict`
- **Request Body mẫu:**
  ```json
  {
    "Title": "Love the fabric and fit!",
    "Review Text": "The material is soft, flattering cut and runs true to size. Highly recommended!",
    "model": "logistic_regression"
  }
  ```
- **Response Body mẫu:**
  ```json
  {
    "status": "success",
    "prediction_class": 1,
    "recommendation": "Có khả năng khuyến nghị sản phẩm",
    "recommendation_probability": 97.5,
    "result_level": "positive",
    "model": "logistic_regression",
    "model_label": "Logistic Regression (khuyến nghị)"
  }
  ```

---

## 5. Hướng dẫn cài đặt & khởi chạy

### 5.1. Khởi tạo môi trường ảo Python
Dự án được kiểm thử và hoạt động tối ưu trên **Python 3.10+** (hỗ trợ Windows, macOS, Linux).

Khuyến nghị tạo môi trường ảo độc lập:
```bash
# Cách 1: Sử dụng standard venv của Python
python -m venv venv

# Kích hoạt trên Windows:
.\venv\Scripts\activate
# Kích hoạt trên Linux/macOS:
source venv/bin/activate
```

Hoặc dùng Conda:
```bash
# Cách 2: Sử dụng Conda
conda create -n ml_deploy python=3.11 -y
conda activate ml_deploy
```

### 5.2. Cài đặt các thư viện phụ thuộc
Cài đặt danh mục thư viện cho các ứng dụng:
```bash
pip install -r diabetes/requirements.txt
pip install -r house_price/requirements.txt
pip install -r customer_behavior/requirements.txt
```
*(Các gói chính bao gồm: `Flask`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyterlab`).*

---

## 6. Tái lập thực nghiệm trên Jupyter Notebook

Mở môi trường Jupyter từ thư mục gốc của dự án:
```bash
jupyter lab
```

Chạy lần lượt ba notebook tương ứng:
1. **Ứng dụng Tiểu đường:** [`diabetes/Diabets_as2.ipynb`](diabetes/Diabets_as2.ipynb) (23 mục, xuất `preprocessor.sav` và 6 mô hình).
2. **Ứng dụng Giá nhà:** [`house_price/House_prices.ipynb`](house_price/House_prices.ipynb) (23 mục, xuất `preprocessor.sav` và 5 mô hình hồi quy).
3. **Ứng dụng Phân tích đánh giá của khách hàng:** [`customer_behavior/Customer_behavior.ipynb`](customer_behavior/Customer_behavior.ipynb) (23 mục, xuất `tfidf_vectorizer.sav` và 6 mô hình phân loại NLP).

> **Nguyên tắc No Data Leakage:**  
> Toàn bộ bộ chuẩn hóa thang đo (`StandardScaler`), vector hóa (`TfidfVectorizer`) và điền giá trị thiếu (`SimpleImputer`) đều chỉ được `fit` trên tập **Train**. Tập kiểm thử và dữ liệu người dùng gửi qua API chỉ áp dụng hàm `.transform()`.

---

## 7. Khởi chạy Backend REST API & Web Dashboard

Mỗi API sử dụng cổng mặc định `5000`. Khi cần kiểm thử ứng dụng nào, di chuyển vào thư mục tương ứng và khởi chạy:

### Khởi chạy Diabetes Service:
```powershell
cd diabetes
python REST_API.py
```
- Mở Dashboard trên trình duyệt: `http://127.0.0.1:5000/web`
- Kiểm tra trạng thái service: `http://127.0.0.1:5000/health`

### Khởi chạy House Price Service:
```powershell
cd house_price
python REST_API.py
```
- Mở Valuation Portal trên trình duyệt: `http://127.0.0.1:5000/web`

### Khởi chạy Customer Behavior Service:
```powershell
cd customer_behavior
python REST_API.py
```
- Mở Review Sentiment Studio trên trình duyệt: `http://127.0.0.1:5000/web`

---

## 8. Hướng dẫn khởi chạy Ứng dụng Flutter Mobile

Mỗi ứng dụng đi kèm một ứng dụng di động Flutter độc lập trong thư mục `mobile/`, giao tiếp với REST API theo mô hình Client - Server:

$$\text{Mobile UI} \longrightarrow \text{HTTP POST} \longrightarrow \text{Flask REST API} \longrightarrow \text{Pipeline \& Model} \longrightarrow \text{JSON Response} \longrightarrow \text{Mobile UI}$$

### Các bước khởi chạy:
```powershell
# Ví dụ chạy ứng dụng di động Diabetes:
cd diabetes/mobile

# Khởi tạo nền tảng nếu cần và tải gói phụ thuộc:
flutter create .
flutter pub get

# Khởi chạy trên thiết bị hoặc máy ảo:
flutter run
```

### Cấu hình địa chỉ IP máy chủ:
- **Trên Android Emulator:** Giữ nguyên URL mặc định: `http://10.0.2.2:5000`.
- **Trên Thiết bị di động thật:** 
  1. Kết nối điện thoại và máy tính vào cùng một mạng Wi-Fi cục bộ.
  2. Mở terminal gõ `ipconfig` (Windows) hoặc `ifconfig` (macOS/Linux) để lấy địa chỉ IPv4 (ví dụ: `192.168.1.15`).
  3. Nhập URL `http://192.168.1.15:5000` vào ô cấu hình REST API trên giao diện ứng dụng.

---

## 9. Khuyến cáo & Phạm vi ứng dụng

- Các kết quả dự đoán được sinh ra từ các mô hình học máy được huấn luyện trên dữ liệu lịch sử mang tính chất tham khảo, học thuật và hỗ trợ ra quyết định.
- Kết quả sàng lọc nguy cơ đái tháo đường **không thay thế kết luận chẩn đoán hoặc chỉ định lâm sàng của y bác sĩ**.
- Dự đoán giá nhà ở dựa trên tập dữ liệu tổng hợp các tin đăng giao dịch và không thay thế cho chứng thư thẩm định giá độc lập có hiệu lực pháp lý.
- Ứng dụng phân tích đánh giá khách hàng phản ánh cảm nhận trên từng bài review cụ thể và không định danh lịch sử dài hạn của từng khách hàng đơn lẻ.
