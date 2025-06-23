# Football-prediction-


-----

## ⚽ **kèo bóng VTT: Trợ lý Phân tích Bóng đá bằng AI** 🚀

Chào mừng bạn đến với **kèo bóng VTT** – một ứng dụng desktop đột phá, được xây dựng bằng Python và PyQt6, mang đến sức mạnh của Trí tuệ Nhân tạo (AI) vào thế giới phân tích bóng đá.

kèo bóng VTT không chỉ là một công cụ, mà là một trợ lý ảo thông minh. Ứng dụng tự động thu thập dữ liệu chuyên sâu từ các nguồn uy tín hàng đầu như **FotMob** và **Transfermarkt**, sau đó sử dụng API **Gemini** mạnh mẽ của Google để đưa ra những phân tích và dự đoán toàn diện. Giao diện trực quan cho phép bạn dễ dàng nhập liệu, phân tích và khám phá mọi góc cạnh của trận đấu, từ thống kê chi tiết đến bản đồ cú sút (shotmap) sống động.

**Mục tiêu của dự án** là trang bị cho người hâm mộ và nhà phân tích một công cụ dựa trên dữ liệu, giúp đưa ra những quyết định sáng suốt và có cái nhìn sâu sắc hơn về môn thể thao vua.

-----

### ✨ **Các Tính Năng Nổi Bật**

| Tính năng | Mô tả chi tiết |
| :--- | :--- |
| 🔍 **Thu thập Dữ liệu Toàn diện & Tự động** | - **FotMob**: Cào dữ liệu trận đấu chi tiết (thống kê, đội hình, diễn biến, lịch sử đối đầu).\<br\>- **Shotmap**: Lấy bản đồ cú sút để phân tích vị trí và hiệu quả dứt điểm.\<br\>- **Transfermarkt Integration**: Thu thập dữ liệu cầu thủ (giá trị, lịch sử chuyển nhượng) - *Sẽ sớm được tích hợp sâu hơn vào luồng phân tích chính.* |
| 🧠 **Phân tích Chuyên sâu với Google Gemini AI** | - **Báo cáo chi tiết**: Tạo ra một bài phân tích chuyên sâu về phong độ, chiến thuật và lịch sử đối đầu.\<br\>- **Dự đoán thông minh**: Đưa ra xác suất Thắng/Hòa/Thua, tổng số bàn thắng kỳ vọng, "best bet" và mức độ tự tin.\<br\>- **Dự đoán Tỷ số**: Liệt kê các kịch bản tỷ số có khả-năng xảy ra nhất kèm theo xác suất. |
| 📊 **Trực quan hóa Dữ liệu Sống động** | - **Biểu đồ Xác suất**: Hiển thị rõ ràng cơ hội chiến thắng của mỗi đội và khả năng hòa.\<br\>- **Shotmap Lịch sử**: Tổng hợp bản đồ cú sút từ các trận đấu gần đây, giúp hình dung xu hướng tấn công và phòng ngự. |
| 🖥️ **Giao diện Desktop Chuyên nghiệp & Thân thiện** | - **PyQt6 Framework**: Giao diện được thiết kế khoa học, dễ sử dụng với các tab chức năng rõ ràng.\<br\>- **Luồng làm việc trực quan**: Các hộp thoại hướng dẫn người dùng nhập URL, chọn đội và nhập kèo nhà cái một cách mượt mà. |

-----

### 🛠️ **Công Nghệ & Nền tảng**

  - **Ngôn ngữ chính**: Python 🐍
  - **Giao diện người dùng (GUI)**: PyQt6 🖼️
  - **Cào dữ liệu động**: Playwright 🎭
  - **Cào dữ liệu tĩnh**: BeautifulSoup & Requests 🌐
  - **Xử lý dữ liệu**: Pandas & NumPy 🐼
  - **Trực quan hóa**: Matplotlib & mplsoccer 📈
  - **Bộ não AI**: Google Gemini API 🤖

-----

### 🚀 **Hướng Dẫn Bắt Đầu**

#### **1️⃣ Bước 1: Thiết lập Môi trường**

Bạn cần cài đặt Python (phiên bản 3.8+ được khuyến nghị) và các thư viện cần thiết.

```bash
# Tạo và kích hoạt môi trường ảo (khuyến khích)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install PyQt6 pandas numpy matplotlib mplsoccer requests beautifulsoup4 playwright google-generativeai

# Cài đặt trình duyệt cho Playwright (khuyến nghị Chromium)
playwright install chromium
```

#### **2️⃣ Bước 2: Chuẩn bị Google Gemini API Key**

Tính năng phân tích AI yêu cầu bạn phải có Google Gemini API Key.

1.  Truy cập **[Google AI Studio](https://aistudio.google.com/)**.
2.  Đăng nhập bằng tài khoản Google của bạn.
3.  Tạo một API Key mới và sao chép nó. 🔑

#### **3️⃣ Bước 3: Chạy Ứng dụng**

1.  Tải về toàn bộ mã nguồn của dự án (`main_app_v2.py` và `football_scraper.py`) và đặt chúng vào cùng một thư mục.
2.  Mở Terminal (hoặc Command Prompt) và điều hướng đến thư mục đó.
3.  Thực thi lệnh sau để khởi chạy ứng dụng:

<!-- end list -->

```bash
python main_app_v2.py
```

#### **4️⃣ Bước 4: Trải nghiệm Phân tích**

1.  **Nhập API Key**: Lần đầu khởi động, vào menu **Cài đặt** -\> **Nhập Gemini API Key** và dán khóa của bạn vào.
2.  **Bắt đầu Phân tích Mới**:
      * Chọn **Hành động** -\> **Bắt đầu Phân tích Mới**.
      * Nhập URL của 2 trận đấu gần nhất từ FotMob để ứng dụng thu thập dữ liệu.
      * Chọn đội nhà và đội khách bạn muốn phân tích.
      * Nhập các tỷ lệ kèo (Châu Âu, Châu Á, Tài Xỉu) để AI có thêm ngữ cảnh phân tích.
3.  **Xem Kết quả**:
      * **Dữ liệu thô**: Xem lại toàn bộ dữ liệu đã được thu thập.
      * **Phân tích AI**: Đọc bài phân tích và các dự đoán chi tiết từ Gemini.
      * **Trực quan hóa AI**: Xem biểu đồ xác suất và các thông tin dự đoán quan trọng.
      * **Shotmap Lịch sử**: Khám phá bản đồ cú sút của hai đội.

-----

### ⚠️ **Tuyên bố Miễn trừ Trách nhiệm Quan trọng**

**kèo bóng VTT** là một dự án nghiên cứu và phát triển cá nhân, được tạo ra với mục đích giáo dục và khám phá tiềm năng của AI trong phân tích thể thao. Ứng dụng này **KHÔNG** khuyến khích, cổ súy hay cung cấp công cụ cho các hoạt động cá cược dưới mọi hình thức.

Mọi quyết định tài chính dựa trên thông tin từ ứng dụng là **trách nhiệm hoàn toàn của người dùng**. Tác giả và những người đóng góp cho dự án không chịu trách nhiệm cho bất kỳ tổn thất hay hậu quả nào phát sinh từ việc sử dụng phần mềm sai mục đích. Hãy sử dụng một cách có trách nhiệm.
