# Quan hệ giữa dw và w, db và b
## Giải thích về tham số 
dw (derivative of w): Là độ dốc của hàm Loss theo trọng số w.
db (derivative of b): Là độ dốc của hàm Loss theo bias b.
Vai trò: Chúng đóng vai trò là chiếc "La bàn" chỉ đường cho mô hình biết phải đi về đâu để giảm sai số.
w và b: Giải thích ở phần **Về công thức đưa ra quyết định dự đoán z=w​.x ​+ b**
* learning rate (alpha): là tham số do người lập trình chọn để quyết định độ lớn của bước nhảy trong mỗi lần cập nhập w và b.
Nếu quá lớn thì sai số sẽ không giảm mà liên tục biến động. (ví dụ sai số đang là 0.2 nhưng learning rate quá cao làm sai số giảm 0.4 nên không thể giảm sai số về 0)
Nếu quá nhỏ thì Tốn rất nhiều thời gian và tài nguyên máy tính để giảm sai số.
## Cơ chế
**dw > 0 thì giảm w để sai số giảm**
dw dương => y predict > y => cần giảm y predict để tiến tới y => cần giảm w => công thức w_update = w_old - alpha*(y^-y)*x (y tỉ lệ thuận với w)
Mà dw = (y^-y).x => w_update = w_old - alpha*dw => dw dương thì w_update sẽ giảm 
**dw < 0 thì tăng w để sai số giảm**
dw âm => y predict < y => cần tăng y predict để tiến tới y => cần tăng w => công thức w_update = w_old - alpha*(y^-y)*x (y tỉ lệ thuận với w)
Mà dw = (y^-y).x => w_update = w_old - alpha*dw => dw âm thì w_update sẽ tăng
**Tương tự với db và b**

# Giải thích về hàm Loss L(y,y^​)=−[ y.log(y^​) + (1−y).log(1−y^​) ]
## Giải thích về tham số
* y: Đáp án đúng (Chỉ có 0 hoặc 1).
* y mũ (y_predicted): Dự đoán của máy (Xác suất từ 0 đến 1).
* log: Logarit tự nhiên (ln).
## Cơ chế công tắc là lí do tại sao mô hình thích hợp cho dự đoán boolen 
* Nếu thực tế là Ác tính (y=1):
Phần sau: (1-1)log(...) = 0log(...) = 0. (Biến mất).
Công thức chỉ còn: Loss = - log(y mũ)
Nếu y mũ gần = 1 => dự đoán đúng => Loss = - log(gần bằng 1) = gần bằng 0 => mô hình sẽ tự hiểu là đang đúng 
Ngược lại y gần = 0 (Ác tính nhưng dự đoán là lành tính) => Loss = - log(gần bằng 0) = số lớn => mô hình sẽ hiểu ngay là sai
Như vậy khi dự đoán đúng và dự đoán sai cách biệt của Loss là rất lớn nên mô hình này mới thích hợp cho dự đoán Boolen 
* Nếu thực tế là Lành tính (y=0):
Phần đầu: 0log(...) = 0. (Biến mất).
Công thức chỉ còn: Loss = - log(1 - y mũ)
Tương tự như trên

# Về công thức đưa ra quyết định dự đoán z=w​.x ​+ b
## Giải thích về tham số
* x là các triệu chứng  
* w là độ quan trọng của các triệu chứng 
* b là tham số giúp độ chính xác trong dự đoán tăng lên
## Cơ chế 
**z là "điểm số thô" (score) cho mỗi mẫu dữ liệu. Điểm càng lớn thì khả năng là Ác tính càng cao, điểm càng nhỏ (âm) thì khả năng là Lành tính càng cao. Điểm số này có thể chạy từ âm vô cùng đến dương vô cùng. => z càng lớn tỉ lệ ung thư càng cao** (y^ = 1/ ( 1+ e^(-z) ))
* w là tham số quyết định xem triệu chứng nào đáng lo ngại.
w lớn (Dương): Triệu chứng này rất nguy hiểm. (Ví dụ: Kích thước càng to => càng dễ là ung thư).
w bằng 0: Triệu chứng vô dụng, không liên quan (Ví dụ: Màu mắt của bệnh nhân không ảnh hưởng đến ung thư).
w âm (Negative): Triệu chứng này ngược lại, càng cao càng an toàn. (Ví dụ: Mật độ tế bào khỏe mạnh càng cao => càng ít khả năng ung thư).
* b là tham số dịch chuyển tiêu chuẩn đánh giá
Nếu không có b, mốc quyết định luôn là số 0
Hãy nhìn vào đồ thị đường thẳng:
Nếu chỉ có y = ax (tức là b=0): Đường thẳng bắt buộc phải đi qua gốc tọa độ (0,0).
Nếu có y = ax + b: Đường thẳng có thể tịnh tiến lên xuống, trái phải tự do.
Trong bài toán ung thư:
Giả sử tất cả các chỉ số xét nghiệm (x) đều bằng 0.
Nếu không có bias: z = 0 => y^ = 1/ ( 1+ e^(-z) ) = 1/2.  
Xác suất 50% (lưỡng lự). Điều này vô lý vì nếu không có dấu hiệu gì thì phải là Lành tính.

# Về công thức y^ = 1/ ( 1+ e^(-z) )
Chuyển đổi "điểm số thô" thành xác suất. Ví dụ: z = 2 => y = 0.88 (88% xác suất là bệnh)

