# Chấm tác vụ thực tế

Mẫu này bổ sung cho bộ JSON tự động. Dùng tài liệu bạn có quyền sử dụng và giữ test độc lập trước khi tune.

Chuẩn bị 4 nhóm: hỏi đáp theo văn bản; giải thích Python; tóm tắt; giữ thông tin và định dạng. Với từng câu, ghi đáp án/rubric trước khi xem output. Nếu lấy nhiều câu từ một tài liệu, giữ chúng trong cùng split.

Mỗi đáp án được chấm ba tiêu chí, mỗi tiêu chí 0-2:

1. Đúng thông tin: 0 sai chính; 1 thiếu/sai nhỏ; 2 đúng các ý cần thiết.
2. Làm đúng yêu cầu: 0 bỏ yêu cầu; 1 chưa đủ; 2 đủ nội dung và định dạng.
3. Dễ hiểu: 0 khó dùng; 1 dùng được nhưng cần sửa; 2 rõ ràng cho người mới.

Đánh dấu riêng lỗi nghiêm trọng/bịa thông tin trong tác vụ dựa trên văn bản. Không bù lỗi nghiêm trọng bằng văn phong hay. Nếu có hai người chấm, ẩn tên cấu hình và đảo thứ tự câu trả lời.

Điểm thang 0-1 = tổng / 6, kèm tỷ lệ lỗi nghiêm trọng. Báo các tiêu chí riêng; không nhập điểm thủ công vào bộ chọn hiện tại nếu chưa cập nhật rõ giao thức và giữ test độc lập.

Gợi ý câu hỏi:

- Cho một thông báo workshop tự viết; hỏi phòng và thời gian đăng ký.
- Cho đoạn Python tính tổng có lỗi off-by-one; yêu cầu chỉ ra lỗi và sửa.
- Cho đoạn văn 200-300 từ; yêu cầu tóm tắt 3 ý và giữ đúng các con số.
- Giải thích sự khác nhau giữa list và dictionary bằng một ví dụ quản lý sách.

Những gợi ý này chưa phải bộ test chuẩn hóa. Người làm đồ án cần bổ sung văn bản nguồn và đáp án cụ thể.
