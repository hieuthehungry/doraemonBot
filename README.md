# doraemonBot
đây là một phiên bản tùy biến đầu vào (createDataset.py) từ bài viết xuất sắc của Adit Despande (https://adeshpande3.github.io/adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me)

> trước tiên thay đổi đường dẫn
<br> np.save('/home/q/Downloads/workPlaceBackUp/doraemonBot/conversationDictionary.npy',dictData) 
<br> ở dòng 36 của file createDataset.py phù hợp với đường trong máy tính của bạn

> dữ liệu lưu ở conversationData cần được thêm nếu muốn bot thú vị hơn, đây là cuộc đối thoại theo cặp mà nobita bắt đầu (Message: ... ) và doraemon là người trả lời (Response: ... )

> $ python createDataset.py

> $ python Word2Vec.py
<br> chọn 'y' khi được hỏi Do you want to create your own vectors through Word2Vec (y/n)?

> chạy lại $ python Word2Vec.py nếu lần đầu có lỗi

> $ python Seq2Seq.py (nếu muốn dùng model đã có sửa lại dòng 201 file Sep2Seq.py)

> ⏱ đợi cho đến khi nào có thư mục model được tạo ra ⏱ 

# Enjoy!

![](https://raw.githubusercontent.com/d0betga1/doraemonBot/master/Screenshot%20from%202018-10-27%2016-38-15.png)
