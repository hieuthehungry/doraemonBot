# doraemonBot
đây là một phiên bản tùy biến đầu vào (createDataset.py) từ bài viết xuất sắc của Adit Despande (https://adeshpande3.github.io/adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me)

> trước tiên thay đổi đường dẫn
<br> np.save('/home/q/Downloads/workPlaceBackUp/doraemonBot/conversationDictionary.npy',dictData) 
<br> ở dòng 36 của file createDataset.py phù hợp với đường trong máy tính của bạn

> dữ liệu lưu ở conversationData cần được thêm nếu muốn bot thú vị hơn, đây là cuộc đối thoại theo cặp mà nobita bắt đầu (Message: ... ) và doraemon là người trả lời (Response: ... )

> $ python createDataset.py

> $ python python Word2Vec.py
<br> chọn 'y' khi được hỏi Do you want to create your own vectors through Word2Vec (y/n)?

> $ python Seq2Seq.py

> ⏱ đợi ít nhất là 10 phút ⏱ 

# Enjoy! 🤭

