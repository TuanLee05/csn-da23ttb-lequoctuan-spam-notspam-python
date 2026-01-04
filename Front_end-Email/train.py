import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Hàm đọc email từ thư mục
def load_emails_from_folder(folder_path):
    emails = []
    for name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, name)
        if os.path.isfile(file_path):
            try:
                with open(file_path, encoding="latin-1") as f:
                    content = f.read().strip()
                    if content:
                        emails.append(content)
                        if len(emails) <= 3:  # in thử vài file đầu
                            print("Đọc thành công:", file_path)
            except Exception as e:
                print(f"Lỗi đọc {file_path}: {e}")
    return emails

# Đường dẫn gốc tới SpamAssassin
BASE_DIR = r"D:\SpamAssassin"

# Đọc dữ liệu từ các thư mục con (có thêm lớp lồng bên trong)
easy_ham = load_emails_from_folder(os.path.join(BASE_DIR, "easy_ham", "easy_ham"))
hard_ham = load_emails_from_folder(os.path.join(BASE_DIR, "hard_ham", "hard_ham"))
spam = load_emails_from_folder(os.path.join(BASE_DIR, "spam_2", "spam_2"))

print("Số email easy_ham:", len(easy_ham))
print("Số email hard_ham:", len(hard_ham))
print("Số email spam:", len(spam))

# Gộp dữ liệu và gán nhãn
texts = easy_ham + hard_ham + spam
labels = ["ham"] * (len(easy_ham) + len(hard_ham)) + ["spam"] * len(spam)

print("Tổng số email:", len(texts))

if len(texts) == 0:
    print("❌ Không đọc được email nào.")
    exit()

# Vector hóa bằng TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
#model = SVC(kernel="linear")#lưu ý thay thế
#C1:model = SVC(kernel="linear", class_weight="balanced")
model = SVC(kernel="linear", probability=True, class_weight="balanced")

model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Lưu mô hình và vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Đã tạo model.pkl và vectorizer.pkl")
