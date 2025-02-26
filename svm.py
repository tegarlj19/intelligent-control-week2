import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
file_path = "SVM-MODEL/colors.csv"
color_data = pd.read_csv(file_path)

# Pastikan hanya kolom numerik digunakan sebagai fitur
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Ubah label kategori ke angka
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisasi fitur agar lebih optimal untuk SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Inisialisasi model SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale')  # Gunakan kernel RBF untuk performa lebih baik
svm_model.fit(X_train, y_train)

# Evaluasi model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model SVM: {accuracy*100:.2f}%")

# Fungsi untuk mencari warna terdekat
def find_nearest_color(rgb_color):
    distances = np.sqrt(np.sum((X - rgb_color) ** 2, axis=1))
    nearest_index = np.argmin(distances)
    return y[nearest_index]

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

detected_colors_1 = []
detected_true_labels_1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    box_size = 50

    # Koordinat bounding box
    x1_start, y1_start = (width // 2 - box_size // 2, height // 2 - box_size // 2)
    x1_end, y1_end = (x1_start + box_size, y1_start + box_size)


    roi1 = frame[y1_start:y1_end, x1_start:x1_end]
    avg_color1 = np.mean(roi1, axis=(0, 1)).astype(int)

    avg_color1_rgb = avg_color1[::-1]

    avg_color1_scaled = scaler.transform([avg_color1_rgb])

    color_pred_index1 = svm_model.predict(avg_color1_scaled)[0]
    color_pred1 = label_encoder.inverse_transform([color_pred_index1])[0]
    true_color1 = find_nearest_color(avg_color1_rgb)

    detected_colors_1.append(color_pred1)
    detected_true_labels_1.append(true_color1)


    if len(detected_colors_1) > 50:
        detected_colors_1.pop(0)
        detected_true_labels_1.pop(0)


    accuracy_1 = accuracy_score(detected_true_labels_1, detected_colors_1) * 100 if detected_colors_1 else 0.0

    # Menampilkan hasil prediksi di terminal
    print(f'Predicted Color 1: {color_pred1}, True Color 1: {true_color1}, Accuracy: {accuracy_1:.2f}%')

    cv2.rectangle(frame, (x1_start, y1_start), (x1_end, y1_end), (255, 255, 0), 2)
    cv2.putText(frame, f'{color_pred1}', (x1_start, y1_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 69, 0), 2)

    cv2.putText(frame, f'Color 1: {color_pred1} (Acc: {accuracy_1:.2f}%)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 69, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()