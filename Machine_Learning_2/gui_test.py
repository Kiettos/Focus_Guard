import customtkinter as ctk
import cv2
from PIL import Image
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
from collections import deque
import csv
from datetime import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN MODEL ---
BASE_DIR = Path(r"C:\Users\KIET PC\Desktop\Focus_Guard")
MODELS_DIR = BASE_DIR / "outputs" / "models"
MODEL_NAME = "RandomForest"  # Chọn model tốt nhất hiện tại của bạn (RandomForest hoặc XGBoost)
LABEL_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Load các model đã train vào bộ nhớ
models = {}
for label in LABEL_COLUMNS:
    model_path = MODELS_DIR / f"{MODEL_NAME}_{label}.pkl"
    if model_path.exists():
        models[label] = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

# Tiện ích MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Index landmark chuẩn
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17]
NOSE_TIP = 1
LEFT_EAR_PT = 234
RIGHT_EAR_PT = 454

def calculate_ear(landmarks, eye_indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def calculate_mar(landmarks, mouth_indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_indices])
    v = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[1])
    return v / (h + 1e-6)

def calculate_head_pose(landmarks):
    nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
    left_ear = np.array([landmarks[LEFT_EAR_PT].x, landmarks[LEFT_EAR_PT].y])
    right_ear = np.array([landmarks[RIGHT_EAR_PT].x, landmarks[RIGHT_EAR_PT].y])
    dist_l = np.linalg.norm(nose - left_ear)
    dist_r = np.linalg.norm(nose - right_ear)
    yaw = (dist_r - dist_l) / (dist_r + dist_l + 1e-6)
    pitch = nose[1] - ((left_ear[1] + right_ear[1]) / 2)
    return yaw, pitch

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class RealTestApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"DAiSEE Real-time Inference ({MODEL_NAME})")
        self.geometry("950x550")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_ui()
        
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # Sliding window buffer (giữ lại 30 frame gần nhất để tính mean, std, slope giống lúc train)
        self.window_size = 30
        self.buffer = {
            "ear": deque(maxlen=self.window_size),
            "mar": deque(maxlen=self.window_size),
            "yaw": deque(maxlen=self.window_size),
            "pitch": deque(maxlen=self.window_size)
        }
        
        self.update_frame()
        
    def setup_ui(self):
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Đang khởi tạo camera...")
        self.video_label.pack(expand=True, fill="both")
        
        self.result_frame = ctk.CTkFrame(self, width=250, corner_radius=10)
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self.result_frame, text=f"MODEL: {MODEL_NAME}", font=ctk.CTkFont(weight="bold", size=14)).pack(pady=20)
        
        self.labels = {}
        for emotion in LABEL_COLUMNS:
            frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
            frame.pack(fill="x", padx=15, pady=8)
            ctk.CTkLabel(frame, text=f"{emotion}:", anchor="w", width=100).pack(side="left")
            lbl_val = ctk.CTkLabel(frame, text="-", font=ctk.CTkFont(weight="bold", size=16), text_color="yellow")
            lbl_val.pack(side="right")
            self.labels[emotion] = lbl_val

    def aggregate_window(self, values: list) -> dict:
        if len(values) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "slope": 0}
        arr = np.array(values)
        t = np.arange(len(arr))
        slope = np.polyfit(t, arr, deg=1)[0] if len(arr) > 1 else 0.0
        return {
            "mean": arr.mean(), "std": arr.std(),
            "min": arr.min(), "max": arr.max(), "slope": slope
        }

    def update_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=rgb, landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        
                        # Trích xuất 4 thông số cơ bản frame hiện tại
                        l_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
                        r_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
                        ear = (l_ear + r_ear) / 2.0
                        mar = calculate_mar(face_landmarks.landmark, MOUTH)
                        yaw, pitch = calculate_head_pose(face_landmarks.landmark)
                        
                        # Đẩy vào buffer
                        self.buffer["ear"].append(ear)
                        self.buffer["mar"].append(mar)
                        self.buffer["yaw"].append(yaw)
                        self.buffer["pitch"].append(pitch)
                        
                        # Chỉ dự đoán khi buffer đã gom đủ dữ liệu tối thiểu
                        if len(self.buffer["ear"]) >= 10:
                            feature_vector = []
                            for feat_name in ["ear", "mar", "yaw", "pitch"]:
                                stats = self.aggregate_window(list(self.buffer[feat_name]))
                                feature_vector.extend([stats["mean"], stats["std"], stats["min"], stats["max"], stats["slope"]])
                            
                            X_input = np.array([feature_vector])
                            
                            # Chạy dự đoán bằng các file .pkl thực tế
                            for emotion in LABEL_COLUMNS:
                                pred = models[emotion].predict(X_input)[0]
                                self.labels[emotion].configure(text=str(pred))
                else:
                    for key in self.labels:
                        self.labels[key].configure(text="No Face")

                img_pil = Image.fromarray(rgb).resize((640, 480), Image.Resampling.LANCZOS)
                self.video_label.configure(image=ctk.CTkImage(light_image=img_pil, size=(640, 480)), text="")
                
        self.after(33, self.update_frame)
        
    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.face_mesh.close()
        self.destroy()

if __name__ == "__main__":
    app = RealTestApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()