import numpy as np
import customtkinter as ctk
import cv2
from PIL import Image
import os
import json
from datetime import datetime
import threading
import time
import joblib
import mediapipe as mp

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("outputs/jsons", exist_ok=True)

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

DURATION_MAP = {
    "15 Seconds": 15,
    "30 Seconds": 30,
    "1 Minute": 60,
    "5 Minutes": 300
}

class ElearningSurveyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("E-learning Video Survey")
        self.geometry("450x800")
        
        self.is_recording = False
        self.cap = cv2.VideoCapture(0)
        self.timer_id = None
        self.video_writer = None
        self.file_base_name = None
        
        # Biến cho Đa luồng (Threading)
        self.running = True
        self.current_frame = None
        self.models = {} # Khởi tạo danh sách mô hình rỗng
        
        self.setup_ui()
        
        # 1. Luồng tải giao diện hiển thị
        self.update_webcam_ui()
        
        # 2. Luồng chạy Camera và ghi ổ cứng
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # 3. Luồng tải AI ngầm (không làm đơ giao diện)
        threading.Thread(target=self.load_ai_models_in_background, daemon=True).start()
     
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def setup_ui(self):
        self.video_frame = ctk.CTkFrame(self, height=320, corner_radius=10)
        self.video_frame.pack(side="top", fill="x", padx=15, pady=(15, 10))
        self.video_frame.pack_propagate(False) 
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Đang tải Camera...", font=ctk.CTkFont(size=14))
        self.video_label.pack(expand=True, fill="both")

        self.form_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.form_frame.pack(side="top", fill="both", expand=True, padx=15, pady=5)
        ctk.CTkLabel(self.form_frame, text="THÔNG TIN KHẢO SÁT", font=ctk.CTkFont(weight="bold", size=16)).pack(anchor="w", pady=(5, 15))

        self.entry_student_id = ctk.CTkEntry(self.form_frame, placeholder_text="Mã sinh viên (VD: SV123)", height=40)
        self.entry_student_id.pack(fill="x", pady=8)
        self.entry_name = ctk.CTkEntry(self.form_frame, placeholder_text="Họ và Tên", height=40)
        self.entry_name.pack(fill="x", pady=8)
        self.entry_test_id = ctk.CTkEntry(self.form_frame, placeholder_text="ID Bài kiểm tra", height=40)
        self.entry_test_id.pack(fill="x", pady=8)

        self.duration_var = ctk.StringVar(value="30 Seconds")
        self.dropdown_duration = ctk.CTkOptionMenu(self.form_frame, variable=self.duration_var, values=list(DURATION_MAP.keys()), height=40)
        self.dropdown_duration.pack(fill="x", pady=8)

        self.status_label = ctk.CTkLabel(self.form_frame, text="Sẵn sàng", text_color="black", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(anchor="center", pady=(15, 0))

        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(side="bottom", fill="x", padx=15, pady=20)

        # Nút bị khóa mặc định chờ tải xong AI
        self.btn_start = ctk.CTkButton(self.control_frame, text="BẮT ĐẦU GHI HÌNH", fg_color="green", hover_color="darkgreen", height=45, font=ctk.CTkFont(weight="bold"), state="disabled", command=self.start_survey)
        self.btn_start.pack(fill="x", pady=5)

        self.btn_stop = ctk.CTkButton(self.control_frame, text="DỪNG KHẨN CẤP", fg_color="red", hover_color="darkred", height=45, font=ctk.CTkFont(weight="bold"), state="disabled", command=lambda: self.stop_survey(is_emergency=True))
        self.btn_stop.pack(fill="x", pady=5)

    def load_ai_models_in_background(self):
        """Chạy ngầm việc đọc file .pkl để giao diện hiện lên ngay lập tức"""
        self.status_label.configure(text="Đang nạp AI...", text_color="#d35400")
        try:
            self.models = {
                "Boredom": joblib.load("outputs/models_tuned2/RandomForest_Boredom_tuned.pkl"),
                "Engagement": joblib.load("outputs/models_tuned2/XGBoost_Engagement_tuned.pkl"),
                "Confusion": joblib.load("outputs/models_tuned2/RandomForest_Confusion_tuned.pkl"),
                "Frustration": joblib.load("outputs/models_tuned2/RandomForest_Frustration_tuned.pkl")
            }
            self.status_label.configure(text="Sẵn sàng", text_color="black")
            self.btn_start.configure(state="normal") # Mở khóa nút bắt đầu
        except Exception as e:
            self.status_label.configure(text="Lỗi: Thiếu file .pkl", text_color="red")

    def camera_loop(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # --- BẮT ĐẦU: CODE TẠO PLACEHOLDER XÁM CHO BÁO CÁO ---
                    h, w = frame.shape[:2]
                    frame = np.full((h, w, 3), 128, dtype=np.uint8) # Khung nền xám
                    text = "[Camera Feed Placeholder]"
                    
                    # Căn giữa đoạn text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    
                    cv2.putText(frame, text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    # --- KẾT THÚC: CODE TẠO PLACEHOLDER ---

                    if self.is_recording and self.video_writer is not None:
                        self.video_writer.write(frame) # Lưu video
                        
                        if self.models:
                            # 1. Chuyển hệ màu để MediaPipe đọc được
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = self.face_mesh.process(rgb_frame)
                            
                            # 2. Nếu phát hiện có khuôn mặt trong camera
                            if results.multi_face_landmarks:
                                face_landmarks = results.multi_face_landmarks[0]
                                
                                # Tạm thời điền mảng 0 để tránh crash nếu bạn chưa kịp viết hàm tính toán
                                current_features = [0.0] * 20 
                                
                                X_input = [current_features]
                                
                                # 3. Đưa mảng đặc trưng thật vào AI dự đoán
                                p_bored = self.models["Boredom"].predict(X_input)[0]
                                p_engag = self.models["Engagement"].predict(X_input)[0]
                                p_confu = self.models["Confusion"].predict(X_input)[0]
                                p_frust = self.models["Frustration"].predict(X_input)[0]
                                
                                self.temp_csv_ram.append([p_bored, p_engag, p_confu, p_frust])
                            else:
                                # Nếu sinh viên cúi gập mặt/rời khỏi camera, ghi nhận nhãn 0
                                self.temp_csv_ram.append([0.0, 0.0, 0.0, 0.0])
                    
                self.current_frame = frame.copy()
        time.sleep(0.01)

    def update_webcam_ui(self):
        if self.current_frame is not None:
            display_frame = self.current_frame.copy()
            
            if self.is_recording:
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb).resize((420, 315), Image.Resampling.LANCZOS)
            imgtk = ctk.CTkImage(light_image=img_pil, size=(420, 315))
            self.video_label.configure(image=imgtk, text="")
                
        self.after(30, self.update_webcam_ui)

    def start_survey(self):
        if not self.entry_student_id.get().strip():
            self.status_label.configure(text="Vui lòng nhập Mã sinh viên!", text_color="red")
            return
            
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.entry_student_id.configure(state="disabled")
        self.entry_name.configure(state="disabled")
        self.entry_test_id.configure(state="disabled")
        self.dropdown_duration.configure(state="disabled")
        
        student_id = self.entry_student_id.get().strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_base_name = f"{student_id}_{timestamp}"

        self.pre_time = 20
        self.run_pre_countdown()

    def run_pre_countdown(self):
        if self.pre_time > 0:
            self.status_label.configure(text=f"Chuẩn bị tư thế... Bắt đầu sau: {self.pre_time}s", text_color="#d35400")
            self.timer_id = self.after(1000, self.run_pre_countdown)
            self.pre_time -= 1
        else:
            self.start_actual_recording()

    def start_actual_recording(self):
        video_path = f"outputs/videos/{self.file_base_name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        fps = 30.0 
        
        if self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
        else:
            width, height = 1280, 720 

        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        self.temp_csv_ram = []
        self.is_recording = True
        
        duration_str = self.duration_var.get()
        self.record_time = DURATION_MAP.get(duration_str, 30)
        
        self.run_record_countdown()

    def run_record_countdown(self):
        if self.record_time > 0:
            self.status_label.configure(text=f"Đang ghi hình... Còn lại: {self.record_time}s", text_color="red")
            self.timer_id = self.after(1000, self.run_record_countdown)
            self.record_time -= 1
        else:
            self.stop_survey(is_emergency=False)

    def stop_survey(self, is_emergency=False):
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None

        self.is_recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        final_scores = {}
        if hasattr(self, 'temp_csv_ram') and len(self.temp_csv_ram) > 0:
            import numpy as np
            data_matrix = np.array(self.temp_csv_ram)
            means = np.mean(data_matrix, axis=0)
            
            final_scores = {
                "boredom_score": round(float(means[0]), 4),
                "engagement_score": round(float(means[1]), 4),
                "confusion_score": round(float(means[2]), 4),
                "frustration_score": round(float(means[3]), 4)
            }
            self.temp_csv_ram = [] 

        if self.file_base_name is not None:
            json_path = f"outputs/jsons/{self.file_base_name}.json"
            metadata = {
                "student_id": self.entry_student_id.get().strip(),
                "full_name": self.entry_name.get().strip(),
                "test_id": self.entry_test_id.get().strip(),
                "duration_setting": self.duration_var.get(),
                "status": "Emergency Stopped" if is_emergency else "Completed",
                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ai_emotion_analysis": final_scores
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.entry_student_id.configure(state="normal")
        self.entry_name.configure(state="normal")
        self.entry_test_id.configure(state="normal")
        self.dropdown_duration.configure(state="normal")
        
        if is_emergency:
            self.status_label.configure(text="Đã dừng khẩn cấp! Đã lưu file AVI.", text_color="red")
        else:
            self.status_label.configure(text="Hoàn thành! Đã lưu AVI và JSON.", text_color="green")

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        self.destroy()

if __name__ == "__main__":
    app = ElearningSurveyApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()