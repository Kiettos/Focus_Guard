"""
extract_features.py
TOAN BO PIPELINE extract landmark, tu 1 video video -> 1 dong feature da nen.

Stage 1: MediaPipe Face Mesh -> 468 landmark tho (x,y,z) moi frame
Stage 2: Tinh feature hinh hoc (EAR, MAR, head pose) tu landmark tho, moi frame
Stage 3: Nen (aggregate) chuoi feature theo thoi gian thanh 1 vector dai dien cho ca video
"""

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ---------------------------------------------------------------------
# Landmark index co dinh cua MediaPipe Face Mesh (khong doi, luon giong nhau)
# ---------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]     # 6 diem quanh mat trai
RIGHT_EYE = [362, 385, 387, 263, 373, 380]   # 6 diem quanh mat phai
MOUTH = [61, 291, 39, 181, 0, 17]            # 6 diem quanh mieng
NOSE_TIP = 1
CHIN = 152
LEFT_EAR_PT = 234
RIGHT_EAR_PT = 454


# =======================================================================
# STAGE 2 — Cong thuc tinh feature hinh hoc tu landmark (per frame)
# =======================================================================

def calculate_ear(landmarks, eye_indices):
    """
    Eye Aspect Ratio = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    Thap = mat nham. Binh thuong ~0.25-0.35.
    """
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)


def calculate_mar(landmarks, mouth_indices):
    """
    Mouth Aspect Ratio = ||P_tren-P_duoi|| / ||P_trai-P_phai||
    Cao = dang ngap/ha mieng.
    """
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_indices])
    vertical = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[1])
    return vertical / (horizontal + 1e-6)


def calculate_head_pose(landmarks):
    """
    Yaw = (d_phai - d_trai) / (d_phai + d_trai)   -- quay trai/phai
    Pitch = y_mui - y_trung_diem_2_tai               -- cui/ngang
    """
    nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
    left_ear = np.array([landmarks[LEFT_EAR_PT].x, landmarks[LEFT_EAR_PT].y])
    right_ear = np.array([landmarks[RIGHT_EAR_PT].x, landmarks[RIGHT_EAR_PT].y])

    dist_left = np.linalg.norm(nose - left_ear)
    dist_right = np.linalg.norm(nose - right_ear)
    yaw = (dist_right - dist_left) / (dist_right + dist_left + 1e-6)

    ear_mid_y = (left_ear[1] + right_ear[1]) / 2
    pitch = nose[1] - ear_mid_y

    return yaw, pitch


def extract_frame_features(landmarks):
    """Tu 1 frame landmark (Stage 1 output) -> 1 dict feature tho (Stage 2 output)."""
    left_ear = calculate_ear(landmarks, LEFT_EYE)
    right_ear = calculate_ear(landmarks, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    mar = calculate_mar(landmarks, MOUTH)
    yaw, pitch = calculate_head_pose(landmarks)

    return {"ear": ear, "mar": mar, "yaw": yaw, "pitch": pitch}


# =======================================================================
# STAGE 3 — Nen (aggregate) chuoi feature theo thoi gian
# =======================================================================

def aggregate_sequence(values: list) -> dict:
    """
    Nen 1 chuoi gia tri thanh: mean, std, min, max, slope (linear regression).
    """
    if len(values) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "slope": np.nan}

    arr = np.array(values)
    t = np.arange(len(arr))
    slope = np.polyfit(t, arr, deg=1)[0] if len(arr) > 1 else 0.0

    return {
        "mean": arr.mean(),
        "std": arr.std(),
        "min": arr.min(),
        "max": arr.max(),
        "slope": slope,
    }


# =======================================================================
# HAM CHINH — Gop Stage 1 + 2 + 3, chay tren 1 video, tra ve 1 dict duy nhat
# =======================================================================

# QUAN TRONG: khoi tao FaceMesh 1 LAN DUY NHAT o muc module (khong nam trong ham),
# de dung lai cho toan bo video thay vi load lai model moi lan goi ham.
# Neu khoi tao lai trong moi lan goi extract_video_features(), voi ~9000 video se
# tro thanh nut that co chai lon nhat cua toan bo pipeline (moi lan khoi tao ton
# vai tram ms - 1s, nhan 9000 lan = hang nghin giay overhead khong can thiet).
_face_mesh_instance = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_video_features(video_path: str, sample_fps: int = 3, max_frames: int = 50) -> dict:
    """
    Input: duong dan 1 video.
    Output: 1 dict phang (flat) gom 20 feature da nen + 3 cot QA
            (frames_read, frames_detected, detection_rate).

    sample_fps: so frame lay mau moi giay (khong lay het video, tranh du thua)
    max_frames: gioi han so frame toi da xu ly (an toan neu video dai bat thuong)
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))

    # Noi luu chuoi gia tri theo thoi gian (Stage 2 output, truoc khi nen)
    sequences = {"ear": [], "mar": [], "yaw": [], "pitch": []}

    frame_idx = 0
    frames_read = 0
    frames_detected = 0

    # Dung lai instance da khoi tao san (KHONG tao moi o day nua)
    face_mesh = _face_mesh_instance

    while cap.isOpened() and frames_read < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frames_read += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- STAGE 1: MediaPipe extract raw landmark ---
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                frames_detected += 1
                landmarks = results.multi_face_landmarks[0].landmark

                # --- STAGE 2: tinh feature hinh hoc tu landmark tho ---
                feat = extract_frame_features(landmarks)
                for key in sequences:
                    sequences[key].append(feat[key])
            # neu khong detect duoc mat o frame nay, bo qua (khong append)

        frame_idx += 1

    cap.release()

    # --- STAGE 3: nen tung chuoi feature thanh thong ke tong hop ---
    output = {}
    for feature_name, seq in sequences.items():
        stats = aggregate_sequence(seq)
        for stat_name, value in stats.items():
            output[f"{feature_name}_{stat_name}"] = value

    output["frames_read"] = frames_read
    output["frames_detected"] = frames_detected
    output["detection_rate"] = frames_detected / frames_read if frames_read > 0 else 0.0

    return output


# =======================================================================
# CHAY THU TRUC TIEP FILE NAY (khong qua notebook) — vi du 1 video
# =======================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cach dung: python extract_features.py duong_dan_video.avi")
        sys.exit(1)

    video_path = sys.argv[1]
    result = extract_video_features(video_path)

    print(f"\nKet qua extract cho: {video_path}")
    for key, value in result.items():
        print(f"  {key}: {value}")