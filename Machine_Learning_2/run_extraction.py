import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
import gc

from extract_features import extract_video_features

# Chặn cảnh báo rác của Protobuf làm lag Terminal
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

def build_video_index(videos_dir: Path) -> dict:
    """
    Quét toàn bộ thư mục ĐÚNG 1 LẦN DUY NHẤT trước khi chạy.
    Trả về dictionary map giữa clip_stem (ClipID) và đường dẫn file thực tế.
    """
    print(f"Đang lập chỉ mục các file video trong {videos_dir}...")
    video_index = {}
    
    # Quét tất cả file .avi (hoặc .mp4), loại bỏ các thư mục rỗng
    for p in videos_dir.rglob("*.*"): 
        if p.is_file() and p.suffix.lower() in [".avi", ".mp4"]:
            video_index[p.stem] = p
            
    print(f"Tìm thấy {len(video_index)} video.")
    return video_index

def sanitize_filename(clip_id: str) -> str:
    return Path(clip_id).stem

def process_split(label_csv_path: Path, videos_dir: Path, output_dir: Path, limit: int = None):
    labels_df = pd.read_csv(label_csv_path)
    labels_df.columns = labels_df.columns.str.strip()

    if limit:
        labels_df = labels_df.head(limit)

    label_columns = [c for c in ["Boredom", "Engagement", "Confusion", "Frustration"] if c in labels_df.columns]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tạo Map Index (Tối ưu I/O)
    video_index = build_video_index(videos_dir)

    success_count = 0
    missing = []

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Extracting {label_csv_path.stem}"):
        clip_id = row["ClipID"]
        clip_stem = Path(clip_id).stem
        
        # Tra cứu O(1) qua Dictionary thay vì rglob
        video_path = video_index.get(clip_stem)

        if video_path is None:
            missing.append(clip_id)
            continue

        features = extract_video_features(str(video_path), sample_fps=3, max_frames=50)
        features["ClipID"] = clip_id

        for col in label_columns:
            features[col] = row[col]

        out_filename = sanitize_filename(clip_id) + ".csv"
        out_path = output_dir / out_filename
        pd.DataFrame([features]).to_csv(out_path, index=False)

        success_count += 1
        
        # Giải phóng RAM chủ động sau mỗi 100 video để tránh Memory Leak
        if success_count % 100 == 0:
            gc.collect()

    print(f"\n{label_csv_path.stem}: {success_count}/{len(labels_df)} video xử lý thành công")
    print(f"Kết quả lưu trong folder: {output_dir.resolve()}")
    if missing:
        preview = missing[:5]
        print(f"Thiếu {len(missing)} clip: {preview}{'...' if len(missing) > 5 else ''}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    process_split(Path(args.labels), Path(args.videos_dir), Path(args.output_dir), limit=args.limit)

if __name__ == "__main__":
    main()