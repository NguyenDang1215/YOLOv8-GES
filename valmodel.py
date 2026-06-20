from ultralytics import YOLO

def main():
    model = YOLO("v5.pt")  # tải mô hình đã huấn luyện

    metrics = model.val(
        data="C:\\Users\\Nguyen\\Final_report\\YOLO-APD\\data\\data.yaml",
        split="val",
        batch=16,
        imgsz=640,
        device=0,
        workers=0  # tránh lỗi multiprocessing trên Windows
    )

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()