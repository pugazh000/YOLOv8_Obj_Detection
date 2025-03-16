from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for a more powerful model

    # Train the model
    model.train(
        data="Your_pathYOLO\dataset\data.yaml",
        epochs=5,  # Reduce for quick training
        imgsz=416,
        device="cuda"  # Change to "cpu" if GPU is unavailable
    )

if __name__ == '__main__':
    main()
