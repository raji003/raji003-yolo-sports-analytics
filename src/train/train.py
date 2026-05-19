from ultralytics import YOLO

# Path to your data.yaml
DATA_YAML = "Images for training/data.yaml"

# Choose a model (yolov8n.pt is nano, you can change to yolov8s.pt, yolov8m.pt, etc.)
MODEL = "yolov8n.pt"

# Training parameters
epochs = 200
imgsz = 640

if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(data=DATA_YAML, epochs=epochs, imgsz=imgsz, patience=20, lr0=0.01, lrf=0.001, augment=True, degrees=10, 
                perspective=0.001, mosaic=1.0, close_mosaic=20)