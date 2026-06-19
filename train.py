from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/home/lolita/testYolo/datasets_bus27/testYolo.v1-v1.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="camion_tp"
)
