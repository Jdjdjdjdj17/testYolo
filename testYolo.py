from ultralytics import YOLO


model = YOLO("/home/lolita/testYolo/runs/detect/camion_tp/weights/best.pt")
results = model("test1.jpg")

for result in results:
    boxes = result.boxes
    result.show()
    result.save(filename="result.jpg")

