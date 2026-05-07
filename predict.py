from ultralytics import YOLO

model = YOLO("runs/classify/train-5/weights/best.pt")

results = model.predict(
    source="test1.webp",
    imgsz=192
)

for r in results:
    class_id = r.probs.top1
    class_name = r.names[class_id]
    confidence = r.probs.top1conf.item()

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence*100:.2f}%")