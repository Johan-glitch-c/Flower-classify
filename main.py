from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data="flower_photos",
        epochs=20,
        imgsz=192,
        workers=0
    )

if __name__ == "__main__":
    freeze_support()
    main()