from ultralytics import YOLO

# Load a model
model = YOLO("yolo12n.pt")

# Train the model
results = model.train(data="data/dataset.yaml", epochs=100, imgsz=640)
