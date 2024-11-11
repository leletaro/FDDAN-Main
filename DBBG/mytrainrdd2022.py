from ultralytics import YOLO

# Load a model
model = YOLO("DBBG.yaml")  # build a new model from YAML
model = YOLO("/PATH")  # load a pretrained model (recommended for training)
model = YOLO("DBBG.yaml").load("/PATH")  # build from YAML and transfer weights

# Train the model
results = model.train(data="RDD2022.yaml", epochs=300, imgsz=640)