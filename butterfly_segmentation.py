from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')
YOLO(task='segment', mode='train', model='yolov8s-seg.pt', data= 'dataset.yaml', epochs=1, imgsz=416 )



