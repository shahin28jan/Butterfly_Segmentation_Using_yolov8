from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source = "test_img", show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False, save_crop=False, line_width=2)


