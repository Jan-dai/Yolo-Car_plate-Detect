from ultralytics import YOLO
MODEL_PATH = "model(.pt)"
WANT_PREDECT_DATA = "Want_predect_images"
PREDECT_DATA_SAVEDIR = "predict"

model = YOLO(MODEL_PATH)
results = model.predict(source = WANT_PREDECT_DATA)


for i, result in enumerate(results):
    boxes = result.boxes  
    # masks = result.masks 
    
    # 在圖像上繪製標籤名稱和機率
    for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
        class_name = model.names[int(cls)]  # 將類別索引轉換為類別名稱
        result.plot(labels=True, boxes=True, conf=True)
        
   
    filename = f"{PREDECT_DATA_SAVEDIR}\\result{i:02d}.jpg"  # 使用 f-string 格式化文件名
    result.save(filename=filename)  # save to disk
