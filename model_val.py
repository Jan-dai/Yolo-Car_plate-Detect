from ultralytics import YOLO

DATA_YAML = "dataset.yaml"
MODEL_PATH = "epoch59.pt"
VAL_MODE = ["train" , "val" , "test"]

model = YOLO(model = MODEL_PATH)

def mAP_output(val_data):
    try:
        map_5095 = val_data.box.map
        map_50 = val_data.box.map50
        map_75 = val_data.box.map75
        
    
        print(f"\n\033[1;32m\u2714\033[0m 驗證完成！")
        try:
            print(f"mAP50-95 : {map_5095:.3f}")  
        except Exception as e:   
            print(f"\u274C mAP50~95驗證失敗!錯誤資訊 : {e}")
        try:
            print(f"mAP50 : {map_50:.3f}") 
        except Exception as e:   
            print(f"\u274C mAP50驗證失敗!錯誤資訊 : {e}")
        try:
            print(f"mAP75 : {map_75:.3f}") 
        except Exception as e:   
            print(f"\u274C mAP75驗證失敗!錯誤資訊 : {e}")
    
    
        #print(f"mAP50-95 : {map_50_75_95}")
        
    except Exception as e: 
        print(f"\u274C 驗證失敗！錯誤資訊：{e}")
        
if __name__ == '__main__':
    print("自組資料包")
    print("-------------------------------------------------")
    for _ in VAL_MODE:
        mAP_output(model.val(data = DATA_YAML,split = _ ))
        print("-------------------------------------------------")