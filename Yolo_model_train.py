from ultralytics import YOLO
import os
import sys
import torch
import re

def check_device():
    """
    檢查並返回設備 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        print(f"\n\033[1;32m\u2714\033[0m 已檢測到 GPU,將使用 {torch.cuda.get_device_name(0)} 進行訓練。")
        return 'cuda'
    else:
        print(f"\n\033[1;33m⚠\033[0m 未檢測到 GPU,將使用 CPU 進行訓練，可能較慢。")
        return 'cpu'


def get_latest_checkpoint_from_folders(save_dir="runs\\detect", base_name="yolo11n_carplate_det"):
    """
    在 save_dir 中搜尋以 base_name 開頭的所有資料夾，並從中找出最新的 epoch.pt。
    :param save_dir: 檢查點的基礎目錄
    :param base_name: 用於篩選的資料夾名稱基礎 (例如 "yolo11n_carplate_det")
    :return: 最新檢查點的檔案路徑（如果找到），否則返回 None
    """
    if not os.path.exists(save_dir):
        print(f"\u274C 檢查點目錄 {save_dir} 不存在！")
        return None

    # 搜尋所有以 base_name 開頭的資料夾
    subfolders = [os.path.join(save_dir, d) for d in os.listdir(save_dir)
                  if os.path.isdir(os.path.join(save_dir, d)) and d.startswith(base_name)]
    if not subfolders:
        print(f"\u26A0 沒有找到符合命名規則的資料夾於 {save_dir}")
        return None

    latest_checkpoint = None
    latest_epoch = -1

    # 遍歷每個子目錄，找出最新的 epoch.pt
    for folder in subfolders:
        print(f"\u231A 搜尋資料夾: {folder}")
        weights_dir = os.path.join(folder, "weights")
        if not os.path.exists(weights_dir):
            print(f"\u26A0 欄目 {weights_dir} 不存在，跳過。")
            continue

        # 搜索 weights 目錄下的 epoch.pt 檔案
        all_files = os.listdir(weights_dir)
        epoch_pattern = re.compile(r"epoch(\d+)\.pt")
        epoch_checkpoints = []
        for file in all_files:
            match = epoch_pattern.search(file)
            if match:
                epoch_checkpoints.append((int(match.group(1)), os.path.join(weights_dir, file)))

        # 如果該資料夾中有 epoch.pt，找出最大的 epoch
        if epoch_checkpoints:
            folder_latest_epoch, folder_latest_ckpt = max(epoch_checkpoints, key=lambda x: x[0])
            print(f"  \u2714 最新檢查點於 {folder}: epoch {folder_latest_epoch}")
            if folder_latest_epoch > latest_epoch:
                latest_epoch = folder_latest_epoch
                latest_checkpoint = folder_latest_ckpt

    if latest_checkpoint:
        print(f"\n\033[1;34m\u26A1\033[0m 已找到所有資料夾中最新的檢查點 {latest_checkpoint} (epoch: {latest_epoch})")
    else:
        print(f"\u26A0 在所有符合條件的資料夾中未找到檢查點。")
    return latest_checkpoint


def model_train(yolo_model, yaml_path, epochs, batch, name, device, save_period):
    """
    訓練 YOLO 模型，支持從檢查點恢復訓練。
    :param yolo_model: 使用的模型 (模型檔案 .pt 或 配置檔案 .yaml)
    :param yaml_path: 數據集配置檔案路徑
    :param epochs: 訓練輪數
    :param batch: 單次資料批量大小 (-1 表示根據設備內存自動調整)
    :param device: 使用設備 ('cpu' 或 'cuda')
    :param name: 模型訓練後的資料與模型儲存檔案夾名稱
    :param save_period: 檢查點儲存頻率
    """
    if not os.path.exists(yaml_path):
        print(f"\u274C 數據集配置檔案 {yaml_path} 不存在，請檢查路徑！")
        return None

    # 搜尋所有資料夾中的最新檢查點
    latest_checkpoint = get_latest_checkpoint_from_folders()
    if latest_checkpoint:
        print(f"\n\033[1;34m\u26A1\033[0m 檢查到最新檢查點 {latest_checkpoint}，將繼續訓練。")
        yolo_model = latest_checkpoint  # 使用最新的檢查點作為初始模型

    model = YOLO(yolo_model)  # 初始化 YOLO 模型

    try:
        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            plots=True,
            device=device,
            name=name,
            save_period=save_period,
            resume = True
        )
        print(f"\n\033[1;32m\u2714\033[0m 訓練完成！")
        return model
    except Exception as e:
        print(f"\u274C 訓練失敗！錯誤資訊：{e}")
        return None


def model_val_data(model, data_path, mode):
    """
    驗證模型數據
    :param model: 預訓練模型
    :param data_path: 數據集配置檔案路徑
    :param mode: 數據分割 ('val' 或 'test')
    """
    try:
        metrics = model.val(data=data_path, split=mode)
      
        print(f"\n\033[1;32m\u2714\033[0m 驗證完成！")
        return [metrics.map]
    except Exception as e:
        print(f"\u274C 驗證失敗！錯誤資訊：{e}")
        return None


def model_predict(model, data, savedir):
    """
    使用模型進行預測
    :param model: 預訓練模型
    :param data: 輸入數據路徑
    :param savedir: 預測結果保存路徑
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    try:
        results = model.predict(source=data)
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs

            # 在圖像上繪製標籤名稱和機率
            for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
                class_name = model.names[int(cls)]  # 將類別索引轉換為類別名稱
                result.plot(labels=True, boxes=True, conf=True)

            filename = os.path.join(savedir, f"result{i:02d}.jpg")  # 使用 f-string 格式化文件名
            result.save(filename=filename)  # save to disk

        print(f"\n\033[1;32m\u2714\033[0m 預測完成！結果已儲存到 {savedir}")
    except Exception as e:
        print(f"\u274C 預測失敗！錯誤資訊：{e}")


if __name__ == '__main__':
    DATA_YAML = "dataset.yaml"
    PREDICT_DATA = "dataset\\test\\images"
    PREDICT_SAVE_DIR = "predict"

    # 自動檢測設備（優先使用 GPU）
    DEVICE = check_device()

    # 訓練模型
    model = model_train(
        yolo_model="YOLO11n.yaml",  # 可使用 YOLO 配置檔或檢查點 .pt
        yaml_path=DATA_YAML,
        batch=-1,
        epochs=60,
        name = "yolo11n_carplate_det",
        device=DEVICE,
        save_period=1
    )
    if model is None:
        sys.exit(0)

    # 驗證模型
    val_map = model_val_data(model=model, data_path=DATA_YAML, mode="val")
    # print(f" 驗證集 mAP@0.5 : {val_map[0]:.4f}")
    # print(f" 驗證集 mAP@0.75 : {val_map[1]:.4f}")
    print(f" 驗證集 mAP@0.5:0.95 : {val_map[0]:.4f}")

    test_map = model_val_data(model=model, data_path=DATA_YAML, mode="test")
    # print(f" 驗證集 mAP@0.5 : {test_map[0]:.4f}")
    # print(f" 驗證集 mAP@0.75 : {test_map[1]:.4f}")
    print(f" 驗證集 mAP@0.5:0.95 : {test_map[0]:.4f}")

    # 模型預測
    model_predict(
        model=model,
        data=PREDICT_DATA,
        savedir=PREDICT_SAVE_DIR
    )
