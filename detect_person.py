import cv2
from ultralytics import YOLO
from plyer import notification
import time
import logging
import os
from datetime import datetime

# --- 配置日志 ---
# 创建一个日志文件夹（如果不存在）
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 定义日志文件名（包含日期）
log_filename = os.path.join(log_dir, f"detection_{datetime.now().strftime('%Y%m%d')}.log")

# 配置日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename), # 将日志写入文件
        logging.StreamHandler() # 也将日志输出到控制台
    ]
)

logging.info("--- 脚本启动 ---")

# --- 配置照片保存 ---
# 创建一个保存照片的文件夹（如果不存在）
photo_dir = "detected_persons"
os.makedirs(photo_dir, exist_ok=True)
logging.info(f"照片将保存在 '{photo_dir}' 文件夹中.")

# --- YOLOv8 配置 ---
# 加载 YOLOv8 模型
try:
    model = YOLO('yolov8n.pt') # 或选择其他模型
    logging.info("YOLOv8 模型加载成功.")
except Exception as e:
    logging.error(f"加载 YOLOv8 模型失败: {e}")
    exit()

# 定义YOLOv8模型检测的类别名称
# COCO数据集中的类别，0 是 'person'
class_names = model.names
person_class_id = None
for class_id, class_name in class_names.items():
    if class_name == 'person':
        person_class_id = class_id
        break

if person_class_id is None:
    logging.error("YOLOv8 模型未包含 'person' 类别.")
    exit()

logging.info(f"目标监测类别: 'person' (ID: {person_class_id}).")

# --- 摄像头配置 ---
# 打开摄像头
# 0 代表默认摄像头，如果有多个摄像头，可能需要更改数字
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    logging.error("错误: 无法打开摄像头.")
    exit()

logging.info("摄像头打开成功.")

# --- 主循环所需变量 ---
# 状态变量，用于避免重复发送通知和重复拍照
person_detected_previously = False
notification_cooldown = 15 # 通知冷却时间，单位秒，避免频繁弹窗
last_notification_time = 0
detection_confidence_threshold = 0.4 # 检测置信度阈值

logging.info("开始摄像头监测...")
logging.info("按下 'q' 键退出.")

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果帧读取不成功，则退出循环
    if not ret:
        logging.warning("警告: 无法读取帧.")
        # 可以选择在这里增加重试逻辑
        break

    # 在当前帧上运行YOLOv8推理，只检测 'person' 类别
    results = model(frame, classes=[person_class_id], conf=detection_confidence_threshold)

    # 检查是否检测到 'person'
    person_detected_current_frame = False
    if results and len(results[0].boxes) > 0:
        person_detected_current_frame = True

    # --- 处理 检测到人 的情况 ---
    if person_detected_current_frame and not person_detected_previously:
        # 如果之前没有检测到人，现在检测到了
        current_time = time.time()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # 获取带毫秒的时间戳

        # 1. 记录日志
        log_message = f"检测到有人! ({len(results[0].boxes)} 个目标)"
        logging.info(log_message)

        # 2. 拍照保存 (保存带框的图像)
        try:
            # 在帧上绘制检测结果
            annotated_frame = results[0].plot()
            photo_filename = os.path.join(photo_dir, f"person_{timestamp_str}.jpg")
            cv2.imwrite(photo_filename, annotated_frame)
            logging.info(f"照片已保存: {photo_filename}")
        except Exception as e:
            logging.error(f"保存照片失败: {e}")
            # 如果保存照片失败，继续主流程，不退出

        # 3. 发送弹窗通知 (受冷却时间限制)
        if current_time - last_notification_time > notification_cooldown:
            try:
                notification.notify(
                    title='有人出现！',
                    message='摄像头检测到有人移动。详情请查看日志和照片文件夹。',
                    # app_name='YOLO Detector', # 可选
                    timeout=5
                )
                last_notification_time = current_time # 更新上次通知时间
                logging.info("已发送弹窗通知.")
            except Exception as e:
                logging.error(f"发送通知失败: {e}")
        else:
             logging.info("通知在冷却时间内，跳过发送.")


        person_detected_previously = True # 更新状态

    # --- 处理 人离开 的情况 ---
    elif not person_detected_current_frame and person_detected_previously:
        # 如果之前检测到人，现在没有检测到
        logging.info("人已离开.")
        person_detected_previously = False # 更新状态

    # --- 显示画面 ---
    # 即使没有检测到人，也显示画面；如果检测到，显示带有标注的画面
    try:
        if person_detected_current_frame:
             # 如果检测到，显示带标注的帧 (annotated_frame 是在检测到人时生成的)
             display_frame = annotated_frame
        else:
             # 如果没检测到，显示原始帧
             display_frame = frame

        cv2.imshow('YOLOv8 Person Detection (Press Q to Quit)', display_frame)
    except Exception as e:
        logging.error(f"显示画面失败: {e}")


    # --- 检查退出条件 ---
    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("检测到 'q' 键，正在退出...")
        break

# --- 清理资源 ---
cap.release()
cv2.destroyAllWindows()
logging.info("摄像头释放，窗口关闭.")
logging.info("--- 脚本结束 ---")