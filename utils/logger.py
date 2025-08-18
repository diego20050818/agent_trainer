import logging
import sys
import time
import colorlog
from colorlog import ColoredFormatter
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import os

class TypingConsoleHandler(logging.StreamHandler):
    """逐字输出日志的控制台 Handler"""
    def __init__(self, stream=None, delay=0.05):
        super().__init__(stream or sys.stdout)
        self.delay = delay

    def emit(self, record):
        try:
            msg = self.format(record)
            # 打字机效果：逐字输出
            for ch in msg:
                self.stream.write(ch)
                self.stream.flush()
                time.sleep(self.delay)
            self.stream.write("\n")
            self.flush()
        except Exception:
            self.handleError(record)

# 创建专门用于训练日志的logger
train_logger = logging.getLogger("train_logger")
train_logger.setLevel(logging.INFO)
train_logger.handlers.clear()

# 为训练日志创建文件处理器
train_file_handler = logging.FileHandler("file.log")
train_file_formatter = logging.Formatter(">>> %(asctime)s - %(levelname)s - %(message)s")
train_file_handler.setFormatter(train_file_formatter)
train_logger.addHandler(train_file_handler)

class TrainingLogCallback(TrainerCallback):
    """自定义训练日志回调"""
    
    def __init__(self, log_dir="trainlog"):
        # 创建 SummaryWriter 实例
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.train_log = {}
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            # 过滤出训练相关的日志
            train_logs = {k: v for k, v in logs.items() if k in [
                'loss', 'grad_norm', 'learning_rate', 'epoch', 
                'train_runtime', 'train_samples_per_second', 'train_steps_per_second', 'train_loss'
            ]}
            if train_logs:
                # 将训练指标写入 TensorBoard
                if 'loss' in train_logs:
                    self.writer.add_scalar('Training/Loss', train_logs['loss'], state.global_step)
                if 'grad_norm' in train_logs:
                    self.writer.add_scalar('Training/Grad Norm', train_logs['grad_norm'], state.global_step)
                if 'learning_rate' in train_logs:
                    self.writer.add_scalar('Training/Learning Rate', train_logs['learning_rate'], state.global_step)
                if 'epoch' in train_logs:
                    self.writer.add_scalar('Training/Epoch', train_logs['epoch'], state.global_step)
                
                train_logger.info(f">>> {train_logs}")
            self.train_log = train_logs
    
    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束时关闭 writer
        if hasattr(self, 'writer'):
            self.writer.close()

# ========== 创建 logger ==========
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# 文件处理器（正常写入，不要逐字）
file_handler = logging.FileHandler("file.log")
file_formatter = logging.Formatter(">>> %(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台处理器（逐字 + 彩色）
typing_handler = TypingConsoleHandler(sys.stdout, delay=0.03)  # 控制每个字母的延迟
console_formatter = ColoredFormatter(
    "%(log_color)s>>> %(asctime)s - %(levelname)-8s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%M:%S",
    log_colors={
        "DEBUG": "bold_light_blue,bg_black",     # 冰蓝色
        "INFO": "white,bg_black",                # 抹茶绿
        "WARNING": "bold_light_yellow,bg_black", # 柠檬黄
        "ERROR": "bold_light_red,bg_black",      # 草莓红
        "CRITICAL": "bold_purple,bg_black",      # 薰衣草紫
    }
)
typing_handler.setFormatter(console_formatter)
logger.addHandler(typing_handler)

if __name__ == "__main__":
    logger.info("这是一个逐字跳出来的日志效果")
    logger.warning("就像打字机一样~")
    logger.error("是不是很有动画感？")