import os
from .logger import logger
import re

def list_training_outputs(output_dir:str):
    """列出output目录下的所有训练输出文件夹 -> list"""
    try:
        # 获取output目录下的所有文件夹
        training_folders = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
        
        if not training_folders:
            logger.error(f"在 {output_dir} 目录下没有找到训练输出文件夹!")
            raise ValueError("没有找到训练输出文件夹!")
            
        return training_folders
    except FileNotFoundError:
        logger.error(f"目录 {output_dir} 不存在!")
        raise

def select_training_model(output_dir:str):
    """通过终端交互让用户选择要测试的训练模型"""
    training_folders = list_training_outputs(output_dir)
    
    print("可用的训练模型文件夹:")
    for i, folder in enumerate(training_folders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = input(f"\n请选择要测试的模型 (1-{len(training_folders)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(training_folders):
                selected_folder = training_folders[choice_idx]
                selected_path = os.path.join(output_dir, selected_folder)
                logger.info(f"已选择模型文件夹: {selected_folder}")
                return selected_path,selected_folder
            else:
                print(f"请输入有效的选项 (1-{len(training_folders)})")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择")
            raise

def search_latest_checkpoint(output_dir:str):
    # 获取所有 checkpoint 文件夹
    checkpoints = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and re.match(r'checkpoint-\d+', d)
    ]

    if not checkpoints:
        logger.error("没有找到 checkpoint 文件夹!")
        raise ValueError("没有找到 checkpoint 文件夹!")

    # 按编号排序，找到最大的
    latest_checkpoint = max(
        checkpoints,
        key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1))
    )

    lora_path = os.path.join(output_dir, latest_checkpoint)
    logger.info(f"最新的 LoRA checkpoint 路径:{lora_path}")
    return lora_path