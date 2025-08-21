from .logger import logger
import datetime
def train_arg_printer(train_arg:dict):
    """从配置文件打印训练参数

    Args:
        train_arg (dict): 传入yaml读取的训练参数
    """
    logger.info("打印训练参数如下")
    for k,v in train_arg.items():
        logger.info(f"  {k} >>> {v}")

def generate_outputdir(model_name:str):
    """生成带时间戳的输出目录

    Args:
        model_name (str): 定义模型名称

    Returns:
        _type_: 生成模型名称+yyyymmddhhmm格式的输出目录
    """
    # 生成带时间戳的输出目录
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_dir = f"./output/{model_name}{current_time}"
    return output_dir