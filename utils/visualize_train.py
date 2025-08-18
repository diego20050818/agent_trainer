import re
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row
from bokeh.models import HoverTool, DatetimeTickFormatter, Select, CustomJS, ColumnDataSource, DatePicker
from bokeh.plotting import curdoc
import json
from datetime import datetime
import argparse

def parse_training_logs(log_file_path, start_time=None, end_time=None):
    """
    解析训练日志文件，提取训练指标数据，并可选择性地过滤时间范围
    
    Args:
        log_file_path (str): 日志文件路径
        start_time (datetime, optional): 起始时间
        end_time (datetime, optional): 结束时间
    """
    # 存储训练数据
    training_data = []
    
    # 定义日志行的正则表达式模式
    log_pattern = r">>> (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - >>> ({.*})"
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(log_pattern, line)
            if match:
                timestamp_str = match.group(1)
                metrics_str = match.group(2)
                
                try:
                    # 解析时间戳
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    
                    # 检查时间范围过滤条件
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    # 解析指标数据
                    metrics = json.loads(metrics_str.replace("'", "\""))
                    
                    # 添加到数据列表
                    training_data.append({
                        'timestamp': timestamp,
                        **metrics
                    })
                except json.JSONDecodeError:
                    # 如果不是JSON格式，跳过该行
                    continue
    
    return pd.DataFrame(training_data)

def create_advanced_interactive_plot(df):
    """
    创建高级交互式训练可视化图表
    """
    # 准备数据
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 创建ColumnDataSource
    source = ColumnDataSource(df)
    
    # 创建图表
    p = figure(
        title="训练指标可视化",
        x_axis_type="datetime",
        width=900,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # 为不同指标创建线条
    loss_line = p.line(
        x='timestamp',
        y='loss',
        source=source,
        line_width=2,
        color='blue',
        legend_label='Loss',
        name='loss'
    )
    
    lr_line = p.line(
        x='timestamp',
        y='learning_rate',
        source=source,
        line_width=2,
        color='green',
        legend_label='Learning Rate',
        name='learning_rate'
    )
    
    grad_line = p.line(
        x='timestamp',
        y='grad_norm',
        source=source,
        line_width=2,
        color='red',
        legend_label='Gradient Norm',
        name='grad_norm'
    )
    
    epoch_line = p.line(
        x='timestamp',
        y='epoch',
        source=source,
        line_width=2,
        color='orange',
        legend_label='Epoch',
        name='epoch'
    )
    
    # 添加悬停工具
    hover = HoverTool(
        tooltips=[
            ("时间", "@timestamp{%Y-%m-%d %H:%M:%S}"),
            ("损失", "@loss{0.0000}"),
            ("学习率", "@learning_rate{0.0000e0}"),
            ("梯度范数", "@grad_norm{0.0000}"),
            ("轮数", "@epoch{0.00}")
        ],
        formatters={
            '@timestamp': 'datetime'
        }
    )
    
    p.add_tools(hover)
    p.legend.location = "top_left"
    p.xaxis.axis_label = "时间"
    p.yaxis.axis_label = "指标值"
    
    # 设置时间轴格式
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M",
        days="%m/%d",    # 如果想要显示周几，可以改成 "%a %d"
        months="%m/%Y",  # 如果想要英文缩写月份，用 "%b %Y"
        years="%Y"
    )
    
    return p, source

def main():
    parser = argparse.ArgumentParser(description='训练日志可视化工具')
    parser.add_argument('--log-file', type=str, default='file.log', help='日志文件路径')
    parser.add_argument('--output', type=str, default='training_visualization.html', help='输出HTML文件路径')
    parser.add_argument('--start-time', type=str, help='起始时间 (格式: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-time', type=str, help='结束时间 (格式: YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    # 解析时间范围参数
    start_time = None
    end_time = None
    
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("起始时间格式错误，应为 YYYY-MM-DD HH:MM:SS")
            return
    
    if args.end_time:
        try:
            end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("结束时间格式错误，应为 YYYY-MM-DD HH:MM:SS")
            return
    
    # 解析日志文件
    df = parse_training_logs(args.log_file, start_time, end_time)
    
    if df.empty:
        print("未找到训练日志数据")
        return
    
    print(f"解析到 {len(df)} 条训练记录")
    
    # 创建可视化图表
    plot, source = create_advanced_interactive_plot(df)
    
    # 保存并显示图表
    output_file(args.output)
    show(plot)
    
    print(f"可视化图表已保存为 {args.output}")

if __name__ == "__main__":
    main()