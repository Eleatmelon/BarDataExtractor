import os
import sys
import argparse
# 首先导入matplotlib并设置为非交互式模式
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，不会显示窗口
from pathlib import Path  # 添加此行导入Path类
import glob  # 添加glob模块用于文件匹配

from image_axis_reader import get_axis_coordinates
from corp import detect_horizontal_lines, show_results
from bar_analyzer import analyze_stacked_bars

# 配置文件和加密相关设置
CONFIG_DIR = Path.home() / '.image_analyzer'
CONFIG_FILE = CONFIG_DIR / 'config.json'
KEY_FILE = CONFIG_DIR / '.key'

def process_single_image(image_path, x_max=None, y_max=None, display_images=False, output_dir=None, no_intermediate_files=False):
    """
    处理单个图像文件
    
    Args:
        image_path: 图像文件路径
        x_max: x轴最大值
        y_max: y轴最大值
        display_images: 是否显示图像预览
        output_dir: 输出目录
        no_intermediate_files: 是否不生成中间文件
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return False
    
    print(f"正在处理图像：{image_path}")
    
    # 第一步：调用图像识别API获取坐标信息
    try:
        print("正在进行图像坐标识别...")
        if x_max is None or y_max is None:
            x_max_new, y_max_new = get_axis_coordinates(image_path)
            
            # 如果在命令行指定了参数，优先使用命令行参数
            if x_max is None and x_max_new is not None:
                x_max = x_max_new
                print(f"获取到的x轴最大值: {x_max}")
            
            if y_max is None and y_max_new is not None:
                y_max = y_max_new
                print(f"获取到的y轴最大值: {y_max}")
        
        # 确保x_max和y_max是整数 
        if x_max is not None:
            try:
                x_max = int(x_max)
                print(f"有效的x轴最大值: {x_max}")
            except (ValueError, TypeError):
                print(f"警告: 获取到的x_max '{x_max}'不是有效的整数")
                x_max = None
        
        if y_max is not None:
            try:
                y_max = float(y_max)  # 使用float允许小数点
                print(f"有效的y轴最大值: {y_max}")
            except (ValueError, TypeError):
                print(f"警告: 获取到的y_max '{y_max}'不是有效的数值")
                y_max = None
        
    except Exception as e:
        print(f"图像识别API调用出错: {e}")
        x_max, y_max = None, None
    
    # 如果x_max或y_max为None，提示用户提供命令行参数
    if x_max is None or y_max is None:
        print("错误: 无法从图像中获取有效的坐标轴值")
        print("请使用以下参数重新运行程序:")
        print(f"python main.py {image_path} --x_max <X轴最大值> --y_max <Y轴最大值>")
        return False
    
    # 第二步：进行图像裁剪
    try:
        print("\n正在进行图像裁剪...")
        original, result, lines, boundaries, cropped, short_merged_lines = detect_horizontal_lines(image_path)
        
        # 显示结果（只在明确指定--display参数时显示）
        show_results(original, result, cropped, os.path.basename(image_path), display=display_images)
        
        # 如果有裁剪图像，则处理
        if cropped is not None:
            if output_dir is None:
                output_dir = os.path.dirname(image_path)
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 保存标记了水平线的中间结果图像（除非禁用中间文件）
            if not no_intermediate_files:
                lines_output_path = os.path.join(output_dir, f"{base_name}_lines.jpg")
                import cv2
                cv2.imwrite(lines_output_path, result)
                print(f"标记水平线的图像已保存到: {lines_output_path}")
            
            # 只在不禁用中间文件时保存裁剪图像
            cropped_path = None
            if not no_intermediate_files:
                cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
                import cv2
                cv2.imwrite(cropped_path, cropped)
                print(f"裁剪后的图像已保存到: {cropped_path}")
            else:
                print("已禁用中间文件生成，跳过保存中间图像")
                # 在内存中继续处理而不保存为文件
                
            # 输出水平线信息
            print(f"检测到 {len(lines)} 条水平线")
            
            # 第三步：分析柱状图
            try:
                print("\n正在分析柱状图...")
                print(f"传递给柱状图分析的参数: x_max={x_max}, y_max={y_max}")
                
                # 直接传递裁剪后的图像数据而不是文件路径
                if no_intermediate_files:
                    analyze_stacked_bars(cropped, x_max, y_max, output_dir, base_name, no_intermediate_files=True)
                else:
                    analyze_stacked_bars(cropped_path, x_max, y_max, output_dir)
                
                return True
            except Exception as e:
                print(f"柱状图分析出错: {e}")
                return False
        else:
            print("未生成裁剪图像，无法进行柱状图分析")
            return False
        
    except Exception as e:
        print(f"图像裁剪过程出错: {e}")
        return False

def process_directory(directory_path, x_max=None, y_max=None, display_images=False, output_dir=None, no_intermediate_files=False):
    """
    处理目录中的所有图像文件
    
    Args:
        directory_path: 目录路径
        x_max: x轴最大值
        y_max: y轴最大值
        display_images: 是否显示图像预览
        output_dir: 输出目录
        no_intermediate_files: 是否不生成中间文件
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误：目录 {directory_path} 不存在")
        return False
    
    # 支持的图像扩展名
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 获取目录中的所有图像文件
    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        # 同时支持大写扩展名
        pattern = os.path.join(directory_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # 排序文件列表
    image_files.sort()
    
    if not image_files:
        print(f"警告：在目录 {directory_path} 中未找到任何图像文件")
        return False
    
    print(f"在目录 {directory_path} 中找到 {len(image_files)} 个图像文件")
    
    # 如果没有指定输出目录，使用输入目录
    if output_dir is None:
        output_dir = directory_path
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图像文件
    success_count = 0
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理图像: {os.path.basename(image_path)}")
        if process_single_image(image_path, x_max, y_max, display_images, output_dir, no_intermediate_files):
            success_count += 1
    
    print(f"\n处理完成. 成功: {success_count}/{len(image_files)}")
    return success_count > 0

def main():
    """
    主函数：接收图像路径，调用图像识别API、图像裁剪函数和柱状图分析函数
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='图像处理工具')
    parser.add_argument('image_path', type=str, help='图像文件或目录的路径')
    parser.add_argument('--x_max', type=int, default=None, help='x轴最大值')
    parser.add_argument('--y_max', type=float, default=None, help='y轴最大值')
    parser.add_argument('--display', action='store_true', help='显示图像预览')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径')
    parser.add_argument('--no_intermediate_files', action='store_true', help='不生成中间文件，只输出CSV')
    args = parser.parse_args()
    
    image_path = args.image_path
    display_images = args.display
    output_dir = args.output_dir
    no_intermediate_files = args.no_intermediate_files
    
    # 检查路径是文件还是目录
    if os.path.isdir(image_path):
        print(f"检测到目录: {image_path}")
        process_directory(image_path, args.x_max, args.y_max, display_images, output_dir, no_intermediate_files)
    elif os.path.isfile(image_path):
        print(f"检测到单个文件: {image_path}")
        process_single_image(image_path, args.x_max, args.y_max, display_images, output_dir, no_intermediate_files)
    else:
        print(f"错误：路径 {image_path} 不存在")
        sys.exit(1)

if __name__ == "__main__":
    main()
