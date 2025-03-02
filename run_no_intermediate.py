#!/usr/bin/env python3
"""
只生成CSV结果的图像分析脚本
此脚本运行图像分析程序而不生成任何中间文件，只输出最终的CSV分析结果
支持处理单个图像文件或文件夹中的所有图像文件
"""

import os
import sys
import argparse
import glob

# 延迟导入main模块，避免在显示帮助信息时初始化API客户端
# from main import main as run_image_to_data

def process_single_image(image_path, x_max=None, y_max=None, output_dir=None, skip_existing=True):
    """
    处理单个图像文件
    
    Args:
        image_path: 图像文件路径
        x_max: x轴最大值
        y_max: y轴最大值
        output_dir: 输出目录
        skip_existing: 若已存在对应的CSV文件则跳过处理
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return False
    
    # 检查输出目录
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # 构建预期的CSV文件路径
    base_name = os.path.basename(image_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    csv_file_path = os.path.join(output_dir, f"{file_name_without_ext}_bar_analysis.csv")
    
    # 检查CSV文件是否已存在
    if skip_existing and os.path.exists(csv_file_path):
        print(f"跳过处理：{base_name} - 已存在对应的CSV文件")
        return True
    
    # 导入主模块
    try:
        from main import main as run_image_to_data
    except ImportError as e:
        print(f"导入主模块失败: {e}")
        return False
    
    # 构建运行参数
    cmd_args = [image_path]
    if x_max is not None:
        cmd_args.extend(['--x_max', str(x_max)])
    if y_max is not None:
        cmd_args.extend(['--y_max', str(y_max)])
    if output_dir is not None:
        cmd_args.extend(['--output_dir', output_dir])
    
    # 添加不生成中间文件的参数
    cmd_args.append('--no_intermediate_files')
    
    # 保存原始命令行参数
    original_argv = sys.argv.copy()
    
    try:
        # 替换系统参数
        sys.argv = [sys.argv[0]] + cmd_args
        # 运行主程序
        run_image_to_data()
        return True
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return False
    finally:
        # 恢复原始命令行参数
        sys.argv = original_argv

def process_folder(folder_path, x_max=None, y_max=None, output_dir=None, extensions=None, skip_existing=True):
    """
    处理文件夹中的所有图像文件
    
    Args:
        folder_path: 文件夹路径
        x_max: x轴最大值
        y_max: y轴最大值
        output_dir: 输出目录
        extensions: 图像文件扩展名列表
        skip_existing: 若已存在对应的CSV文件则跳过处理
    """
    if not os.path.isdir(folder_path):
        print(f"错误：{folder_path} 不是有效的文件夹")
        return False
    
    # 默认支持的图像扩展名
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 获取文件夹中所有图像文件
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        # 同时支持大写扩展名
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # 排序文件列表，确保处理顺序一致
    image_files.sort()
    
    if not image_files:
        print(f"警告：在文件夹 {folder_path} 中未找到任何图像文件")
        return False
    
    print(f"在文件夹 {folder_path} 中找到 {len(image_files)} 个图像文件")
    
    # 设置默认输出目录
    if output_dir is None:
        output_dir = folder_path
    
    # 顺序处理每个图像文件
    success_count = 0
    skipped_count = 0
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理图像: {os.path.basename(image_path)}")
        
        # 检查是否需要跳过已存在的文件
        base_name = os.path.basename(image_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        csv_file_path = os.path.join(output_dir, f"{file_name_without_ext}_bar_analysis.csv")
        
        if skip_existing and os.path.exists(csv_file_path):
            print(f"跳过处理：{base_name} - 已存在对应的CSV文件")
            skipped_count += 1
            success_count += 1
            continue
        
        if process_single_image(image_path, x_max, y_max, output_dir, skip_existing=False):
            success_count += 1
    
    print(f"\n处理完成. 成功: {success_count}/{len(image_files)}, 其中跳过: {skipped_count}")
    return True

def main():
    """
    主函数：解析命令行参数并运行图像分析程序，不生成中间文件
    """
    parser = argparse.ArgumentParser(description='图像分析工具 - 只生成CSV输出')
    parser.add_argument('path', type=str, help='图像文件或文件夹的路径')
    parser.add_argument('--x_max', type=int, default=None, help='x轴最大值')
    parser.add_argument('--y_max', type=float, default=None, help='y轴最大值')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径')
    parser.add_argument('--force', action='store_true', help='强制处理所有图像，即使已存在对应CSV文件')
    args = parser.parse_args()
    
    # 获取命令行参数
    path = args.path
    x_max = args.x_max
    y_max = args.y_max
    output_dir = args.output_dir
    skip_existing = not args.force
    
    # 检查路径是文件还是文件夹
    if os.path.isdir(path):
        print(f"检测到文件夹路径: {path}")
        process_folder(path, x_max, y_max, output_dir, skip_existing=skip_existing)
    elif os.path.isfile(path):
        print(f"检测到单个文件: {path}")
        process_single_image(path, x_max, y_max, output_dir, skip_existing=skip_existing)
    else:
        print(f"错误：路径 {path} 不存在或无效")
        sys.exit(1)

if __name__ == "__main__":
    main() 