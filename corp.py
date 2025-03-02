import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import matplotlib.font_manager as fm
import warnings

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def detect_horizontal_lines(image_path, min_line_length=100, line_gap=10, edge_margin_percent=0.05, 
                           y_threshold=5, initial_length_percent=0.01, merged_length_percent=0.25):
    """
    识别图像中的水平线，排除顶部和底部的线，并将垂直位置相近的线合并
    然后计算所有水平线的边界来裁剪图像
    
    参数:
        image_path: 图像文件路径
        min_line_length: 最小线段长度
        line_gap: 同一条线上的最大间隙
        edge_margin_percent: 顶部和底部要排除的区域百分比（相对于图像高度）
        y_threshold: 垂直方向上的合并阈值，相差小于此值的线将被合并
        initial_length_percent: 初始线段筛选的长度下限（占图像宽度的百分比）
        merged_length_percent: 合并后线段的长度下限（占图像宽度的百分比）
    
    返回:
        原始图像、标记了水平线的图像、合并后的线条、边界坐标、裁剪后的图像、被排除的短合并线
    """
    # 读取图像
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 获取图像高度
    height, width = img.shape[:2]
    
    # 计算顶部和底部的边缘区域
    edge_margin = int(height * edge_margin_percent)
    top_edge = edge_margin
    bottom_edge = height - edge_margin
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 120, apertureSize=3)
    
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=min_line_length, 
        maxLineGap=line_gap
    )
    
    # 创建图像副本用于显示结果
    result_img = img.copy()
    
    # 筛选水平线（斜率接近于0的线），并排除顶部和底部的线
    horizontal_lines = []
    edge_lines = []  # 用于存储被排除的边缘线
    too_short_lines = []  # 用于存储长度过短的线
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算斜率 (避免除以零)
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # 如果斜率接近于0，认为是水平线
            if abs(slope) < 0.1:  # 斜率阈值可以调整
                # 计算线段长度
                line_length = abs(x2 - x1)
                min_initial_length = width * initial_length_percent  # 图像宽度的1%
                
                # 检查线是否位于顶部或底部边缘区域
                y_avg = (y1 + y2) / 2
                if y_avg < top_edge or y_avg > bottom_edge:
                    edge_lines.append(line[0])
                    # 在结果图像中用不同颜色（例如绿色）标记被排除的边缘线
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                elif line_length < min_initial_length:
                    too_short_lines.append(line[0])
                    # 在结果图像中用黄色标记被排除的短线
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                else:
                    horizontal_lines.append([x1, y1, x2, y2, y_avg])  # 保存平均y坐标
    
    # 按y坐标排序
    horizontal_lines.sort(key=lambda x: x[4])
    
    # 合并垂直位置相近的线
    merged_lines = []
    short_merged_lines = []  # 存储被舍弃的短合并线
    
    if horizontal_lines:
        current_group = [horizontal_lines[0]]
        
        for i in range(1, len(horizontal_lines)):
            current_line = horizontal_lines[i]
            prev_line = current_group[-1]
            
            # 如果当前线与上一条线的y坐标差异小于阈值，认为是同一条线
            if abs(current_line[4] - prev_line[4]) < y_threshold:
                current_group.append(current_line)
            else:
                # 合并当前组中的所有线
                if current_group:
                    merged_line = merge_line_group(current_group)
                    
                    # 检查合并后的线段长度是否达到图像宽度的指定百分比
                    x1, y1, x2, y2 = merged_line
                    line_length = abs(x2 - x1)
                    min_required_length = width * merged_length_percent
                    
                    if line_length >= min_required_length:
                        merged_lines.append(merged_line)
                    else:
                        short_merged_lines.append(merged_line)
                        # 在结果图像中用紫色标记被排除的短合并线
                        cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                
                # 开始新的一组
                current_group = [current_line]
        
        # 处理最后一组
        if current_group:
            merged_line = merge_line_group(current_group)
            
            # 检查最后一组合并后的线段长度
            x1, y1, x2, y2 = merged_line
            line_length = abs(x2 - x1)
            min_required_length = width * merged_length_percent
            
            if line_length >= min_required_length:
                merged_lines.append(merged_line)
            else:
                short_merged_lines.append(merged_line)
                # 在结果图像中用紫色标记被排除的短合并线
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
    
    # 输出过滤信息
    total_horizontal = len(horizontal_lines) + len(too_short_lines)
    print(f"检测到 {total_horizontal} 条潜在水平线")
    print(f"其中 {len(too_short_lines)} 条因长度不足图像宽度的{initial_length_percent*100}%而被排除")
    print(f"合并前保留了 {len(horizontal_lines)} 条有效水平线")
    
    # 计算所有水平线的边界极值
    boundaries = None
    cropped_img = None
    
    if merged_lines:
        # 初始化边界值
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        
        # 计算所有线的最小外接矩形
        for line in merged_lines:
            x1, y1, x2, y2 = line
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)
        
        # 添加小边距并向下平移2个像素
        padding = 3
        min_x = max(0, min_x - padding)
        max_x = min(width, max_x + padding)
        min_y = max(0, min_y - padding + 2)  # 向下平移2个像素
        max_y = min(height, max_y + padding + 2)  # 向下平移2个像素
        
        boundaries = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y
        }
        
        # 在结果图像上绘制边界矩形
        cv2.rectangle(result_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        
        # 裁剪图像
        if max_y > min_y and max_x > min_x:
            cropped_img = img[min_y:max_y, min_x:max_x]
    
    # 在结果图像上绘制合并后的线
    for line in merged_lines:
        x1, y1, x2, y2 = line
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return img, result_img, merged_lines, boundaries, cropped_img, short_merged_lines

def merge_line_group(line_group):
    """
    合并一组线段
    
    参数:
        line_group: 线段列表，每个线段为 [x1, y1, x2, y2, y_avg]
    
    返回:
        合并后的线段 [x1, y1, x2, y2]
    """
    # 计算组内所有线的平均y坐标
    avg_y = sum(line[4] for line in line_group) / len(line_group)
    
    # 找出最左边的点和最右边的点
    left_x = min(min(line[0], line[2]) for line in line_group)
    right_x = max(max(line[0], line[2]) for line in line_group)
    
    # 创建合并后的线段
    return [left_x, int(avg_y), right_x, int(avg_y)]

def save_result_images(original_path, result_img, cropped_img=None):
    """保存检测结果和裁剪图像到文件"""
    output_dir = os.path.dirname(original_path)
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    
    # 保存标记了水平线的中间结果图像
    lines_output_path = os.path.join(output_dir, f"{base_name}_lines.jpg")
    cv2.imwrite(lines_output_path, result_img)
    print(f"标记水平线的图像已保存到: {lines_output_path}")
    
    # 如果有裁剪图像，则保存
    if cropped_img is not None:
        output_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        cv2.imwrite(output_path, cropped_img)
        print(f"裁剪后的图像已保存到: {output_path}")

def show_results(original, result, cropped=None, title="", display=False):
    """
    显示原始图像、结果图像和裁剪后的图像
    
    Args:
        original: 原始图像
        result: 处理后的图像
        cropped: 裁剪后的图像，可选
        title: 图像标题，可选
        display: 是否显示图像，默认为False
    """
    if cropped is not None:
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f'原始图像 {title}')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'检测到的水平线 {title}')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title(f'裁剪后的图像 {title}')
        plt.axis('off')
    else:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f'原始图像 {title}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'检测到的水平线 {title}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 只有当display为True时才显示图像
    if display:
        plt.show()
    else:
        plt.close()  # 关闭图形，不显示

def process_single_image(image_path, display=False):
    """处理单个图像文件"""
    try:
        print(f"处理图像: {image_path}")
        original, result, lines, boundaries, cropped, short_merged_lines = detect_horizontal_lines(image_path)
        
        # 显示结果，但不弹出窗口
        show_results(original, result, cropped, os.path.basename(image_path), display=display)
        
        # 输出每条水平线的坐标
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            print(f"水平线 {i+1}: 从 ({x1}, {y1}) 到 ({x2}, {y2})")
        
        # 打印被排除的水平线
        if short_merged_lines:
            print("\n被忽略的水平线坐标:")
            for i, line in enumerate(short_merged_lines):
                x1, y1, x2, y2 = line
                print(f"被忽略的水平线 {i+1}: 从 ({x1}, {y1}) 到 ({x2}, {y2})")
        else:
            print("\n没有被忽略的水平线")
        
        # 保存结果图像
        save_result_images(image_path, result, cropped)
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")

def process_directory(directory_path, extensions=['*.jpg', '*.jpeg', '*.png'], display=False):
    """处理目录中的所有图像文件"""
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    if not image_files:
        print(f"在目录 {directory_path} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for image_path in image_files:
        process_single_image(image_path, display)

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='图像水平线检测和裁剪工具')
    parser.add_argument('--image_path', type=str, required=True, help='图像文件或目录的路径')
    parser.add_argument('--display', action='store_true', help='是否显示图像预览')
    parser.add_argument('--initial_length', type=float, default=0.01, help='初始线段筛选的长度下限（占图像宽度的百分比，默认0.01即1%）')
    parser.add_argument('--merged_length', type=float, default=0.25, help='合并后线段的长度下限（占图像宽度的百分比，默认0.25即25%）')
    args = parser.parse_args()
    
    image_dir = args.image_path
    display_images = args.display
    
    # 检测目录是否存在
    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
    elif os.path.isfile(image_dir):
        # 处理单个文件
        process_single_image(image_dir, display_images)
    else:
        # 处理目录中的所有图像
        process_directory(image_dir, display=display_images)
