import cv2
import numpy as np
import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹出窗口
import sys

def analyze_stacked_bars(image, x_max=None, y_max=None, output_dir=None, base_name=None, no_intermediate_files=False):
    """
    分析堆叠柱状图，计算每根柱子的高度及其所占图像高度的百分比和实际数值
    
    Args:
        image: 图像路径或直接传入的OpenCV图像对象
        x_max: x轴最大值，用于计算柱子数量
        y_max: y轴最大值，用于计算具体数值
        output_dir: 输出目录，默认为图像所在目录
        base_name: 图像基本名称，当传入OpenCV图像对象时需提供
        no_intermediate_files: 是否禁用中间文件生成，只输出CSV
        
    Returns:
        DataFrame: 包含柱子信息的数据表
    """
    print("\n===== 开始柱状图分析 =====")
    
    # 读取图像 - 支持文件路径或直接传入的OpenCV图像对象
    if isinstance(image, str):
        print(f"读取图像文件: {image}")
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图像: {image}")
        # 如果传入的是文件路径，从文件名获取base_name
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(image))[0]
            print(f"从文件名获取基础名称: {base_name}")
    else:
        print("使用直接传入的OpenCV图像对象")
        # 如果直接传入的是OpenCV图像对象
        img = image
        if img is None:
            raise ValueError("传入的图像对象无效")
        # 确保传入了base_name
        if base_name is None:
            raise ValueError("当直接传入图像对象时，必须提供base_name参数")
        print(f"使用提供的基础名称: {base_name}")
    
    # 获取图像尺寸
    height, width, _ = img.shape
    print(f"图像尺寸: {width}x{height} 像素")
    
    # 确保x_max和y_max是有效值
    if x_max is None:
        raise ValueError("x轴最大值不能为空")

    if y_max is None:
        raise ValueError("y轴最大值不能为空")
    
    print(f"输入参数: x_max={x_max}, y_max={y_max}")
    
    # 计算柱子数量
    num_bars = int(x_max / 25) + 1
    print(f"根据公式 int(x_max/25)+1 计算得到柱子数量: {num_bars}")
    
    # 转换为HSV颜色空间，方便颜色识别
    print("转换图像至HSV颜色空间...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义三种颜色的HSV范围（蓝色、红色、绿色）
    print("定义颜色范围:")
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    print(f"  蓝色范围: H[{blue_lower[0]}-{blue_upper[0]}], S[{blue_lower[1]}-{blue_upper[1]}], V[{blue_lower[2]}-{blue_upper[2]}]")
    
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    print(f"  红色范围1: H[{red_lower1[0]}-{red_upper1[0]}], S[{red_lower1[1]}-{red_upper1[1]}], V[{red_lower1[2]}-{red_upper1[2]}]")
    print(f"  红色范围2: H[{red_lower2[0]}-{red_upper2[0]}], S[{red_lower2[1]}-{red_upper2[1]}], V[{red_lower2[2]}-{red_upper2[2]}]")
    
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    print(f"  绿色范围: H[{green_lower[0]}-{green_upper[0]}], S[{green_lower[1]}-{green_upper[1]}], V[{green_lower[2]}-{green_upper[2]}]")
    
    # 创建颜色掩码
    print("创建颜色掩码...")
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # 合并所有颜色掩码
    combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(red_mask, green_mask))
    
    # 计算垂直投影 - 统计每列非零像素
    print("计算垂直投影...")
    projection = np.sum(combined_mask, axis=0)
    print(f"投影最大值: {np.max(projection)}, 平均值: {np.mean(projection):.2f}")
    
    # 平滑处理
    kernel_size = 5
    print(f"对投影进行平滑处理，使用核大小: {kernel_size}")
    projection_smooth = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')
    
    # 找到最左侧的柱子
    # 设置阈值，投影值超过阈值的点认为是柱子区域的一部分
    threshold = np.max(projection_smooth) * 0.2
    print(f"设置柱子检测阈值: {threshold:.2f} (最大投影值的20%)")
    
    # 寻找最左侧柱子的左边缘
    print("寻找最左侧柱子边缘...")
    left_edge = 0
    while left_edge < width and projection_smooth[left_edge] < threshold:
        left_edge += 1
    
    if left_edge >= width:
        print("警告：无法检测到最左侧柱子，使用默认边距")
        left_edge = int(width * 0.05)  # 默认左边距为宽度的5%
    
    # 寻找最左侧柱子的右边缘
    right_edge = left_edge
    while right_edge < width and projection_smooth[right_edge] >= threshold:
        right_edge += 1
    
    # 计算最左侧柱子的宽度和左边距
    first_bar_width = right_edge - left_edge
    left_margin = left_edge
    
    print(f"最左侧柱子检测结果: 左边距={left_margin}像素, 宽度={first_bar_width}像素")
    
    # 计算柱子间隔
    print("计算柱子间隔...")
    print(f"  总宽度: {width}像素")
    print(f"  左边距: {left_margin}像素")
    print(f"  右边距: {left_margin}像素 (假设等于左边距)")
    print(f"  单个柱子宽度: {first_bar_width}像素")
    print(f"  柱子总数: {num_bars}")
    
    # 总可用空间 = 总宽度 - 左边距*2 - 所有柱子的宽度
    # 间隔数量 = 柱子数量-1
    total_bar_width = first_bar_width * num_bars
    total_margin = left_margin * 2
    total_gap_space = width - total_margin - total_bar_width
    
    print(f"  所有柱子总宽度: {total_bar_width}像素")
    print(f"  总边距: {total_margin}像素")
    print(f"  可用于间隔的总空间: {total_gap_space}像素")
    
    if num_bars > 1:
        gap_width = total_gap_space / (num_bars - 1)
        print(f"  间隔数量: {num_bars - 1}")
    else:
        gap_width = 0
        print("  只有一个柱子，无需计算间隔")
    
    print(f"计算得到的柱子间隔: {gap_width:.2f}像素")
    
    # 定义验证柱子位置的函数
    def validate_bar_positions():
        """
        验证检测到的柱子位置是否合理，如果不合理则返回均匀分布的柱子位置
        
        Returns:
            list or None: 如果需要使用均匀分布，返回新的柱子位置列表；否则返回None
        """
        print("\n开始验证柱子位置计算结果...")
        
        # 计算均匀分布下的柱子中心位置 - 保留原始柱子宽度
        # 如果检测到的first_bar_width为异常值，使用图像宽度的5%作为默认值
        bar_width = first_bar_width if (first_bar_width > 0 and first_bar_width < width * 0.2) else int(width * 0.05)
        print(f"均匀分布参考值:")
        print(f"  使用柱子宽度: {bar_width}像素")
        
        # 计算柱子中心点之间的间距
        if num_bars > 1:
            # 分配可用空间给num_bars个柱子中心点
            center_spacing = width / num_bars
            print(f"  柱子中心点间距: {center_spacing:.2f}像素")
        else:
            center_spacing = width
            print(f"  只有一个柱子，居中放置")
        
        # 计算第一个柱子的中心位置（均匀分布下）
        uniform_first_bar_center = center_spacing / 2
        
        # 计算当前方法的第一根柱子中心位置
        current_first_bar_center = left_margin + first_bar_width / 2
        position_diff = abs(current_first_bar_center - uniform_first_bar_center)
        
        print(f"位置差异分析:")
        print(f"  当前方法第一根柱子中心: {current_first_bar_center:.2f}像素")
        print(f"  均匀分布第一根柱子中心: {uniform_first_bar_center:.2f}像素")
        print(f"  位置差异: {position_diff:.2f}像素")
        print(f"  允许的最大差异 (1/4柱子中心间距): {center_spacing/4:.2f}像素")
        
        # 如果差异过大，切换到均匀分布方法
        # 使用中心间距的1/4作为阈值（更严格的判断）
        if position_diff > center_spacing / 4:
            print("警告: 检测到的柱子位置与理论位置差异过大")
            print("切换到均匀分布方法...")
            
            # 计算均匀分布下的柱子位置
            uniform_positions = []
            for i in range(num_bars):
                # 柱子中心 = (i + 0.5) * 中心间距
                bar_center = (i + 0.5) * center_spacing
                bar_left = int(bar_center - bar_width / 2)
                bar_right = int(bar_center + bar_width / 2)
                
                # 确保不超出图像边界
                bar_left = max(0, bar_left)
                bar_right = min(width - 1, bar_right)
                
                uniform_positions.append({
                    'bar_id': i + 1,
                    'center_x': int(bar_center),
                    'left_x': bar_left,
                    'right_x': bar_right
                })
            
            print("已根据均匀分布重新计算所有柱子位置")
            return uniform_positions
        else:
            print("验证通过: 检测到的柱子位置在合理范围内")
            return None
    
    # 验证柱子位置
    uniform_bar_positions = validate_bar_positions()
    
    # 定义所有柱子的位置
    print("\n计算每个柱子的精确位置...")
    bar_positions = []
    
    # 跟踪使用的方法（1: 原始检测方法, 2: 均匀分布方法）
    method_used = 2 if uniform_bar_positions is not None else 1
    method_suffix = f"_method_{method_used}"
    print(f"使用方法 {method_used}: {'均匀分布方法' if method_used == 2 else '原始检测方法'}")
    
    # 如果验证返回了均匀分布的柱子位置，则使用它
    if uniform_bar_positions is not None:
        bar_positions = uniform_bar_positions
        # 输出均匀分布后的柱子位置
        for bar in bar_positions:
            print(f"  柱子 {bar['bar_id']}: 左边缘={bar['left_x']}px, 中心={bar['center_x']}px, 右边缘={bar['right_x']}px")
    else:
        # 否则，使用原始方法计算柱子位置
        for i in range(num_bars):
            # 柱子左边缘 = 左边距 + i * (柱子宽度 + 间隔)
            bar_left = left_margin + i * (first_bar_width + gap_width)
            bar_center = bar_left + first_bar_width / 2
            bar_right = bar_left + first_bar_width
            
            bar_positions.append({
                'bar_id': i + 1,
                'center_x': int(bar_center),
                'left_x': int(bar_left),
                'right_x': int(bar_right)
            })
            print(f"  柱子 {i+1}: 左边缘={int(bar_left)}px, 中心={int(bar_center)}px, 右边缘={int(bar_right)}px")
    
    # 创建可视化图像
    visualization = img.copy()
    
    # 存储柱子信息的列表
    bars_data = []
    
    # 对每个预测的柱子位置进行分析
    print("\n===== 开始分析每个柱子 =====")
    for bar in bar_positions:
        bar_id = bar['bar_id']
        left_x = bar['left_x']
        right_x = bar['right_x']
        center_x = bar['center_x']
        
        print(f"\n分析柱子 #{bar_id} (X位置: {center_x})")
        
        # 绘制柱子位置 - 使用亮紫色替代黄色，并增加线宽
        cv2.line(visualization, (center_x, 0), (center_x, height), (255, 0, 255), 2)  # 亮紫色，宽度为2
        cv2.putText(visualization, f"{bar_id}", (center_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)  # 文字也用亮紫色
        
        # 初始化颜色区域的上下边界
        color_regions = {
            'blue': {'min_y': height, 'max_y': 0},
            'red': {'min_y': height, 'max_y': 0},
            'green': {'min_y': height, 'max_y': 0}
        }
        
        # 扫描柱子区域的每一个像素
        print(f"  扫描区域: X[{left_x}-{right_x}], Y[0-{height}]")
        for y in range(height):
            for x in range(left_x, right_x + 1):
                if x >= width or y >= height:
                    continue
                
                # 获取像素HSV值
                pixel_hsv = hsv[y, x]
                h, s, v = pixel_hsv
                
                # 检查像素颜色
                is_blue = (blue_lower[0] <= h <= blue_upper[0]) and (blue_lower[1] <= s <= blue_upper[1]) and (blue_lower[2] <= v <= blue_upper[2])
                is_red = ((red_lower1[0] <= h <= red_upper1[0]) or (red_lower2[0] <= h <= red_upper2[0])) and \
                         (red_lower1[1] <= s <= red_upper1[1]) and (red_lower1[2] <= v <= red_upper1[2])
                is_green = (green_lower[0] <= h <= green_upper[0]) and (green_lower[1] <= s <= green_upper[1]) and (green_lower[2] <= v <= green_upper[2])
                
                # 更新各颜色的区域范围
                if is_blue:
                    color_regions['blue']['min_y'] = min(color_regions['blue']['min_y'], y)
                    color_regions['blue']['max_y'] = max(color_regions['blue']['max_y'], y)
                if is_red:
                    color_regions['red']['min_y'] = min(color_regions['red']['min_y'], y)
                    color_regions['red']['max_y'] = max(color_regions['red']['max_y'], y)
                if is_green:
                    color_regions['green']['min_y'] = min(color_regions['green']['min_y'], y)
                    color_regions['green']['max_y'] = max(color_regions['green']['max_y'], y)
        
        # 计算各颜色的高度
        blue_height = color_regions['blue']['max_y'] - color_regions['blue']['min_y'] + 1 if color_regions['blue']['max_y'] >= color_regions['blue']['min_y'] else 0
        red_height = color_regions['red']['max_y'] - color_regions['red']['min_y'] + 1 if color_regions['red']['max_y'] >= color_regions['red']['min_y'] else 0
        green_height = color_regions['green']['max_y'] - color_regions['green']['min_y'] + 1 if color_regions['green']['max_y'] >= color_regions['green']['min_y'] else 0
        
        print("  检测到的颜色区域:")
        if blue_height > 0:
            print(f"    蓝色: Y[{color_regions['blue']['min_y']}-{color_regions['blue']['max_y']}], 高度={blue_height}像素")
        else:
            print(f"    蓝色: 未检测到")
            
        if red_height > 0:
            print(f"    红色: Y[{color_regions['red']['min_y']}-{color_regions['red']['max_y']}], 高度={red_height}像素")
        else:
            print(f"    红色: 未检测到")
            
        if green_height > 0:
            print(f"    绿色: Y[{color_regions['green']['min_y']}-{color_regions['green']['max_y']}], 高度={green_height}像素")
        else:
            print(f"    绿色: 未检测到")
        
        # 计算总高度
        total_height = blue_height + red_height + green_height
        print(f"  柱子总高度: {total_height}像素")
        
        # 计算高度百分比
        print("  计算高度百分比:")
        blue_percent = (blue_height / height) * 100 if height > 0 else 0
        red_percent = (red_height / height) * 100 if height > 0 else 0
        green_percent = (green_height / height) * 100 if height > 0 else 0
        total_percent = (total_height / height) * 100 if height > 0 else 0
        
        print(f"    蓝色: {blue_height}px / {height}px * 100 = {blue_percent:.2f}%")
        print(f"    红色: {red_height}px / {height}px * 100 = {red_percent:.2f}%")
        print(f"    绿色: {green_height}px / {height}px * 100 = {green_percent:.2f}%")
        print(f"    总计: {total_height}px / {height}px * 100 = {total_percent:.2f}%")
        
        # 计算具体数值
        print(f"  计算实际数值 (基于y轴最大值 {y_max}):")
        blue_value = blue_percent * y_max / 100
        red_value = red_percent * y_max / 100
        green_value = green_percent * y_max / 100
        total_value = total_percent * y_max / 100
        
        print(f"    蓝色: {blue_percent:.2f}% * {y_max} / 100 = {blue_value:.2f}")
        print(f"    红色: {red_percent:.2f}% * {y_max} / 100 = {red_value:.2f}")
        print(f"    绿色: {green_percent:.2f}% * {y_max} / 100 = {green_value:.2f}")
        print(f"    总计: {total_percent:.2f}% * {y_max} / 100 = {total_value:.2f}")
        
        # 存储柱子信息
        bars_data.append({
            'bar_id': bar_id,
            'x_position': center_x,
            'width': right_x - left_x + 1,
            'blue_height': blue_height,
            'red_height': red_height,
            'green_height': green_height,
            'total_height': total_height,
            'blue_percent': blue_percent,
            'red_percent': red_percent,
            'green_percent': green_percent,
            'total_percent': total_percent,
            'blue_value': blue_value,
            'red_value': red_value,
            'green_value': green_value,
            'total_value': total_value
        })
        
        # 在可视化图像上标记柱子高度 - 使用更粗的线条
        if green_height > 0:
            cv2.rectangle(visualization, (left_x, color_regions['green']['min_y']), 
                         (right_x, color_regions['green']['max_y']), (0, 255, 0), 2)
        if red_height > 0:
            cv2.rectangle(visualization, (left_x, color_regions['red']['min_y']), 
                         (right_x, color_regions['red']['max_y']), (0, 0, 255), 2)
        if blue_height > 0:
            cv2.rectangle(visualization, (left_x, color_regions['blue']['min_y']), 
                         (right_x, color_regions['blue']['max_y']), (255, 0, 0), 2)
    
    # 创建DataFrame
    df = pd.DataFrame(bars_data)
    
    # 设置默认输出目录
    if output_dir is None:
        if isinstance(image, str):
            output_dir = os.path.dirname(image)
        else:
            output_dir = os.getcwd()  # 如果是内存图像，默认为当前工作目录
    
    print(f"\n===== 保存分析结果 =====")
    # 保存CSV文件 - 添加方法后缀
    csv_path = os.path.join(output_dir, f"{base_name}{method_suffix}_bar_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"柱状图分析结果已保存到: {csv_path}")
    
    # 只在不禁用中间文件生成时保存图像
    if not no_intermediate_files:
        # 保存可视化图像 - 添加方法后缀
        vis_path = os.path.join(output_dir, f"{base_name}{method_suffix}_bar_analysis.jpg")
        cv2.imwrite(vis_path, visualization)
        print(f"可视化结果已保存到: {vis_path}")
        
        # 创建柱状图可视化 - 添加方法后缀
        create_bar_chart(df, output_dir, f"{base_name}{method_suffix}")
    else:
        print("已禁用中间文件生成，跳过保存可视化图像和柱状图")
    
    # 创建比较图像，同时显示两种方法的柱子位置
    def create_comparison_image(original_positions, uniform_positions):
        """
        创建比较图像，同时显示原始检测和均匀分布两种方法的柱子位置
        
        Args:
            original_positions: 原始检测方法计算的柱子位置
            uniform_positions: 均匀分布方法计算的柱子位置
        
        Returns:
            比较图像
        """
        comparison_img = img.copy()
        
        # 绘制原始检测方法的柱子位置 - 使用紫色
        for bar in original_positions:
            center_x = bar['center_x']
            cv2.line(comparison_img, (center_x, 0), (center_x, height), (255, 0, 255), 2)  # 紫色
            cv2.putText(comparison_img, f"{bar['bar_id']}", (center_x-15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 绘制均匀分布方法的柱子位置 - 使用黄色
        for bar in uniform_positions:
            center_x = bar['center_x']
            cv2.line(comparison_img, (center_x, 0), (center_x, height), (0, 255, 255), 2)  # 黄色
            cv2.putText(comparison_img, f"{bar['bar_id']}", (center_x+15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 添加图例
        legend_y = 80
        cv2.putText(comparison_img, "紫色: 原始检测方法", (20, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(comparison_img, "黄色: 均匀分布方法", (20, legend_y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return comparison_img
    
    # 计算原始方法的柱子位置（无论验证是否通过）
    original_positions = []
    for i in range(num_bars):
        bar_left = left_margin + i * (first_bar_width + gap_width)
        bar_center = bar_left + first_bar_width / 2
        bar_right = bar_left + first_bar_width
        
        original_positions.append({
            'bar_id': i + 1,
            'center_x': int(bar_center),
            'left_x': int(bar_left),
            'right_x': int(bar_right)
        })
    
    # 计算均匀分布的柱子位置（无论验证是否通过）
    center_spacing = width / num_bars
    bar_width = first_bar_width if (first_bar_width > 0 and first_bar_width < width * 0.2) else int(width * 0.05)
    
    uniform_positions = []
    for i in range(num_bars):
        bar_center = (i + 0.5) * center_spacing
        bar_left = int(bar_center - bar_width / 2)
        bar_right = int(bar_center + bar_width / 2)
        
        # 确保不超出图像边界
        bar_left = max(0, bar_left)
        bar_right = min(width - 1, bar_right)
        
        uniform_positions.append({
            'bar_id': i + 1,
            'center_x': int(bar_center),
            'left_x': bar_left,
            'right_x': bar_right
        })
    
    # 创建比较图像
    comparison_img = create_comparison_image(original_positions, uniform_positions)
    
    # 保存比较图像 - 添加方法后缀
    if not no_intermediate_files:
        comparison_path = os.path.join(output_dir, f"{base_name}{method_suffix}_bar_comparison.jpg")
        cv2.imwrite(comparison_path, comparison_img)
        print(f"柱子位置比较图已保存到: {comparison_path}")
    
    print("===== 柱状图分析完成 =====\n")
    return df

def create_bar_chart(df, output_dir, base_name):
    """创建柱状图可视化"""
    print("创建柱状图可视化...")
    plt.figure(figsize=(14, 8))
    
    # 准备数据
    bar_ids = df['bar_id']
    blue_heights = df['blue_percent']
    red_heights = df['red_percent']
    green_heights = df['green_percent']
    
    # 创建堆叠柱状图
    plt.bar(bar_ids, green_heights, color='green', label='绿色部分')
    plt.bar(bar_ids, red_heights, bottom=green_heights, color='red', label='红色部分')
    plt.bar(bar_ids, blue_heights, bottom=green_heights+red_heights, color='blue', label='蓝色部分')
    
    plt.xlabel('柱子ID')
    plt.ylabel('高度百分比 (%)')
    plt.title('柱状图高度分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(bar_ids)
    
    # 保存图表 - 已经包含方法后缀
    chart_path = os.path.join(output_dir, f"{base_name}_bar_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"柱状图可视化已保存到: {chart_path}")

def main():
    parser = argparse.ArgumentParser(description='柱状图高度分析工具')
    parser.add_argument('image_path', type=str, help='裁剪后的图像路径')
    parser.add_argument('--x_max', type=int, default=None, help='x轴最大值')
    parser.add_argument('--y_max', type=int, default=None, help='y轴最大值')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径')
    parser.add_argument('--no_intermediate_files', action='store_true', help='不生成中间文件，只输出CSV')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"错误: 文件 {args.image_path} 不存在")
        return
    
    try:
        analyze_stacked_bars(args.image_path, args.x_max, args.y_max, args.output_dir, no_intermediate_files=args.no_intermediate_files)
    except Exception as e:
        import traceback
        print(f"处理图像时出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 