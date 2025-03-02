import fitz  # PyMuPDF
import os
import sys
import cv2
import numpy as np
from PIL import Image
import io
import logging
import re
import glob
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def is_bar_chart(image_bytes, debug=False):
    """判断图像是否为柱状图"""
    try:
        # 将图像字节转换为OpenCV可处理的格式
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # 检查图像形状，确保处理正确
        if len(image_np.shape) < 3:
            # 单通道图像，转为三通道
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            # RGBA格式，转换为RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        # 获取图像尺寸
        height, width = image_np.shape[:2]
        
        # 如果图像太小，不太可能是有用的柱状图
        if width < 100 or height < 100:
            return False
            
        # 转为灰度图
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 方法1: 基于轮廓检测的方法
        # 使用自适应阈值二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
                                      
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓，找出矩形候选
        rectangles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 过滤掉太小或太大的轮廓
            area_ratio = (w * h) / (width * height)
            if w > 5 and h > 10 and 0.001 < area_ratio < 0.2:
                rectangles.append((x, y, w, h))
        
        if debug:
            logger.info(f"找到 {len(rectangles)} 个可能的柱形")
            
        # 如果找到足够多的矩形，可能是柱状图
        if len(rectangles) >= 3:
            # 按x坐标排序
            rectangles.sort(key=lambda r: r[0])
            
            # 检查是否有多个矩形在水平方向上分布
            x_coords = [r[0] for r in rectangles]
            x_spacing = np.diff(x_coords)
            
            if len(x_spacing) > 0:
                # 计算x间距的标准差与均值之比
                spacing_variation = np.std(x_spacing) / (np.mean(x_spacing) + 1e-5)
                
                # 计算高度的变化
                heights = [r[3] for r in rectangles]
                height_variation = np.std(heights) / (np.mean(heights) + 1e-5)
                
                # 计算底部y坐标
                bottoms = [r[1] + r[3] for r in rectangles]
                bottom_variation = np.std(bottoms) / (np.mean(bottoms) + 1e-5)
                
                # 柱状图特征: 规则的x间距，高度变化较大，底部对齐
                if (spacing_variation < 0.5 and height_variation > 0.2 and bottom_variation < 0.2) or \
                   (len(rectangles) >= 5 and height_variation > 0.15):
                    if debug:
                        logger.info(f"方法1检测到柱状图: spacing_var={spacing_variation:.2f}, height_var={height_variation:.2f}, bottom_var={bottom_variation:.2f}")
                    return True
        
        # 方法2: 基于颜色直方图的垂直投影分析
        # 这种方法更适合检测有明显颜色区分的柱状图
        
        # 获取图像下半部分 (通常柱状图在图表的下半部分)
        lower_half = image_np[height//3:, :]
        
        # 转为HSV颜色空间，便于颜色分析
        hsv = cv2.cvtColor(lower_half, cv2.COLOR_RGB2HSV)
        
        # 计算色相通道的直方图
        hist_hue = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        
        # 计算饱和度通道的直方图
        hist_sat = cv2.calcHist([hsv], [1], None, [10], [0, 256])
        
        # 如果有多种明显的颜色 (柱状图通常使用不同颜色区分)
        peak_count_hue = np.sum(hist_hue > np.mean(hist_hue) * 1.5)
        peak_count_sat = np.sum(hist_sat > np.mean(hist_sat) * 1.5)
        
        if peak_count_hue >= 3 or peak_count_sat >= 3:
            if debug:
                logger.info(f"方法2检测到柱状图: 色相峰值={peak_count_hue}, 饱和度峰值={peak_count_sat}")
            return True
            
        # 方法3: 基于边缘检测的垂直线检测
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 使用霍夫变换检测垂直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=height/10, maxLineGap=10)
        
        if lines is not None:
            vertical_lines = 0
            horizontal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # 垂直线 (接近90度)
                if 80 <= angle <= 100:
                    vertical_lines += 1
                # 水平线 (接近0或180度)
                elif angle <= 10 or angle >= 170:
                    horizontal_lines += 1
                    
            # 柱状图特征: 多个垂直线，至少有一些水平线（可能是坐标轴或网格线）
            if vertical_lines >= 4 and horizontal_lines >= 1:
                if debug:
                    logger.info(f"方法3检测到柱状图: 垂直线={vertical_lines}, 水平线={horizontal_lines}")
                return True
        
        # 方法4: 检测图表中常见的网格特征
        # 应用二值化突出显示图表网格
        _, binary_grid = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 统计水平和垂直像素密度
        h_density = np.sum(binary_grid, axis=1) / width
        v_density = np.sum(binary_grid, axis=0) / height
        
        # 寻找密度峰值，表示可能的网格线
        h_peaks = np.sum(h_density > np.mean(h_density) * 1.5)
        v_peaks = np.sum(v_density > np.mean(v_density) * 1.5)
        
        # 如果水平和垂直方向都有明显峰值，可能是带网格的图表
        if h_peaks >= 2 and v_peaks >= 3:
            if debug:
                logger.info(f"方法4检测到柱状图: 水平峰值={h_peaks}, 垂直峰值={v_peaks}")
            return True
        
        return False
    except Exception as e:
        logger.error(f"柱状图检测错误: {str(e)}")
        return False

def extract_date_from_filename(filename):
    """从文件名中提取日期并转换为YYYY-MM-DD格式"""
    try:
        # 尝试匹配格式："xxxx年xx月运行日报（xx.xx）"
        pattern = r'(\d{4})年(\d{1,2})月.*?[（(](\d{1,2})\.(\d{1,2})[)）]'
        match = re.search(pattern, filename)
        
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)  # 确保月份是两位数
            day = match.group(4).zfill(2)  # 使用第二个数字作为日期，并确保是两位数
            return f"{year}-{month}-{day}"
            
        # 如果上面的模式不匹配，尝试其他格式
        # 例如：尝试直接从文件名中提取日期(如果文件名包含日期字符串)
        date_patterns = [
            r'(\d{4})[/-]?(\d{1,2})[/-]?(\d{1,2})',  # YYYY-MM-DD 或 YYYY/MM/DD
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'      # DD-MM-YYYY 或 DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.group(3)) == 4:  # 如果第三组是年份(YYYY)
                    day = match.group(1).zfill(2)
                    month = match.group(2).zfill(2)
                    year = match.group(3)
                    return f"{year}-{month}-{day}"
                else:  # 第一组是年份(YYYY)
                    year = match.group(1)
                    month = match.group(2).zfill(2)
                    day = match.group(3).zfill(2)
                    return f"{year}-{month}-{day}"
        
        # 如果无法提取日期，返回当前日期
        return datetime.now().strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"提取日期出错: {str(e)}")
        return datetime.now().strftime("%Y-%m-%d")

def extract_images_from_pdf(pdf_path, output_folder, debug=False):
    # 打开 PDF 文件
    document = fitz.open(pdf_path)
    
    # 获取PDF文件名（不含扩展名）
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 从文件名中提取日期
    date_format = extract_date_from_filename(pdf_filename)
    
    if debug:
        logger.info(f"处理PDF: {pdf_filename}, 提取的日期: {date_format}, 共{len(document)}页")
    
    # 检查是否已存在对应日期的图片
    existing_images = glob.glob(os.path.join(output_folder, f"{date_format}.*"))
    if existing_images:
        if debug:
            logger.info(f"已存在日期 {date_format} 的图片: {os.path.basename(existing_images[0])}, 跳过处理")
        document.close()
        return "SKIPPED"  # 返回特殊标记表示跳过
    
    # 遍历所有页面查找柱状图
    for page_number in range(len(document)):
        if debug:
            logger.info(f"处理第{page_number+1}页")
        
        page = document.load_page(page_number)
        images = page.get_images(full=True)
        
        if debug:
            logger.info(f"第{page_number+1}页找到{len(images)}张图片")
        
        # 检查页面上的每张图片
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 判断图像是否为柱状图
            if is_bar_chart(image_bytes, debug):
                # 是柱状图，保存图像
                image_ext = base_image["ext"]
                # 使用提取的日期格式作为文件名
                image_filename = f"{date_format}.{image_ext}"
                image_path = os.path.join(output_folder, image_filename)
                
                if debug:
                    logger.info(f"找到柱状图! 页码:{page_number+1}, 索引:{img_index}, 保存为:{image_filename}")
                
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                # 找到并保存第一张柱状图后立即退出
                document.close()
                return True
            elif debug:
                logger.info(f"第{page_number+1}页图片{img_index}不是柱状图")
    
    if debug:
        logger.info(f"在PDF '{pdf_filename}' 中未找到柱状图")
    
    document.close()
    return False

def process_input(input_path, output_dir=None, debug=False):
    # 确定输出目录
    if output_dir:
        images_folder = os.path.join(output_dir, "images")
    else:
        # 如果输入是文件，使用其所在目录
        if os.path.isfile(input_path):
            parent_dir = os.path.dirname(input_path)
            images_folder = os.path.join(parent_dir, "images")
        # 如果输入是目录，在该目录下创建images文件夹
        else:
            images_folder = os.path.join(input_path, "images")
    
    # 创建统一的images目录
    os.makedirs(images_folder, exist_ok=True)
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        # 单个 PDF 文件
        result = extract_images_from_pdf(input_path, images_folder, debug)
        if result == "SKIPPED":
            print(f"文件 {os.path.basename(input_path)} 对应的图片已存在，跳过处理")
        elif not result:
            print(f"在文件 {os.path.basename(input_path)} 中未找到柱状图")
    elif os.path.isdir(input_path):
        # 文件夹中的所有PDF文件
        found_any = False
        skipped_count = 0
        processed_count = 0
        
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_path, filename)
                result = extract_images_from_pdf(pdf_path, images_folder, debug)
                
                if result == "SKIPPED":
                    skipped_count += 1
                    if debug:
                        print(f"文件 {filename} 对应的图片已存在，跳过处理")
                elif result:
                    found_any = True
                    processed_count += 1
                else:
                    processed_count += 1
                    print(f"在文件 {filename} 中未找到柱状图")
        
        # 打印统计信息
        if skipped_count > 0:
            print(f"共跳过 {skipped_count} 个已有图片的文件")
        
        if processed_count > 0 and not found_any:
            print(f"在处理的 {processed_count} 个PDF文件中均未找到柱状图")
    else:
        print("输入路径无效，请提供 PDF 文件或包含 PDF 文件的文件夹。")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("用法: python main.py <pdf文件路径或文件夹路径> [输出目录] [--debug]")
    else:
        input_path = sys.argv[1]
        output_dir = None
        debug_mode = False
        
        # 解析参数
        for arg in sys.argv[2:]:
            if arg == "--debug":
                debug_mode = True
            else:
                output_dir = arg
        
        process_input(input_path, output_dir, debug_mode)
