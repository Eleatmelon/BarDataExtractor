import base64
import os
import argparse
import json
import getpass
from pathlib import Path
import cv2
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# 通过 pip install volcengine-python-sdk[ark] cryptography 安装必要库
try:
    from volcenginesdkarkruntime import Ark
    HAS_ARK_SDK = True
except ImportError:
    print("警告: 未安装volcengine-python-sdk，坐标识别功能将不可用")
    print("请运行: pip install volcengine-python-sdk[ark] cryptography")
    HAS_ARK_SDK = False

# 配置文件和加密相关设置
CONFIG_DIR = Path.home() / '.image_analyzer'
CONFIG_FILE = CONFIG_DIR / 'config.json'
KEY_FILE = CONFIG_DIR / '.key'
SALT = b'image_analyzer_salt'  # 固定盐值用于密钥派生

# 图片压缩阈值
MAX_WIDTH = 3000  # 最大横向像素
MAX_SIZE_MB = 1   # 最大文件大小（MB）

def get_or_create_key():
    """获取或创建加密密钥"""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(mode=0o700)  # 创建目录并设置只有所有者可访问
    
    if not KEY_FILE.exists():
        # 使用一个简单的密码和盐值生成密钥
        password = "default_password".encode()  # 在实际应用中可以使用更安全的方式
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=SALT,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        KEY_FILE.write_bytes(key)
        KEY_FILE.chmod(0o600)  # 设置只有所有者可读写
    
    return KEY_FILE.read_bytes()

def encrypt_api_key(api_key):
    """加密API密钥"""
    key = get_or_create_key()
    f = Fernet(key)
    return f.encrypt(api_key.encode()).decode()

def decrypt_api_key():
    """解密API密钥"""
    if not CONFIG_FILE.exists():
        return None
    
    try:
        config = json.loads(CONFIG_FILE.read_text())
        encrypted_key = config.get('api_key')
        if not encrypted_key:
            return None
        
        key = get_or_create_key()
        f = Fernet(key)
        return f.decrypt(encrypted_key.encode()).decode()
    except Exception as e:
        print(f"解密API密钥出错: {e}")
        return None

def save_api_key(api_key):
    """保存API密钥到配置文件"""
    encrypted_key = encrypt_api_key(api_key)
    
    config = {}
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
        except:
            pass
    
    config['api_key'] = encrypted_key
    CONFIG_FILE.write_text(json.dumps(config))
    CONFIG_FILE.chmod(0o600)  # 设置只有所有者可读写
    
    return True

def get_api_key():
    """获取API密钥，如果不存在则提示用户输入"""
    # 先检查环境变量
    api_key = os.getenv('ARK_API_KEY')
    if api_key:
        print("使用环境变量中的API密钥")
        return api_key
    
    # 尝试从配置文件获取
    api_key = decrypt_api_key()
    if api_key:
        print("使用已保存的API密钥")
        return api_key
    
    # 提示用户输入
    print("\n需要设置火山方舟API密钥才能进行坐标识别。")
    print("API密钥会被加密存储在您的本地系统中，后续使用无需再次输入。")
    api_key = getpass.getpass("请输入您的火山方舟API密钥: ")
    
    if api_key:
        save_api_key(api_key)
        print("API密钥已加密保存")
        return api_key
    
    return None

# 有条件地初始化客户端
client = None
api_key = get_api_key()

if HAS_ARK_SDK and api_key:
    try:
        client = Ark(api_key=api_key)
    except Exception as e:
        print(f"初始化Ark客户端失败: {e}")
        print("坐标识别功能将不可用")

# 定义方法将指定路径图片转为Base64编码
def encode_image(image_path):
    """
    将图像转换为Base64编码，如果图片过大则进行压缩
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        str: Base64编码的图像数据
    """
    # 检查文件大小
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    
    # 检查图像尺寸
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    height, width = img.shape[:2]
    
    # 如果图片不需要压缩，直接编码
    if width <= MAX_WIDTH and file_size_mb <= MAX_SIZE_MB:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # 需要压缩
    print(f"图片需要压缩: 宽度 {width}px (阈值: {MAX_WIDTH}px), 大小 {file_size_mb:.2f}MB (阈值: {MAX_SIZE_MB}MB)")
    
    # 计算缩放比例
    scale = 1.0
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
    
    # 初始化压缩质量
    quality = 90
    
    # 如果仅需要调整尺寸
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"调整图片尺寸: {width}x{height} -> {new_width}x{new_height}")
    
    # 压缩图片
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)
    current_size_mb = len(encoded_img) / (1024 * 1024)
    
    # 如果还是过大，继续调整质量
    while current_size_mb > MAX_SIZE_MB and quality > 30:
        quality -= 10
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img, encode_param)
        current_size_mb = len(encoded_img) / (1024 * 1024)
    
    print(f"图片压缩完成: 最终质量 {quality}, 大小 {current_size_mb:.2f}MB")
    return base64.b64encode(encoded_img.tobytes()).decode('utf-8')

# 定义函数，接收图片路径作为参数
def process_image(image_path):
    # 将图片转为Base64编码
    base64_image = encode_image(image_path)
    return base64_image

# 新增函数：调用API获取坐标信息
def get_axis_coordinates(image_path):
    """
    调用图像识别API获取坐标信息并解析成变量
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        tuple: (x_max, y_max) 横纵坐标的最大值
    """
    # 检查客户端是否可用
    if client is None:
        print("错误: Ark客户端未初始化或API密钥未设置")
        print("请重新运行并在提示时输入API密钥")
        return (None, None)
    
    # 将图片转为Base64编码
    base64_image = process_image(image_path)
    
    try:
        response = client.chat.completions.create(
            model="doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "输出横坐标和纵坐标的最大值,如果你识别到了(xxx,xxx]这样的格式，你取后面那个数，分别以\"横坐标[]；纵坐标[]\"的形式返回，不要生成任何其他内容",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        
        # 从API返回结果中提取文本内容
        result_text = response.choices[0].message.content
        
        # 解析文本，提取坐标值
        import re
        
        # 使用正则表达式提取括号中的数字
        x_match = re.search(r'横坐标\[(\d+)\]', result_text)
        y_match = re.search(r'纵坐标\[(\d+)\]', result_text)
        
        if x_match and y_match:
            x_max = int(x_match.group(1))
            y_max = int(y_match.group(1))
            print(f"解析得到坐标: x_max={x_max}, y_max={y_max}")
            return (x_max, y_max)
        else:
            print(f"无法解析坐标信息，原始文本: {result_text}")
            return (None, None)  # 不提供默认值，返回None让调用者处理
    except Exception as e:
        print(f"API调用失败: {e}")
        return (None, None)

# 如果直接运行此文件，则通过命令行参数获取图片路径
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图像坐标识别工具')
    parser.add_argument('image_path', type=str, help='图像文件的路径')
    parser.add_argument('--reset_api_key', action='store_true', help='重置API密钥')
    args = parser.parse_args()
    
    # 检查是否需要重置API密钥
    if args.reset_api_key:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        print("API密钥已重置，将在下次运行时提示输入新的密钥")
        import sys
        sys.exit(0)
    
    # 检查文件是否存在
    if not os.path.exists(args.image_path):
        print(f"错误：文件 {args.image_path} 不存在")
        import sys
        sys.exit(1)
    
    # 调用函数处理图片并获取坐标
    x_max, y_max = get_axis_coordinates(args.image_path)
    print(f"提取的坐标值: x_max={x_max}, y_max={y_max}")