import os
import re
import requests
from io import BytesIO
from PIL import Image

def get_video_id(url: str) -> str:
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu.be\/|youtube.com\/embed\/)([^&\n?#]+)',
        r'youtube.com\/shorts\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("无效的 YouTube URL: 无法提取视频 ID")

def get_thumbnail_url(video_id: str) -> str:
    thumbnail_formats = [
        f'https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg',
        f'https://i.ytimg.com/vi/{video_id}/sddefault.jpg',
        f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg'
    ]
    for url in thumbnail_formats:
        response = requests.head(url)
        if response.status_code == 200:
            return url
    raise Exception("无法获取可用的缩略图 URL")

def scale_image(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = img.size
    scale_w = target_width / width
    scale_h = target_height / height
    scale = max(scale_w, scale_h)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def process_thumbnail(url: str, min_width: int = 1200, min_height: int = 900):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("下载缩略图失败，HTTP 状态码非 200")
    img = Image.open(BytesIO(response.content))
    width, height = img.size
    if width < min_width or height < min_height:
        print(f"原始分辨率 ({width}x{height}) 低于要求，正在放大...")
        img = scale_image(img, min_width, min_height)
        width, height = img.size
        print(f"已放大至 {width}x{height}")
    output = BytesIO()
    img.save(output, format='JPEG', quality=95)
    return output.getvalue(), width, height

def save_thumbnail(content: bytes, video_id: str) -> str:
    """
    将缩略图保存到桌面路径。跨平台写法：
    - Mac/Linux: ~/Desktop
    - Windows: C:\\Users\\<用户名>\\Desktop
    """
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop_path):
        raise Exception(f"桌面路径不存在: {desktop_path}")

    filename = os.path.join(desktop_path, f"thumbnail_{video_id}.jpg")
    with open(filename, 'wb') as f:
        f.write(content)
    return filename

def main():
    url = ""
    min_width = 1200
    min_height = 900

    try:
        print(f"目标链接: {url}")
        video_id = get_video_id(url)
        print(f"视频 ID: {video_id}")
        thumbnail_url = get_thumbnail_url(video_id)
        print(f"可用缩略图 URL: {thumbnail_url}")
        content, width, height = process_thumbnail(thumbnail_url, min_width, min_height)
        print(f"最终图片分辨率: {width}x{height}")
        filename = save_thumbnail(content, video_id)
        print(f"缩略图已保存为: {filename}")
    except Exception as e:
        print(f"处理过程中出现错误：{e}")

if __name__ == '__main__':
    main()
