import os
import requests
import geopandas as gpd
from PIL import Image
from io import BytesIO
import math
import time
from tqdm import tqdm

# 天地图配置
TIANDITU_KEY = "51330f6b053b9e67161305351c58cc32"
TIANDITU_VEC_URL = "http://t0.tianditu.gov.cn/vec_w/wmts"

# 系统配置
MAX_TILES_PER_REQUEST = 50
REQUEST_INTERVAL = 0.5
CACHE_DIR = "tile_cache"
TILE_SIZE = 256


def deg2tile(lat_deg, lon_deg, zoom):
    """优化的坐标转换函数"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    try:
        ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / (2 * math.pi)) * n)
    except:
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def get_tile(z, x, y):
    """增强错误处理的瓦片下载"""
    cache_dir = os.path.join(CACHE_DIR, str(z))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{x}_{y}.jpg")

    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path)
        except IOError:
            os.remove(cache_path)  # 删除损坏的缓存

    params = {
        'SERVICE': 'WMTS',
        'REQUEST': 'GetTile',
        'VERSION': '1.0.0',
        'LAYER': 'vec',
        'STYLE': 'default',
        'TILEMATRIXSET': 'w',
        'TILEMATRIX': z,
        'TILEROW': y,
        'TILECOL': x,
        'FORMAT': 'image/jpeg',
        'tk': TIANDITU_KEY
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
        'Referer': 'http://localhost'
    }

    try:
        response = requests.get(TIANDITU_VEC_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        with open(cache_path, 'wb') as f:
            f.write(response.content)

        time.sleep(REQUEST_INTERVAL)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"\n瓦片 {z}/{x}/{y} 错误: {str(e)[:80]}")
        return None


def main():
    # 输入文件配置
    geojson_path = r"C:\Users\Administrator\Desktop\郑州市_市.geojson"
    output_filename = "郑州地图拼接结果.jpg"

    # 动态获取桌面路径
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop_dir, output_filename)

    try:
        # 读取地理数据
        gdf = gpd.read_file(geojson_path).to_crs('EPSG:4326')
        min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

        # 推荐缩放级别逻辑
        area = (max_lon - min_lon) * (max_lat - min_lat)
        zoom = 12 if area < 1 else 10  # 根据区域面积自动调整
        print(f"自动选择缩放级别: z{zoom}")

        # 计算瓦片范围
        tiles = [
            deg2tile(min_lat, min_lon, zoom),
            deg2tile(max_lat, max_lon, zoom),
            deg2tile(min_lat, max_lon, zoom),
            deg2tile(max_lat, min_lon, zoom)
        ]
        min_x = min(t[0] for t in tiles)
        max_x = max(t[0] for t in tiles)
        min_y = min(t[1] for t in tiles)
        max_y = max(t[1] for t in tiles)

        # 有效性校验
        if (max_x - min_x) * (max_y - min_y) > MAX_TILES_PER_REQUEST:
            raise ValueError(f"所需瓦片数超过最大限制 {MAX_TILES_PER_REQUEST}，请降低缩放级别")

        # 创建画布
        img_width = (max_x - min_x + 1) * TILE_SIZE
        img_height = (max_y - min_y + 1) * TILE_SIZE
        result_img = Image.new('RGB', (img_width, img_height))

        # 进度条显示
        total = (max_x - min_x + 1) * (max_y - min_y + 1)
        progress = tqdm(total=total, desc="下载瓦片", unit='tile')

        # 瓦片下载与拼接
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                tile = get_tile(zoom, x, y)
                if tile:
                    position = (
                        (x - min_x) * TILE_SIZE,
                        (y - min_y) * TILE_SIZE
                    )
                    result_img.paste(tile, position)
                progress.update(1)
        progress.close()

        # 保存结果
        os.makedirs(desktop_dir, exist_ok=True)
        result_img.save(output_path, quality=95)
        print(f"成功保存到: {output_path}")

        # 自动打开保存目录（仅Windows）
        if os.name == 'nt':
            os.startfile(desktop_dir)

    except Exception as e:
        print(f"\n处理失败: {str(e)}")
        print("常见问题排查：")
        print("1. 检查天地图密钥是否有效且Referer设置正确")
        print("2. 确认GeoJSON文件路径正确且为WGS84坐标系")
        print("3. 尝试降低缩放级别或缩小区域范围")
        print("4. 检查网络连接是否正常")


if __name__ == "__main__":
    main()