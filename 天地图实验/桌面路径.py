import requests
import geopandas as gpd
import numpy as np
from PIL import Image
from io import BytesIO
import math
import time
from tqdm import tqdm
import os
# 查找GDAL数据路径（可通过print(os.__file__)确认conda环境路径）
GDAL_DATA_PATH =r"D:\env\anaconda\Lib\..\Library\share\gdal"


# 天地图配置
TIANDITU_KEY = "51330f6b053b9e67161305351c58cc32"  # 务必替换为有效的浏览器端Key
TIANDITU_VEC_URL = "http://t0.tianditu.gov.cn/vec_w/wmts"  # 矢量底图服务

# 合规配置
MAX_TILES_PER_REQUEST = 50  # 单次最大请求瓦片数
REQUEST_INTERVAL = 0.5  # 请求间隔(秒)
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")  # 桌面路径
CACHE_DIR = os.path.join(DESKTOP_PATH, "tile_cache")  # 桌面上的缓存目录
TILE_SIZE = 256  # 天地图瓦片尺寸为256x256像素


def deg2tile(lat_deg, lon_deg, zoom):
    """将经纬度坐标转换为天地图WMTS瓦片坐标（Web墨卡托，Y轴从上到下）"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def get_tile(z, x, y, layer='vec', style='default', format='image/jpeg'):
    """获取单个瓦片，支持缓存和速率限制"""
    cache_dir = os.path.join(CACHE_DIR, str(z))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{x}_{y}.jpg")

    # 读取缓存
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path)
        except:
            pass

    # 构造请求
    params = {
        'SERVICE': 'WMTS',
        'REQUEST': 'GetTile',
        'VERSION': '1.0.0',
        'LAYER': layer,
        'STYLE': style,
        'TILEMATRIXSET': 'w',  # Web墨卡托坐标系
        'TILEMATRIX': z,
        'TILEROW': y,
        'TILECOL': x,
        'FORMAT': format,
        'tk': TIANDITU_KEY
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
        'Referer': 'http://localhost'  # 必须与白名单中的域名一致
    }

    try:
        response = requests.get(TIANDITU_VEC_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        # 保存缓存
        with open(cache_path, 'wb') as f:
            f.write(response.content)

        time.sleep(REQUEST_INTERVAL)  # 遵守请求间隔
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"瓦片 {z}/{x}/{y} 下载失败: {e}")
        time.sleep(REQUEST_INTERVAL * 2)  # 错误时延长间隔
        return None


def main():
    geojson_file = r"C:\Users\Administrator\Desktop\参数文件\郑州市_市.geojson"  # 替换为实际文件路径

    # 读取GeoJSON并检查坐标系
    try:
        gdf = gpd.read_file(geojson_file)
        if gdf.crs != 'EPSG:4326':  # 确保为经纬度坐标系
            gdf = gdf.to_crs('EPSG:4326')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 计算区域边界（lonmin, latmin, lonmax, latmax）
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

    # 设置缩放级别（建议10-14级，根据区域大小调整）
    zoom = 10
    print(f"缩放级别: {zoom}, 区域范围: {min_lon:.2f}E~{max_lon:.2f}E, {min_lat:.2f}N~{max_lat:.2f}N")

    # 转换边界为瓦片坐标
    min_x, min_y = deg2tile(min_lat, min_lon, zoom)  # 注意参数顺序：lat在前，lon在后
    max_x, max_y = deg2tile(max_lat, max_lon, zoom)

    # 修正瓦片范围（确保在有效区间内）
    max_tile = 2 ** zoom - 1
    min_x, max_x = max(0, min_x), min(max_tile, max_x)
    min_y, max_y = max(0, min_y), min(max_tile, max_y)

    # 确保瓦片范围正确（防止min > max）
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y  # 交换Y轴方向，解决负数问题

    # 计算瓦片数量
    x_tiles = max_x - min_x + 1
    y_tiles = max_y - min_y + 1
    total_tiles = x_tiles * y_tiles

    # 限制最大请求瓦片数
    if total_tiles > MAX_TILES_PER_REQUEST:
        print(f"警告: 计划下载 {total_tiles} 瓦片，超过限制{MAX_TILES_PER_REQUEST}，自动缩小范围")
        x_adj = min(MAX_TILES_PER_REQUEST // y_tiles, x_tiles)
        y_adj = min(MAX_TILES_PER_REQUEST // x_adj, y_tiles)
        max_x = min_x + x_adj - 1
        max_y = min_y + y_adj - 1
        x_tiles, y_tiles = x_adj, y_adj
        total_tiles = x_tiles * y_tiles

    print(f"下载范围: 列{x_tiles}×行{y_tiles}，共{total_tiles}瓦片")

    # 创建空白画布
    if x_tiles <= 0 or y_tiles <= 0:
        print(f"错误: 计算的瓦片数量无效（x={x_tiles}, y={y_tiles}），请尝试调整缩放级别")
        return

    result_img = Image.new('RGB', (x_tiles * TILE_SIZE, y_tiles * TILE_SIZE))

    # 下载并拼接瓦片
    for y in tqdm(range(min_y, max_y + 1), desc="下载行"):
        for x in range(min_x, max_x + 1):
            tile = get_tile(zoom, x, y)
            if tile:
                x_pos = (x - min_x) * TILE_SIZE
                y_pos = (y - min_y) * TILE_SIZE
                result_img.paste(tile, (x_pos, y_pos))

    # 保存结果到桌面
    output_path = os.path.join(DESKTOP_PATH, "拼接结果.jpg")
    result_img.save(output_path)
    print(f"拼接完成，保存至: {output_path}")

    # 备用保存方案
    if 'result_img' in locals():
        try:
            manual_path = os.path.join(DESKTOP_PATH, "manual_save.jpg")
            result_img.save(manual_path)
            print(f"备用保存成功：{manual_path}")
        except Exception as e:
            print(f"备用保存失败：{e}")


if __name__ == "__main__":
    main()