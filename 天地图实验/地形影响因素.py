import os
import requests
import geopandas as gpd
import numpy as np
from PIL import Image
from io import BytesIO
import math
import time
from tqdm import tqdm
from osgeo import gdal, osr

# 天地图配置
TIANDITU_KEY = "51330f6b053b9e67161305351c58cc32"  # 务必替换为有效的浏览器端Key
TIANDITU_VEC_URL = "http://t0.tianditu.gov.cn/vec_w/wmts"  # 矢量底图服务
TIANDITU_IMG_URL = "http://t0.tianditu.gov.cn/img_w/wmts"  # 影像底图服务

# 合规配置
MAX_TILES_PER_REQUEST = 100  # 单次最大请求瓦片数
REQUEST_INTERVAL = 0.5  # 请求间隔(秒)
CACHE_DIR = "tile_cache"  # 瓦片缓存目录
TILE_SIZE = 256  # 天地图瓦片尺寸为256x256像素


def deg2tile(lat_deg, lon_deg, zoom):
    """将经纬度坐标转换为天地图WMTS瓦片坐标（Web墨卡托，Y轴从上到下）"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def tile2deg(xtile, ytile, zoom):
    """将瓦片坐标转换为经纬度坐标（返回瓦片左上角点）"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile(z, x, y, layer='vec', style='default', format='image/jpeg'):
    """获取单个瓦片，支持缓存和速率限制"""
    cache_dir = os.path.join(CACHE_DIR, layer, str(z))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{x}_{y}.jpg")

    # 读取缓存
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path)
        except:
            pass

    # 根据图层选择URL
    base_url = TIANDITU_VEC_URL if layer == 'vec' else TIANDITU_IMG_URL

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
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        # 检查是否返回有效图片
        if response.headers.get('Content-Type') != 'image/jpeg':
            print(f"无效响应类型: {response.headers.get('Content-Type')}")
            print(f"响应内容: {response.text[:200]}...")
            return None

        # 保存缓存
        with open(cache_path, 'wb') as f:
            f.write(response.content)

        time.sleep(REQUEST_INTERVAL)  # 遵守请求间隔
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"瓦片 {z}/{x}/{y} 下载失败: {e}")
        time.sleep(REQUEST_INTERVAL * 2)  # 错误时延长间隔
        return None


def save_as_geotiff(image, min_lon, min_lat, max_lon, max_lat, output_path):
    """将影像保存为GeoTIFF格式，添加地理参考信息"""
    width, height = image.size

    # 创建GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_path,
        width,
        height,
        3,  # 3个波段 (RGB)
        gdal.GDT_Byte
    )

    # 设置地理参考（左上角坐标、像素宽度、旋转参数）
    geotransform = (
        min_lon,  # 左上角X坐标（经度）
        (max_lon - min_lon) / width,  # 像素宽度（经度差/像素数）
        0,  # X方向旋转
        max_lat,  # 左上角Y坐标（纬度）
        0,  # Y方向旋转
        -(max_lat - min_lat) / height  # 像素高度（负值表示从上到下）
    )

    dataset.SetGeoTransform(geotransform)

    # 设置投影（WGS 84）
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS 84
    dataset.SetProjection(srs.ExportToWkt())

    # 写入数据
    bands = list(image.split())
    for i, band in enumerate(bands):
        dataset.GetRasterBand(i + 1).WriteArray(np.array(band))

    # 保存并关闭
    dataset = None
    print(f"已保存GeoTIFF文件: {output_path}")


def main():
    geojson_file = r"C:\Users\Administrator\Desktop\郑州市_市.geojson"  # 替换为实际文件路径

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

    # 让用户选择图层类型
    layer_choice = input("请选择图层类型 (1=矢量图, 2=影像图, 默认=1): ").strip()
    layer = 'vec' if layer_choice != '2' else 'img'

    # 设置缩放级别（建议10-16级，根据区域大小调整）
    while True:
        try:
            zoom = int(input(f"请输入缩放级别 (建议10-16，默认=12): ").strip() or "12")
            if 10 <= zoom <= 16:
                break
            else:
                print("缩放级别超出范围，请重新输入")
        except ValueError:
            print("请输入有效的整数")

    print(f"图层类型: {layer}，缩放级别: {zoom}")
    print(f"区域范围: {min_lon:.6f}E~{max_lon:.6f}E, {min_lat:.6f}N~{max_lat:.6f}N")

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
        min_y, max_y = max_y, min_y

    # 计算瓦片数量
    x_tiles = max_x - min_x + 1
    y_tiles = max_y - min_y + 1
    total_tiles = x_tiles * y_tiles

    # 限制最大请求瓦片数
    if total_tiles > MAX_TILES_PER_REQUEST:
        print(f"警告: 计划下载 {total_tiles} 瓦片，超过限制{MAX_TILES_PER_REQUEST}")
        proceed = input("是否继续下载？(y/n, 默认=n): ").strip().lower() or "n"
        if proceed != "y":
            print("操作已取消")
            return

        # 自动缩小范围
        x_adj = min(MAX_TILES_PER_REQUEST // y_tiles, x_tiles)
        y_adj = min(MAX_TILES_PER_REQUEST // x_adj, y_tiles)
        max_x = min_x + x_adj - 1
        max_y = min_y + y_adj - 1
        x_tiles, y_tiles = x_adj, y_adj
        total_tiles = x_tiles * y_tiles
        print(f"已自动调整下载范围: 列{x_tiles}×行{y_tiles}，共{total_tiles}瓦片")
    else:
        print(f"下载范围: 列{x_tiles}×行{y_tiles}，共{total_tiles}瓦片")

    # 创建空白画布
    if x_tiles <= 0 or y_tiles <= 0:
        print(f"错误: 计算的瓦片数量无效（x={x_tiles}, y={y_tiles}），请尝试调整缩放级别")
        return

    result_img = Image.new('RGB', (x_tiles * TILE_SIZE, y_tiles * TILE_SIZE))

    # 下载并拼接瓦片
    print("开始下载瓦片...")
    success_count = 0
    for y in tqdm(range(min_y, max_y + 1), desc="下载行"):
        for x in range(min_x, max_x + 1):
            tile = get_tile(zoom, x, y, layer=layer)
            if tile:
                x_pos = (x - min_x) * TILE_SIZE
                y_pos = (y - min_y) * TILE_SIZE
                result_img.paste(tile, (x_pos, y_pos))
                success_count += 1

    print(f"瓦片下载完成，成功{success_count}/{total_tiles}")

    # 计算实际地理范围（考虑瓦片边界）
    actual_min_lat, actual_min_lon = tile2deg(min_x, min_y, zoom)
    actual_max_lat, actual_max_lon = tile2deg(max_x + 1, min_y + 1, zoom)
    print(f"实际地理范围: {actual_min_lon:.6f}E~{actual_max_lon:.6f}E, {actual_min_lat:.6f}N~{actual_max_lat:.6f}N")

    # 保存结果
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "郑州地形数据")
    os.makedirs(output_dir, exist_ok=True)

    # 保存为普通JPG
    jpg_path = os.path.join(output_dir, f"郑州_{layer}_{zoom}级.jpg")
    result_img.save(jpg_path)
    print(f"拼接完成，已保存JPG: {jpg_path}")

    # 保存为GeoTIFF（带地理参考）
    geotiff_path = os.path.join(output_dir, f"郑州_{layer}_{zoom}级.tif")
    save_as_geotiff(result_img, actual_min_lon, actual_min_lat, actual_max_lon, actual_max_lat, geotiff_path)


if __name__ == "__main__":
    main()