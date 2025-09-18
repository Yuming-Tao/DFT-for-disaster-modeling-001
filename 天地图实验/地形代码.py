import os
import requests
import geopandas as gpd
from PIL import Image
from io import BytesIO
import math
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# 服务配置
SERVICE_CONFIG = {
    '地形晕渲': {
        'c': ('ter_c', 'c', 'image/png'),
        'w': ('ter_w', 'w', 'image/png')
    },
    '地形注记': {
        'c': ('cta_c', 'c', 'image/png'),
        'w': ('cta_w', 'w', 'image/png')
    }
}

TIANDITU_KEY = "51330f6b053b9e67161305351c58cc32"
BASE_URL = "http://t0.tianditu.gov.cn/{service}/wmts"

# 系统配置
MAX_TILES_PER_REQUEST = 16024693260
REQUEST_INTERVAL = 0.5
TILE_SIZE = 256


def get_projection_params(projection_type):
    """获取投影参数"""
    return {
        'c': ('经纬度投影', 'EPSG:4326'),
        'w': ('球面墨卡托投影', 'EPSG:3857')
    }[projection_type]


def deg2tile(lat_deg, lon_deg, zoom, projection):
    """通用瓦片坐标转换"""
    if projection == 'c':
        n = 2 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((90.0 - lat_deg) / 180.0 * n)
    else:
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def download_tile(service_type, projection, z, x, y):
    """下载单个瓦片（支持多图层）"""
    layer, matrix_set, img_format = SERVICE_CONFIG[service_type][projection]
    cache_dir = os.path.join("tile_cache", service_type, projection, str(z))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{x}_{y}.png")

    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path), service_type
        except IOError:
            os.remove(cache_path)

    url = BASE_URL.format(service=layer)
    params = {
        'SERVICE': 'WMTS',
        'REQUEST': 'GetTile',
        'VERSION': '1.0.0',
        'LAYER': layer.split('_')[0],
        'STYLE': 'default',
        'TILEMATRIXSET': matrix_set,
        'TILEMATRIX': z,
        'TILEROW': y,
        'TILECOL': x,
        'FORMAT': img_format,
        'tk': TIANDITU_KEY
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
        'Referer': 'http://localhost'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        with open(cache_path, 'wb') as f:
            f.write(response.content)

        time.sleep(REQUEST_INTERVAL)
        return Image.open(BytesIO(response.content)), service_type
    except Exception as e:
        print(f"\n{service_type} 瓦片 {z}/{x}/{y} 错误: {str(e)[:80]}")
        return None, service_type


def composite_images(base_img, overlay_img):
    """合成地形底图与注记"""
    if base_img.size != overlay_img.size:
        raise ValueError("图像尺寸不一致")

    # 设置注记透明度（60%）
    overlay_img = overlay_img.convert("RGBA")
    overlay_data = overlay_img.getdata()
    new_overlay_data = []
    for item in overlay_data:
        if item[3] > 0:  # 非透明像素
            new_overlay_data.append((item[0], item[1], item[2], int(255 * 0.6)))
        else:
            new_overlay_data.append(item)
    overlay_img.putdata(new_overlay_data)

    return Image.alpha_composite(base_img.convert('RGBA'), overlay_img)


def process_tile(args):
    """并行处理瓦片下载任务"""
    x, y, projection, zoom = args
    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(download_tile, '地形晕渲', projection, zoom, x, y),
            executor.submit(download_tile, '地形注记', projection, zoom, x, y)
        }
        for future in futures:
            img, layer_type = future.result()
            if img:
                results[layer_type] = img
    return (x, y, results)


def main():
    # 用户输入
    projection = input("请选择投影方式（c=经纬度/w=墨卡托）: ").strip().lower()
    if projection not in ['c', 'w']:
        print("无效的输入参数！")
        return

    proj_name, epsg_code = get_projection_params(projection)
    geojson_path = r"C:\Users\Administrator\Desktop\郑州市_市.geojson"
    output_filename = f"郑州_地形叠加_{proj_name}.png"

    try:
        # 读取并转换坐标系
        gdf = gpd.read_file(geojson_path).to_crs(epsg_code)
        bounds = gdf.total_bounds

        # 自动计算推荐缩放级别
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        zoom = max(10, 14 - int(math.log10(area))) if projection == 'w' else 10

        print(f"投影: {proj_name} | 缩放级别: z{zoom}")

        # 计算瓦片范围
        min_x, min_y = deg2tile(bounds[1], bounds[0], zoom, projection)
        max_x, max_y = deg2tile(bounds[3], bounds[2], zoom, projection)

        # 修正瓦片范围
        max_tile = 2 ** zoom - 1
        min_x, max_x = sorted([max(0, min_x), min(max_tile, max_x)])
        min_y, max_y = sorted([max(0, min_y), min(max_tile, max_y)])

        # 校验瓦片数量
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        if total_tiles > MAX_TILES_PER_REQUEST:
            raise ValueError(f"所需瓦片数 {total_tiles} 超过限制 {MAX_TILES_PER_REQUEST}")

        # 创建画布
        canvas_size = ((max_x - min_x + 1) * TILE_SIZE,
                       (max_y - min_y + 1) * TILE_SIZE)
        terrain_img = Image.new('RGBA', canvas_size)
        annotate_img = Image.new('RGBA', canvas_size)

        # 生成任务列表
        tasks = [(x, y, projection, zoom)
                 for y in range(min_y, max_y + 1)
                 for x in range(min_x, max_x + 1)]

        # 并行下载并处理瓦片
        with tqdm(total=len(tasks), desc="处理进度") as progress:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_tile, task) for task in tasks]
                for future in futures:
                    x, y, results = future.result()
                    x_pos = (x - min_x) * TILE_SIZE
                    y_pos = (y - min_y) * TILE_SIZE

                    if '地形晕渲' in results:
                        terrain_img.paste(results['地形晕渲'], (x_pos, y_pos))
                    if '地形注记' in results:
                        annotate_img.paste(results['地形注记'], (x_pos, y_pos))

                    progress.update(1)

        # 合成最终图像
        final_img = composite_images(terrain_img, annotate_img)

        # 保存结果
        output_path = os.path.join(os.path.expanduser("~"), "Desktop", output_filename)
        final_img.save(output_path)
        print(f"\n成果已保存至: {output_path}")

    except Exception as e:
        print(f"\n处理失败: {str(e)}")
        print("常见问题排查：")
        print("1. 确认密钥同时支持地形和注记服务")
        print("2. 检查网络连接和API调用限制")
        print("3. 尝试降低缩放级别至12以下")


if __name__ == "__main__":
    main()