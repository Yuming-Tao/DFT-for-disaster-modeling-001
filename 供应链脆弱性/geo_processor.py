import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import logging
import os
import time
import hashlib
from shapely.geometry import Point
from config import get_amap_key

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------
# 路径配置
# --------------------------
# 预定义网格相关路径
FIXED_GRID_PATH = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"
GRID_CACHE_PATH = "data_cache/fixed_grid_cache.gpkg"  # 缓存网格矢量数据（多边形+属性）

# 距离矩阵相关路径
DIST_MATRIX_CACHE = "data_cache/distance_matrix.npy"  # 缓存距离矩阵数据
DIST_MATRIX_META = "data_cache/distance_matrix_meta.csv"  # 缓存元数据（坐标哈希）

# 确保缓存目录存在
os.makedirs(os.path.dirname(GRID_CACHE_PATH), exist_ok=True)


# --------------------------
# 预定义网格处理（含缓存）
# --------------------------
def load_fixed_grid(grid_path=FIXED_GRID_PATH, use_cache=True):
    """加载预定义网格（优先使用GPKG缓存）"""
    # 尝试加载网格缓存
    if use_cache and os.path.exists(GRID_CACHE_PATH):
        try:
            grid_gdf = gpd.read_file(GRID_CACHE_PATH)
            grid_id_col = _get_grid_id_column(grid_gdf)
            logger.info(f"使用网格缓存：{GRID_CACHE_PATH}（{len(grid_gdf)}个单元）")
            return grid_gdf, grid_id_col
        except Exception as e:
            logger.warning(f"网格缓存无效（{e}），将重新加载原始文件")

    # 加载原始SHP文件
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"网格文件不存在：{grid_path}")

    grid_gdf = gpd.read_file(grid_path)
    logger.info(f"加载原始网格：{grid_path}（{len(grid_gdf)}个单元）")

    # 转换为WGS84坐标系（与高德API一致）
    if grid_gdf.crs != "EPSG:4326":
        grid_gdf = grid_gdf.to_crs("EPSG:4326")
        logger.info("网格已转换为WGS84坐标系（EPSG:4326）")

    # 识别网格唯一ID列
    grid_id_col = _get_grid_id_column(grid_gdf)
    if not grid_id_col:
        raise ValueError("网格文件中未找到唯一标识列（如ID、grid_id、tid等）")

    # 保存网格缓存（GPKG格式，加载更快）
    try:
        grid_gdf.to_file(GRID_CACHE_PATH, driver="GPKG")
        logger.info(f"网格缓存已保存：{GRID_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"网格缓存保存失败（{e}），下次将重新加载原始文件")

    return grid_gdf, grid_id_col


def _get_grid_id_column(grid_gdf):
    """自动识别网格的唯一标识列 → 优先匹配小写`tid`"""
    # 【核心修改】将`tid`放在优先级最高的位置
    possible_ids = ['tid', 'TID', 'ID', 'id', 'grid_id', 'GRID_ID', '网格ID', 'FID', 'CODE']
    for col in possible_ids:
        if col in grid_gdf.columns and grid_gdf[col].nunique() == len(grid_gdf):
            logger.info(f"识别网格ID列：{col}（优先匹配小写`tid`）")
            return col
    # 查找非几何列中的唯一值列
    for col in grid_gdf.columns:
        if col != 'geometry' and grid_gdf[col].nunique() == len(grid_gdf):
            logger.warning(f"使用非标准网格ID列：{col}（建议规范为'tid'）")
            return col
    return None


def assign_to_fixed_grid(coords, grid_gdf, grid_id_col):
    """将坐标分配到预定义网格"""
    if not coords:
        return []

    # 过滤无效坐标
    valid_coords = [(lng, lat) for lng, lat in coords if lng is not None and lat is not None]
    valid_indices = [i for i, (lng, lat) in enumerate(coords) if lng is not None and lat is not None]
    grid_ids = [None] * len(coords)

    if not valid_coords:
        return grid_ids

    # 创建点要素GeoDataFrame
    point_gdf = gpd.GeoDataFrame(
        {'orig_index': valid_indices},
        geometry=[Point(lng, lat) for lng, lat in valid_coords],
        crs="EPSG:4326"
    )

    # 空间连接（点在多边形内）
    try:
        joined = gpd.sjoin(point_gdf, grid_gdf, how="left", predicate="within")
    except Exception as e:
        logger.error(f"空间连接失败：{e}")
        return grid_ids

    # 映射结果到原始坐标顺序
    for _, row in joined.iterrows():
        orig_idx = row['orig_index']
        grid_ids[orig_idx] = row[grid_id_col] if not pd.isna(row[grid_id_col]) else None

    matched_count = sum(1 for id in grid_ids if id is not None)
    logger.info(f"坐标分配完成：{matched_count}/{len(coords)}个点匹配到网格")
    return grid_ids


def get_grid_centers(grid_gdf, grid_id_col):
    """提取网格中心点坐标"""
    centers = []
    for _, row in grid_gdf.iterrows():
        try:
            centroid = row['geometry'].centroid
            centers.append({
                'grid_id': row[grid_id_col],  # 此处使用识别出的ID列（如'tid'）
                'lng': round(centroid.x, 6),  # 保留6位小数
                'lat': round(centroid.y, 6)
            })
        except Exception as e:
            logger.warning(f"计算网格{row[grid_id_col]}中心点失败：{e}")
    return pd.DataFrame(centers)


# --------------------------
# 距离矩阵计算（含缓存）
# --------------------------
def _get_coords_hash(coords):
    """计算坐标列表的哈希值（用于验证缓存有效性）"""
    # 坐标保留6位小数（约10厘米精度），减少哈希冲突
    coords_normalized = tuple(tuple(round(c, 6) for c in coord) for coord in coords)
    return hashlib.md5(str(coords_normalized).encode()).hexdigest()


def _save_distance_cache(matrix, coords):
    """保存距离矩阵缓存"""
    try:
        # 保存矩阵数据（.npy格式，高效存储numpy数组）
        np.save(DIST_MATRIX_CACHE, matrix)

        # 保存元数据（坐标哈希+时间戳）
        meta = pd.DataFrame({
            'coords_hash': [_get_coords_hash(coords)],
            'timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            'shape': [f"{matrix.shape[0]}x{matrix.shape[1]}"]
        })
        meta.to_csv(DIST_MATRIX_META, index=False)
        logger.info(f"距离矩阵缓存已保存（{matrix.shape[0]}x{matrix.shape[1]}）")
    except Exception as e:
        logger.warning(f"距离矩阵缓存保存失败：{e}")


def _load_distance_cache(coords):
    """加载距离矩阵缓存（若坐标匹配）"""
    if not os.path.exists(DIST_MATRIX_CACHE) or not os.path.exists(DIST_MATRIX_META):
        return None

    try:
        # 读取元数据
        meta = pd.read_csv(DIST_MATRIX_META)
        if len(meta) == 0:
            return None

        # 验证坐标哈希
        current_hash = _get_coords_hash(coords)
        if meta['coords_hash'].iloc[0] != current_hash:
            logger.info(f"坐标已变化（缓存哈希：{meta['coords_hash'].iloc[0]}，当前哈希：{current_hash}），缓存无效")
            return None

        # 读取矩阵数据
        matrix = np.load(DIST_MATRIX_CACHE)
        logger.info(f"使用距离矩阵缓存（{meta['shape'].iloc[0]}，生成于{meta['timestamp'].iloc[0]}）")
        return matrix
    except Exception as e:
        logger.warning(f"距离矩阵缓存加载失败：{e}")
        return None


def calculate_distance_matrix(coords, max_retry=3, batch_size=100):
    """计算坐标点间距离矩阵（公里），优先使用缓存"""
    n = len(coords)
    if n == 0:
        logger.warning("坐标列表为空，返回空矩阵")
        return np.zeros((0, 0))

    # 尝试加载缓存
    cached_matrix = _load_distance_cache(coords)
    if cached_matrix is not None:
        return cached_matrix

    # 缓存无效，重新计算
    logger.info(f"开始计算距离矩阵：{n}个点（每批处理{batch_size}个起点）")
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for orig_batch_start in range(0, n, batch_size):
        orig_batch_end = min(orig_batch_start + batch_size, n)
        orig_batch = coords[orig_batch_start:orig_batch_end]
        orig_batch_size = orig_batch_end - orig_batch_start
        origins_str = "|".join([f"{lng},{lat}" for lng, lat in orig_batch])
        logger.debug(f"处理起点批次：{orig_batch_start}~{orig_batch_end - 1}（{orig_batch_size}个点）")

        for dest_idx in range(n):
            dest_lng, dest_lat = coords[dest_idx]
            params = {
                'key': get_amap_key(),
                'origins': origins_str,
                'destination': f"{dest_lng},{dest_lat}",
                'type': 1  # 驾车距离
            }

            retry = 0
            success = False
            while retry < max_retry and not success:
                try:
                    time.sleep(0.5)  # 控制API调用频率
                    response = requests.get(
                        "https://restapi.amap.com/v3/distance",
                        params=params,
                        timeout=15
                    )
                    data = response.json()

                    if data.get('status') == '1':
                        results = data.get('results', [])
                        if len(results) == orig_batch_size:
                            # 填充距离矩阵（转换为公里）
                            for i in range(orig_batch_size):
                                global_orig_idx = orig_batch_start + i
                                distance = float(results[i].get('distance', 0)) / 1000
                                dist_matrix[global_orig_idx, dest_idx] = distance
                            success = True
                        else:
                            logger.warning(f"结果数量不匹配（预期{orig_batch_size}，实际{len(results)}）")
                            retry += 1
                    else:
                        logger.warning(f"API错误（终点{dest_idx}）：{data.get('info')}（错误码：{data.get('infocode')}）")
                        retry += 1
                        time.sleep(2)  # 错误时延长等待时间
                except Exception as e:
                    logger.error(f"请求异常（终点{dest_idx}，重试{retry + 1}）：{e}")
                    retry += 1
                    time.sleep(3)

            if not success:
                logger.error(f"处理失败：起点批次{orig_batch_start}~{orig_batch_end - 1} → 终点{dest_idx}")

    # 保存新缓存
    _save_distance_cache(dist_matrix, coords)
    logger.info(f"距离矩阵计算完成：{n}×{n}")
    return dist_matrix


# --------------------------
# 地理编码与路径规划（辅助功能）
# --------------------------
def geocode(address, city=None, max_retry=2):
    """地址转坐标（地理编码）"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        'key': get_amap_key(),
        'address': address,
        'output': 'JSON',
        'city': city
    }

    for _ in range(max_retry):
        try:
            response = requests.get(url, params=params, timeout=8)
            data = response.json()
            if data.get('status') == '1' and int(data.get('count', 0)) > 0:
                lng, lat = data['geocodes'][0]['location'].split(',')
                return float(lng), float(lat)
        except Exception as e:
            logger.error(f"地理编码失败（{address}）：{e}")
            time.sleep(0.5)
    logger.warning(f"地理编码失败：{address}")
    return None


def reverse_geocode(lng, lat):
    """坐标转地址（逆地理编码）"""
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        'key': get_amap_key(),
        'location': f"{lng},{lat}",
        'extensions': 'base'
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        if data.get('status') == '1':
            return data['regeocode']['formatted_address']
    except Exception as e:
        logger.error(f"逆地理编码失败（{lng},{lat}）：{e}")
    return None


def driving_route(origin, destination):
    """驾车路径规划"""
    url = "https://restapi.amap.com/v5/direction/driving"
    params = {
        'key': get_amap_key(),
        'origin': f"{origin[0]},{origin[1]}",
        'destination': f"{destination[0]},{destination[1]}",
        'show_fields': 'polyline',
        'strategy': '0'  # 速度优先
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get('status') != '1':
            logger.warning(f"路径规划失败：{data.get('info')}")
            return None

        path = data['route']['paths'][0]
        points = []
        for step in path.get('steps', []):
            for point in step.get('polyline', '').split(';'):
                if ',' in point:
                    lng, lat = point.split(',', 1)
                    points.append((float(lng), float(lat)))

        # 补充起点终点
        if points and points[0] != origin:
            points.insert(0, origin)
        if points and points[-1] != destination:
            points.append(destination)

        return {
            'distance': int(path.get('distance', 0)),
            'points': points or [origin, destination]
        }
    except Exception as e:
        logger.error(f"路径规划异常：{e}")
        return None


# --------------------------
# 测试函数
# --------------------------
def test_all_functions():
    """测试所有功能（含缓存机制）"""
    try:
        # 1. 测试网格缓存
        logger.info("=== 测试网格缓存 ===")
        grid_gdf, grid_id_col = load_fixed_grid()
        assert len(grid_gdf) > 0, "网格加载失败"
        assert grid_id_col in ['tid', 'TID'], "未优先匹配`tid`或`TID`"  # 验证ID列识别

        # 2. 测试坐标分配
        logger.info("=== 测试坐标分配 ===")
        test_coords = [(113.64, 34.75), (113.65, 34.76), (113.66, 34.77)]  # 郑州坐标
        grid_ids = assign_to_fixed_grid(test_coords, grid_gdf, grid_id_col)
        assert any(grid_ids), "坐标分配失败"

        # 3. 测试网格中心点
        logger.info("=== 测试网格中心点 ===")
        centers = get_grid_centers(grid_gdf, grid_id_col)
        assert not centers.empty, "提取网格中心点失败"

        # 4. 测试距离矩阵缓存
        logger.info("=== 测试距离矩阵缓存 ===")
        small_coords = test_coords  # 小规模坐标用于测试
        # 首次计算（无缓存）
        matrix1 = calculate_distance_matrix(small_coords)
        # 二次计算（使用缓存）
        matrix2 = calculate_distance_matrix(small_coords)
        assert np.allclose(matrix1, matrix2), "距离矩阵缓存不一致"

        logger.info("=== 所有功能测试通过 ===")
    except Exception as e:
        logger.error(f"测试失败：{e}", exc_info=True)


if __name__ == "__main__":
    test_all_functions()