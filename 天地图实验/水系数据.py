import geopandas as gpd
import requests
from requests.exceptions import RequestException
import json
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import xml.etree.ElementTree as ET
import matplotlib.font_manager as fm

# ====================== 配置参数 ======================
TIANDITU_WFS_URL = "http://gisserver.tianditu.gov.cn/TDTService/wfs"  # 水系服务URL
TIANDITU_KEY = "51330f6b053b9e67161305351c58cc32"  # 替换为有效的浏览器端Key
OUTPUT_CRS = "EPSG:4326"  # 输出坐标系（经纬度）

# 目标区域（郑州市范围，经纬度）
TARGET_BBOX = (112.8, 34.5, 114.2, 34.9)  # (min_lon, min_lat, max_lon, max_lat)
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "water_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zhengzhou_water_features.geojson")

# 请求配置
MAX_RETRIES = 3  # 最大重试次数
REQUEST_DELAY = 1  # 请求间隔（秒）
FEATURES_PER_REQUEST = 1000  # 每次请求的最大要素数


# ====================== 工具函数 ======================
def get_wfs_capabilities() -> dict:
    """获取WFS服务能力信息，提取可用图层"""
    print("正在获取服务能力信息...")

    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetCapabilities",
        "tk": TIANDITU_KEY
    }

    try:
        response = requests.get(TIANDITU_WFS_URL, params=params, timeout=30)
        response.raise_for_status()

        # 保存能力文档用于调试
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "wfs_capabilities.xml"), "w", encoding="utf-8") as f:
            f.write(response.text)

        # 解析XML提取图层信息
        root = ET.fromstring(response.text)
        namespace = {'wfs': 'http://www.opengis.net/wfs'}

        layers = []
        for feature_type in root.findall('.//wfs:FeatureType', namespace):
            name = feature_type.find('wfs:Name', namespace).text
            title = feature_type.find('wfs:Title', namespace).text
            layers.append({'name': name, 'title': title})

        print(f"发现 {len(layers)} 个可用图层")
        return layers

    except Exception as e:
        print(f"获取服务能力信息失败: {e}")
        return []


def detect_water_layers(layers: list) -> list:
    """从所有图层中检测可能的水系图层"""
    water_keywords = ["water", "hyd", "river", "lake", "stream", "canal", "waterway"]
    water_layers = []

    for layer in layers:
        name = layer['name'].lower()
        title = layer['title'].lower() if 'title' in layer else ""

        if any(keyword in name or keyword in title for keyword in water_keywords):
            water_layers.append(layer['name'])

    print(f"检测到 {len(water_layers)} 个可能的水系图层")
    for layer in water_layers:
        print(f"- {layer}")

    return water_layers


def fetch_wfs_data(bbox: tuple, layer: str, start_index: int = 0) -> dict:
    """通过WFS服务获取矢量数据"""
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typename": layer,
        "srsname": OUTPUT_CRS,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{OUTPUT_CRS}",
        "outputFormat": "application/json",
        "maxFeatures": FEATURES_PER_REQUEST,
        "startIndex": start_index,
        "tk": TIANDITU_KEY
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            print(f"发送请求: {layer} (起始索引: {start_index})")
            response = requests.get(
                TIANDITU_WFS_URL,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            # 验证响应是否为JSON格式
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                raise ValueError(f"无效响应类型: {content_type}，内容: {response.text[:100]}...")

            data = response.json()
            return data

        except RequestException as e:
            print(f"请求失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))  # 指数退避
            else:
                raise


def get_all_water_features(bbox: tuple, layer: str) -> gpd.GeoDataFrame:
    """获取指定区域内的所有水系要素（支持分页）"""
    all_features = []
    start_index = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"正在从图层 '{layer}' 获取水系数据...")

    while True:
        try:
            data = fetch_wfs_data(bbox, layer, start_index)

            # 检查是否有要素返回
            features = data.get('features', [])
            if not features:
                break

            all_features.extend(features)
            start_index += len(features)

            print(f"已获取 {start_index} 个要素")

            # 保存中间结果（防止中断）
            with open(os.path.join(OUTPUT_DIR, f"temp_features_{layer.replace(':', '_')}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": all_features}, f)

            # 检查是否需要继续请求
            if len(features) < FEATURES_PER_REQUEST:
                break

            time.sleep(REQUEST_DELAY)  # 避免请求过于频繁

        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            break

    if not all_features:
        print("未获取到任何水系要素")
        return gpd.GeoDataFrame()

    # 转换为GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features)

    # 设置坐标系
    if gdf.crs is None:
        gdf = gdf.set_crs(OUTPUT_CRS)

    print(f"成功获取 {len(gdf)} 个水系要素")
    return gdf


def filter_water_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """过滤并清洗水系数据"""
    if gdf.empty:
        return gdf

    # 常见水系要素类型（中英文对照）
    water_types = [
        "河流", "运河", "湖泊", "水库", "溪流", "沟渠", "海洋", "池塘",
        "water", "river", "canal", "lake", "reservoir", "stream", "ditch", "ocean", "pond"
    ]

    # 检查可用字段
    detected_fields = []
    for field in ["FCLASS", "type", "TYPE", "class", "NAME", "name"]:
        if field in gdf.columns:
            detected_fields.append(field)

    if not detected_fields:
        print("警告: 未找到识别水系要素的字段，保留所有要素")
        return gdf

    print(f"可用字段: {', '.join(detected_fields)}")

    # 优先使用类型字段过滤
    filtered_gdf = gdf.copy()
    for field in detected_fields:
        if field.lower() in ["fclass", "type", "class", "type"]:
            filtered_gdf = gdf[gdf[field].astype(str).str.lower().isin([t.lower() for t in water_types])]
            if not filtered_gdf.empty:
                break

    # 如果类型字段过滤后为空，尝试使用名称字段
    if filtered_gdf.empty and "NAME" in detected_fields:
        print("尝试通过名称字段识别水系要素...")
        water_keywords = ["河", "湖", "水库", "沟", "渠", "塘", "水", "溪", "江", "海", "池"]
        filtered_gdf = gdf[gdf["NAME"].astype(str).str.contains('|'.join(water_keywords), na=False)]

    print(f"过滤前: {len(gdf)} 个要素 -> 过滤后: {len(filtered_gdf)} 个水系要素")
    return filtered_gdf


def detect_available_chinese_fonts() -> list:
    """检测系统中可用的中文字体"""
    chinese_fonts = []
    chinese_keywords = ['黑', '宋', '仿宋', '楷', '微软', '文泉驿', 'Heiti', 'Sim']

    for font in fm.fontManager.ttflist:
        font_name = font.name
        if any(keyword in font_name for keyword in chinese_keywords):
            chinese_fonts.append(font_name)

    return list(set(chinese_fonts))  # 去重


def visualize_water_data(gdf: gpd.GeoDataFrame, output_img: str = None) -> None:
    """Visualize water data"""
    if gdf.empty:
        print("No data to visualize")
        return

    # 检测系统中可用的中文字体（英文可视化无需中文字体，但保留检测逻辑）
    available_fonts = detect_available_chinese_fonts()

    if not available_fonts:
        print("Warning: No Chinese fonts detected, using default font")
        plt.rcParams["font.family"] = ["sans-serif"]
    else:
        print(f"Available Chinese fonts: {', '.join(available_fonts)}")
        plt.rcParams["font.family"] = available_fonts[0]  # 英文文本实际使用系统默认英文字体

    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))

    # 按类型绘制不同颜色
    if "FCLASS" in gdf.columns:
        water_types = gdf["FCLASS"].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(water_types)))

        for i, water_type in enumerate(water_types):
            subset = gdf[gdf["FCLASS"] == water_type]
            if "Line" in str(subset.geom_type.iloc[0]):
                subset.plot(ax=ax, color=colors[i], label=water_type, alpha=0.7, linewidth=1.5)
            else:
                subset.plot(ax=ax, color=colors[i], label=water_type, alpha=0.7)
    elif "type" in gdf.columns:
        water_types = gdf["type"].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(water_types)))

        for i, water_type in enumerate(water_types):
            subset = gdf[gdf["type"] == water_type]
            if "Line" in str(subset.geom_type.iloc[0]):
                subset.plot(ax=ax, color=colors[i], label=water_type, alpha=0.7, linewidth=1.5)
            else:
                subset.plot(ax=ax, color=colors[i], label=water_type, alpha=0.7)
    else:
        if "Line" in str(gdf.geom_type.iloc[0]):
            gdf.plot(ax=ax, color='blue', alpha=0.7, linewidth=1.5)
        else:
            gdf.plot(ax=ax, color='blue', alpha=0.7)

    # 修改为英文标题和标签
    ax.set_title("Zhengzhou Water Features Distribution Map", fontsize=16)
    if "FCLASS" in gdf.columns or "type" in gdf.columns:
        ax.legend(title="Water Type", loc="upper right")  # 图例标题改为英文
    ax.set_xlabel("Longitude", fontsize=12)  # 经度改为英文
    ax.set_ylabel("Latitude", fontsize=12)  # 纬度改为英文
    ax.grid(True, linestyle='--', alpha=0.5)

    # 添加郑州市边界参考（假设已有边界数据）
    try:
        boundary_file = os.path.join(os.path.dirname(OUTPUT_FILE), "zhengzhou_boundary.geojson")
        if os.path.exists(boundary_file):
            boundary = gpd.read_file(boundary_file)
            boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0)
    except Exception as e:
        print(f"Failed to add boundary reference: {e}")

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if output_img:
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_img}")

    plt.show()


# ====================== 主流程 ======================
def main():
    print(f"===== 天地图水系数据获取工具 =====")
    print(f"目标范围: {TARGET_BBOX[0]:.2f}E~{TARGET_BBOX[2]:.2f}E, {TARGET_BBOX[1]:.2f}N~{TARGET_BBOX[3]:.2f}N")
    print(f"输出目录: {OUTPUT_DIR}")

    try:
        # 检查Key是否设置
        if not TIANDITU_KEY or TIANDITU_KEY == "your_valid_key_here":
            print("错误: 请设置有效的天地图Key")
            return

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. 获取服务能力并检测水系图层
        capabilities = get_wfs_capabilities()
        if not capabilities:
            print("警告: 无法获取服务能力信息，使用预设图层列表")
            water_layers = ["TDT:HYD", "TDT:water", "TDT:rivers", "TDT:lakes", "TDT:waterways"]
        else:
            water_layers = detect_water_layers(capabilities)
            if not water_layers:
                print("警告: 未检测到水系图层，使用预设图层列表")
                water_layers = ["TDT:HYD", "TDT:water", "TDT:rivers", "TDT:lakes", "TDT:waterways"]

        # 2. 尝试获取水系数据
        water_gdf = None

        for layer in water_layers:
            print(f"\n===== 尝试图层 '{layer}' =====")
            gdf = get_all_water_features(TARGET_BBOX, layer)

            if not gdf.empty:
                # 检测数据质量
                print("\n数据质量分析:")
                print(f"- 要素数量: {len(gdf)}")
                print(f"- 字段: {', '.join(gdf.columns.tolist())}")

                # 检查是否包含几何字段
                if gdf.geometry.empty:
                    print(f"警告: 图层 {layer} 不包含有效几何数据，跳过")
                    continue

                # 检查几何类型
                geom_types = gdf.geom_type.unique()
                print(f"- 几何类型: {', '.join(geom_types)}")

                water_gdf = gdf
                break

        if water_gdf is None or water_gdf.empty:
            print("错误: 尝试了所有图层，但未获取到任何水系数据")

            # 提供详细的排查建议
            print("\n===== 排查建议 =====")
            print("1. 确认天地图Key有效且已开通WFS服务权限")
            print("2. 检查服务URL是否正确 (当前使用: {TIANDITU_WFS_URL})")
            print("3. 手动检查wfs_capabilities.xml文件，确认可用图层名称")
            print("4. 尝试扩大查询范围或调整缩放级别")
            return

        # 3. 过滤和清洗数据
        filtered_gdf = filter_water_features(water_gdf)

        if filtered_gdf.empty:
            print("错误: 过滤后无有效水系数据，可能需要调整过滤条件")
            # 保存原始数据用于分析
            water_gdf.to_file(os.path.join(OUTPUT_DIR, "raw_features.geojson"), driver="GeoJSON")
            print(f"原始数据已保存至: {os.path.join(OUTPUT_DIR, 'raw_features.geojson')}")
            return

        # 4. 保存结果
        filtered_gdf.to_file(OUTPUT_FILE, driver="GeoJSON")
        print(f"\n数据已成功保存至: {OUTPUT_FILE}")

        # 5. 保存统计信息
        with open(os.path.join(OUTPUT_DIR, "data_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"数据摘要 - {datetime.now()}\n")
            f.write(f"图层: {layer}\n")
            f.write(f"要素总数: {len(filtered_gdf)}\n")
            f.write(f"字段: {', '.join(filtered_gdf.columns.tolist())}\n")
            f.write(f"几何类型: {', '.join(filtered_gdf.geom_type.unique())}\n")

        # 6. 可视化结果
        print("\n正在生成可视化图像...")
        visualize_water_data(filtered_gdf, os.path.join(OUTPUT_DIR, "water_visualization.png"))

        print("\n===== 处理完成 =====")
        print(f"数据已保存至: {OUTPUT_FILE}")
        print(f"可视化结果已保存至: {os.path.join(OUTPUT_DIR, 'water_visualization.png')}")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        # 记录错误日志
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a") as f:
            f.write(f"[{datetime.now()}] {str(e)}\n")


if __name__ == "__main__":
    main()