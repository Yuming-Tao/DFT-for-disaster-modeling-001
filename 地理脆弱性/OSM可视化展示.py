import geopandas as gpd
import matplotlib.pyplot as plt
import os
import contextily as ctx
from matplotlib.colors import ListedColormap

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def load_osm_layers(data_dir):
    """加载OSM数据的所有图层"""
    # 查找所有.shp文件
    shp_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.shp'):
                shp_files.append(os.path.join(root, file))

    # 加载所有图层
    layers = {}
    for shp_file in shp_files:
        try:
            layer_name = os.path.basename(shp_file).replace('.shp', '')
            gdf = gpd.read_file(shp_file)
            layers[layer_name] = gdf
            print(f"已加载图层: {layer_name} ({len(gdf)} 要素)")
        except Exception as e:
            print(f"无法加载文件 {shp_file}: {e}")

    return layers


def visualize_osm_layers(layers, region_boundary=None, output_path=None):
    """可视化OSM图层"""
    if not layers:
        print("没有图层可可视化")
        return

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    # 绘制区域边界（如果提供）
    if region_boundary is not None:
        region_boundary.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.7)
        ax.set_xlim(region_boundary.total_bounds[0] - 0.1, region_boundary.total_bounds[2] + 0.1)
        ax.set_ylim(region_boundary.total_bounds[1] - 0.1, region_boundary.total_bounds[3] + 0.1)

    # 定义不同图层的颜色
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_index = 0

    # 绘制所有图层
    for layer_name, gdf in layers.items():
        # 跳过空图层
        if gdf.empty:
            continue

        # 设置颜色和样式
        color = colors[color_index % len(colors)]
        color_index += 1

        # 根据几何类型设置不同的绘制参数
        if gdf.geom_type.iloc[0] == 'LineString' or gdf.geom_type.iloc[0] == 'MultiLineString':
            gdf.plot(ax=ax, color=color, linewidth=1, label=layer_name)
        elif gdf.geom_type.iloc[0] == 'Polygon' or gdf.geom_type.iloc[0] == 'MultiPolygon':
            gdf.plot(ax=ax, color=color, alpha=0.3, edgecolor='black', linewidth=0.5, label=layer_name)
        else:  # 点要素
            gdf.plot(ax=ax, color=color, markersize=5, label=layer_name)

    # 添加图例
    ax.legend(loc='upper right', fontsize='small', title='OSM图层')

    # 添加底图
    try:
        ctx.add_basemap(ax, crs=list(layers.values())[0].crs, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"无法添加底图: {e}")

    # 设置标题和样式
    plt.title('OSM数据可视化', fontsize=18)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.tight_layout()

    # 保存或显示图形
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图形已保存至: {output_path}")
    plt.show()


def main():
    # 文件路径
    data_dir = r"C:\Users\Administrator\Desktop\参数文件\河南省OSM数据"  # 指向已解压的OSM数据目录
    boundary_path = r"C:\Users\Administrator\Desktop\参数文件\郑州市_市.geojson"  # 区域边界文件

    # 加载区域边界
    region_boundary = None
    if os.path.exists(boundary_path):
        try:
            region_boundary = gpd.read_file(boundary_path)
            print("已加载区域边界")
        except Exception as e:
            print(f"无法加载区域边界: {e}")

    # 加载OSM图层
    print("正在加载OSM图层...")
    layers = load_osm_layers(data_dir)

    # 可视化
    print("正在生成可视化结果...")
    visualize_osm_layers(layers, region_boundary, output_path=r"D:\osm_visualization.png")


if __name__ == "__main__":
    main()