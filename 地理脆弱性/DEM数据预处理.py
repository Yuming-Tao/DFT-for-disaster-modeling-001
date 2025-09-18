import os
import sys
import subprocess
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from packaging import version
from shapely.geometry import shape

# 配置参数
DEM_DIR = r"C:\Users\Administrator\Desktop\参数文件\郑州市30m分辨率的GDTM高程数据"
ADMIN_GEOJSON = r"C:\Users\Administrator\Desktop\参数文件\郑州市_市.geojson"
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\输出结果"

# 设置GDAL_DATA环境变量
try:
    gdal_data = subprocess.check_output(["gdal-config", "--datadir"], universal_newlines=True).strip()
    if gdal_data:
        os.environ['GDAL_DATA'] = gdal_data
except (subprocess.CalledProcessError, FileNotFoundError):
    # 如果自动检测失败，尝试常见的conda路径
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        gdal_data_path = os.path.join(conda_env, 'share', 'gdal')
        if os.path.exists(gdal_data_path):
            os.environ['GDAL_DATA'] = gdal_data_path

# 检查rasterio版本
required_version = "1.3.0"
if version.parse(rasterio.__version__) < version.parse(required_version):
    raise ImportError(f"需要rasterio版本{required_version}或更高，当前版本为{rasterio.__version__}")

print(f"当前rasterio版本: {rasterio.__version__}")
print(f"GDAL_DATA环境变量: {os.environ.get('GDAL_DATA', '未设置')}")


# 备选掩码创建方法
def create_mask_with_shapely(geometries, shape, transform):
    """使用shapely创建掩码，替代rasterio.features.geometry_mask"""
    from shapely.ops import unary_union

    # 合并所有几何图形
    geom = unary_union([shape(g) for g in geometries])

    # 创建网格坐标
    rows, cols = shape
    y, x = np.mgrid[:rows, :cols]
    x_coords = transform[2] + x * transform[0] + y * transform[1]
    y_coords = transform[5] + x * transform[3] + y * transform[4]

    # 创建掩码（注意：这种方法较慢，适用于中小规模数据）
    mask = np.zeros(shape, dtype=bool)

    # 为了提高效率，只检查边界框内的点
    minx, miny, maxx, maxy = geom.bounds

    # 找到在边界框内的点
    in_bbox = (x_coords >= minx) & (x_coords <= maxx) & (y_coords >= miny) & (y_coords <= maxy)

    # 对边界框内的点进行更精确的检查
    for i in range(rows):
        for j in range(cols):
            if in_bbox[i, j]:
                # 创建一个小矩形表示栅格单元
                cell = shape({
                    'type': 'Polygon',
                    'coordinates': [[
                        (x_coords[i, j], y_coords[i, j]),
                        (x_coords[i, j + 1], y_coords[i, j + 1]),
                        (x_coords[i + 1, j + 1], y_coords[i + 1, j + 1]),
                        (x_coords[i + 1, j], y_coords[i + 1, j]),
                        (x_coords[i, j], y_coords[i, j])
                    ]]
                })
                mask[i, j] = geom.intersects(cell)

    return mask


def load_tiff_dataset():
    """加载分幅TIFF数据集"""
    # 查找所有TIFF文件
    tiff_files = [
        os.path.join(DEM_DIR, f)
        for f in os.listdir(DEM_DIR)
        if f.lower().endswith(('.tif', '.tiff'))
    ]

    if not tiff_files:
        raise FileNotFoundError(f"指定目录 {DEM_DIR} 未找到TIFF文件")

    print(f"找到 {len(tiff_files)} 个TIFF文件")

    # 打开所有文件并验证基础参数
    src_list = []
    for fp in tiff_files:
        try:
            src = rasterio.open(fp)
            src_list.append(src)
        except rasterio.RasterioIOError as e:
            print(f"警告: 文件 {os.path.basename(fp)} 无法读取 - {str(e)}")
            continue

    if not src_list:
        raise FileNotFoundError("没有可读取的TIFF文件")

    # 获取基准参数（从第一个文件）
    base_crs = src_list[0].crs
    base_res = src_list[0].res  # 直接获取分辨率元组
    base_count = src_list[0].count

    print(f"基准参数 - CRS: {base_crs}, 分辨率: {base_res}, 波段数: {base_count}")

    # 参数校验
    for idx, src in enumerate(src_list):
        error_prefix = f"文件 {os.path.basename(src.name)} "

        # 坐标系校验
        if src.crs != base_crs:
            raise ValueError(error_prefix + f"坐标系不一致 ({src.crs} vs {base_crs})")

        # 分辨率校验（允许0.1米容差）
        if not np.allclose(src.res, base_res, atol=0.1):
            raise ValueError(error_prefix +
                             f"分辨率不一致 ({src.res} vs {base_res})")

        # 波段数校验
        if src.count != base_count:
            raise ValueError(error_prefix +
                             f"波段数不一致 ({src.count} vs {base_count})")

    return src_list


def merge_and_clip():
    """执行数据拼接与裁剪"""
    # 加载数据
    src_list = load_tiff_dataset()

    try:
        # 执行拼接
        print("开始拼接DEM数据...")
        mosaic, transform = merge(src_list)
        mosaic = mosaic[0]  # 获取第一个波段
        print(f"拼接完成 - 数据形状: {mosaic.shape}")

        # 创建元数据模板
        meta = src_list[0].meta.copy()
        meta.update({
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "transform": transform,
            "driver": "GTiff",  # 强制输出为GeoTIFF
            "tiled": True,  # 启用分块存储
            "compress": 'lzw'  # 使用LZW压缩
        })

        # 行政边界裁剪
        print("开始裁剪DEM数据...")
        admin_gdf = gpd.read_file(ADMIN_GEOJSON).to_crs(meta['crs'])

        # 确保边界数据有效
        if admin_gdf.empty:
            raise ValueError("行政边界数据为空或无法读取")

        # 尝试使用rasterio.features创建掩码
        try:
            print("尝试使用rasterio.features创建掩码...")
            from rasterio import features
            mask = features.geometry_mask(
                admin_gdf.geometry,
                mosaic.shape,
                transform,
                invert=True
            )
            print("成功使用rasterio.features创建掩码")
        except (AttributeError, ImportError) as e:
            print(f"警告: rasterio.features不可用 ({str(e)})，使用备选方法创建掩码")
            # 使用备选方法
            mask = create_mask_with_shapely(admin_gdf.geometry, mosaic.shape, transform)
            print("成功使用备选方法创建掩码")

        clipped = np.where(mask, mosaic, meta['nodata'])
        print("裁剪完成")

        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "郑州DEM_拼接裁剪.tiff")

        print(f"保存结果到: {output_path}")
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(clipped, 1)

        return output_path

    finally:
        # 确保资源被释放
        for src in src_list:
            src.close()


def visualize_result(tiff_path):
    """可视化验证结果"""
    print(f"可视化结果: {tiff_path}")

    with rasterio.open(tiff_path) as src:
        # 设置支持中文的字体
        import matplotlib.font_manager as fm

        # 尝试查找系统中可用的中文字体
        chinese_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        font_found = False
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams["font.family"] = font
                font_found = True
                print(f"已设置中文字体: {font}")
                break

        if not font_found:
            print("警告: 未找到支持中文的字体，中文可能无法正确显示")
            print("可用字体列表:", available_fonts[:10])  # 显示前10个可用字体

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))

        # DEM渲染
        show(src, ax=ax, cmap='terrain', title='郑州市DEM高程图')

        # 叠加行政边界
        admin_gdf = gpd.read_file(ADMIN_GEOJSON).to_crs(src.crs)

        # 确保边界数据有效
        if not admin_gdf.empty:
            admin_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=1)
        else:
            print("警告: 行政边界数据为空，无法叠加显示")

        # 添加比例尺
        from matplotlib_scalebar.scalebar import ScaleBar

        # 设置坐标轴比例一致
        ax.set_aspect('equal')

        # 添加比例尺，禁用旋转警告
        ax.add_artist(ScaleBar(1, location='lower right'))

        # 添加色标
        plt.colorbar(ax.images[0], label="高程 (米)")

        # 保存图像
        output_img = os.path.join(OUTPUT_DIR, "dem_visualization.png")
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"可视化图像已保存到: {output_img}")

        # 显示图像
        plt.show()


if __name__ == "__main__":
    try:
        print("===== 开始处理郑州市DEM数据 =====")

        # 执行处理流程
        result_path = merge_and_clip()
        print(f"处理完成，结果保存至: {result_path}")

        # 可视化验证
        visualize_result(result_path)

        # 后续处理建议
        print("\n后续步骤建议：")
        print("1. 使用生成的DEM计算坡度/坡向")
        print("2. 集成河道数据进行脆弱性分析")
        print("3. 利用rasterio计算地形湿度指数")

    except Exception as e:
        import traceback

        print(f"处理中断，错误信息: {str(e)}")
        print("详细堆栈信息:")
        traceback.print_exc()