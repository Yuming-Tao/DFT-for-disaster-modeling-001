import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RiverPotentialField:
    """河道要素势场计算模型"""

    # 定义默认参数
    DEFAULT_PARAMS = {
        'distance_decay_max': 1000.0,  # 距离衰减最大距离
        'distance_decay_type': 'exponential',  # 距离衰减类型
        'hydraulic_alpha': 1.2,  # 水力特性影响系数
        'smoothing_sigma': 3.0,  # 平滑处理sigma值
        'roughness_lookup': {
            'river': 0.035,  # 主要河流
            'stream': 0.045,  # 支流
            'canal': 0.025,  # 运河
            'drain': 0.05  # 排水沟
        },
        'default_river_width': 10.0  # 默认河道宽度
    }

    def __init__(self, dem_path: str, output_dir: str, params: dict = None,
                 river_geojson_path: str = None, distance_tif_path: str = None):
        """初始化模型参数"""
        self.dem_path = dem_path
        self.river_geojson_path = river_geojson_path
        self.distance_tif_path = distance_tif_path
        self.output_dir = output_dir
        self.ref_crs = None
        self.ref_transform = None
        self.ref_shape = None
        self.resolution = None
        self.params = self._validate_params(params)

        os.makedirs(output_dir, exist_ok=True)

    def _validate_params(self, params: dict) -> dict:
        """验证并合并参数"""
        if params is None:
            return self.DEFAULT_PARAMS

        # 合并用户参数和默认参数
        merged_params = self.DEFAULT_PARAMS.copy()
        merged_params.update(params)

        # 参数验证
        if merged_params['distance_decay_type'] not in ['linear', 'exponential', 'gaussian']:
            logger.warning(f"不支持的衰减类型: {merged_params['distance_decay_type']}，使用默认值'exponential'")
            merged_params['distance_decay_type'] = 'exponential'

        if merged_params['smoothing_sigma'] < 0:
            logger.warning(f"平滑sigma值不能为负，使用默认值3.0")
            merged_params['smoothing_sigma'] = 3.0

        return merged_params

    def load_raster_data(self, raster_path: str, target_crs=None, target_transform=None,
                         target_shape=None) -> np.ndarray:
        """加载栅格数据并自动转换到参考坐标系"""
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"栅格文件不存在: {raster_path}")

        try:
            with rasterio.open(raster_path) as src:
                src_data = src.read(1).astype(np.float32)
                src_crs = src.crs
                src_transform = src.transform
                src_nodata = src.nodatavals[0] if src.nodatavals else np.nan

                if target_crs is None or target_transform is None or target_shape is None:
                    self.ref_crs = src.crs
                    self.ref_transform = src_transform
                    self.ref_shape = (src.height, src.width)
                    self.resolution = abs(src_transform[0])
                    target_crs = src.crs
                    target_transform = src_transform
                    target_shape = self.ref_shape
                else:
                    if src_crs != target_crs:
                        transform, width, height = calculate_default_transform(
                            src_crs, target_crs, src.width, src.height, *src.bounds
                        )
                        target_transform = transform
                        target_shape = (height, width)

                dst_data = np.empty(target_shape, dtype=np.float32)

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_width=target_shape[1],
                    dst_height=target_shape[0],
                    resampling=Resampling.bilinear
                )

                dst_data = np.where(dst_data == src_nodata, np.nan, dst_data)
                return dst_data
        except Exception as e:
            logger.error(f"加载栅格数据时出错: {str(e)}")
            raise

    def load_river_data(self) -> None:
        """加载河道数据（支持GeoJSON和距离栅格两种方式）"""
        if self.river_geojson_path:
            self._load_river_from_geojson()
        elif self.distance_tif_path:
            self._load_river_from_distance_tif()
        else:
            raise ValueError("必须提供河道GeoJSON路径或距离栅格TIF路径")

    def _load_river_from_geojson(self) -> None:
        """从GeoJSON加载河道数据"""
        if not os.path.exists(self.river_geojson_path):
            raise FileNotFoundError(f"河道GeoJSON文件不存在: {self.river_geojson_path}")

        try:
            self.river_data = gpd.read_file(self.river_geojson_path)

            # 检查数据是否包含geometry列
            if 'geometry' not in self.river_data.columns:
                raise ValueError("河道数据不包含geometry列")

            # 强制转换为DEM坐标系
            if self.ref_crs and self.river_data.crs != self.ref_crs:
                logger.info(f"将河道数据从 {self.river_data.crs} 转换为 {self.ref_crs}")
                self.river_data = self.river_data.to_crs(self.ref_crs)

            # 提取河道坐标点（支持LineString和MultiLineString）
            self.river_coords = []
            for geom in self.river_data.geometry:
                if geom.geom_type == 'LineString':
                    self.river_coords.extend(geom.coords)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        self.river_coords.extend(line.coords)
            self.river_coords = np.array(self.river_coords, dtype=np.float32)

            logger.info(f"成功加载河道数据，包含 {len(self.river_coords)} 个坐标点")
        except Exception as e:
            logger.error(f"加载河道GeoJSON数据时出错: {str(e)}")
            raise

    def _load_river_from_distance_tif(self) -> None:
        """从距离栅格TIF加载河道数据"""
        if not os.path.exists(self.distance_tif_path):
            raise FileNotFoundError(f"距离栅格TIF文件不存在: {self.distance_tif_path}")

        try:
            # 加载距离栅格
            self.river_distance = self.load_raster_data(self.distance_tif_path)
            self.river_mask = self.river_distance <= self.resolution  # 标记河道所在栅格

            # 从距离栅格反推河道位置（提取距离为0的点）
            river_indices = np.where(self.river_distance <= 1e-8)  # 考虑精度误差
            rows, cols = river_indices

            # 转换为坐标点
            x_coords = np.arange(self.ref_shape[1]) * self.resolution + self.ref_transform[2]
            y_coords = np.arange(self.ref_shape[0]) * self.resolution + self.ref_transform[5]

            self.river_coords = np.column_stack((
                x_coords[cols],
                y_coords[rows]
            )).astype(np.float32)

            logger.info(f"从距离栅格提取河道位置，识别出 {len(self.river_coords)} 个河道点")
        except Exception as e:
            logger.error(f"从距离栅格加载河道数据时出错: {str(e)}")
            raise

    def calculate_distance_to_rivers(self) -> np.ndarray:
        """计算到河道的距离栅格（如果未从TIF加载）"""
        if hasattr(self, 'river_distance'):
            logger.info("已从TIF加载距离栅格，跳过计算")
            return self.river_distance

        if not hasattr(self, 'ref_shape') or not hasattr(self, 'resolution'):
            raise ValueError("请先加载参考栅格数据")

        if not hasattr(self, 'river_coords') or len(self.river_coords) == 0:
            raise ValueError("请先加载河道数据")

        try:
            rows, cols = self.ref_shape

            # 生成栅格中心点坐标
            x_coords = np.arange(cols) * self.resolution + self.ref_transform[2]
            y_coords = np.arange(rows) * self.resolution + self.ref_transform[5]
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)

            # 使用cKDTree加速距离计算
            grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel())).astype(np.float32)
            river_tree = cKDTree(self.river_coords)
            distances, _ = river_tree.query(grid_points, k=1, distance_upper_bound=1e9)

            self.river_distance = distances.reshape(rows, cols).astype(np.float32)
            self.river_mask = self.river_distance <= self.resolution  # 标记河道所在栅格

            logger.info(f"成功计算距离栅格，尺寸: {rows}x{cols}")
            return self.river_distance
        except Exception as e:
            logger.error(f"计算到河道的距离时出错: {str(e)}")
            raise

    def calculate_river_risk(self) -> np.ndarray:
        """计算河道影响风险因子（距离衰减模型）"""
        if not hasattr(self, 'river_distance'):
            self.calculate_distance_to_rivers()

        try:
            dist = self.river_distance.copy()
            dist[np.isnan(dist)] = np.max(dist[~np.isnan(dist)])

            d_max = self.params['distance_decay_max']
            decay_type = self.params['distance_decay_type']

            # 应用距离衰减函数
            if decay_type == 'linear':
                phi_r = np.where(dist <= d_max, 1 - dist / d_max, 0)
            elif decay_type == 'exponential':
                k = 3.0 / d_max  # 使距离d_max处衰减至约0.05
                phi_r = np.exp(-k * dist)
            elif decay_type == 'gaussian':
                sigma = d_max / 3.0
                phi_r = np.exp(-0.5 * ((dist / sigma) ** 2))
            else:
                raise ValueError(f"不支持的衰减类型: {decay_type}")

            phi_r[self.river_mask] = 1  # 河道内风险为1

            logger.info(f"成功计算河道风险因子，衰减类型: {decay_type}")
            return phi_r
        except Exception as e:
            logger.error(f"计算河道风险因子时出错: {str(e)}")
            raise

    def calculate_hydraulic_factor(self) -> np.ndarray:
        """计算水力特性风险因子（结合河道宽度和糙率）"""
        if not hasattr(self, 'river_data'):
            if self.river_geojson_path:
                self._load_river_from_geojson()
            else:
                logger.warning("没有GeoJSON格式的河道数据，无法计算水力特性因子，返回默认值")
                return np.zeros(self.ref_shape, dtype=np.float32)

        try:
            # 创建水力因子栅格
            hydraulic_factor = np.zeros(self.ref_shape, dtype=np.float32)

            # 获取参数
            roughness_lookup = self.params['roughness_lookup']
            default_width = self.params['default_river_width']

            # 生成栅格坐标
            rows, cols = self.ref_shape
            x_coords = np.arange(cols) * self.resolution + self.ref_transform[2]
            y_coords = np.arange(rows) * self.resolution + self.ref_transform[5]
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel())).astype(np.float32)

            # 为每个河道创建KDTree并查询影响范围内的栅格
            for idx, river in self.river_data.iterrows():
                river_type = river.get('waterway', 'river')
                roughness = roughness_lookup.get(river_type, 0.035)
                width = river.get('width', default_width)  # 默认宽度

                if river.geometry.geom_type == 'LineString':
                    coords = np.array(river.geometry.coords, dtype=np.float32)
                elif river.geometry.geom_type == 'MultiLineString':
                    coords = np.array([p for line in river.geometry.geoms for p in line.coords], dtype=np.float32)
                else:
                    continue

                # 影响范围（宽度的倍数）
                buffer_distance = width * 1.5

                # 快速查询在影响范围内的栅格点
                river_tree = cKDTree(coords)
                indices = river_tree.query_ball_point(grid_points, r=buffer_distance)

                for i, point_indices in enumerate(indices):
                    if point_indices:
                        # 计算距离并应用衰减
                        row, col = i // cols, i % cols
                        min_dist = np.min([np.linalg.norm(grid_points[i] - coords[j]) for j in point_indices])

                        # 影响强度随距离衰减
                        factor = roughness * (1 - min_dist / buffer_distance) * (width / 10.0)
                        if factor > hydraulic_factor[row, col]:
                            hydraulic_factor[row, col] = factor

            # 归一化
            if np.max(hydraulic_factor) > 0:
                hydraulic_factor = hydraulic_factor / np.max(hydraulic_factor)

            logger.info(f"成功计算水力特性风险因子，处理了 {len(self.river_data)} 条河道")
            return hydraulic_factor
        except Exception as e:
            logger.error(f"计算水力特性风险因子时出错: {str(e)}")
            raise

    def build_river_potential_field(self) -> np.ndarray:
        """构建河道要素势场"""
        try:
            # 计算河道风险因子
            logger.info("开始计算河道风险因子...")
            phi_r = self.calculate_river_risk()

            # 计算水力特性因子
            logger.info("开始计算水力特性风险因子...")
            hydraulic_factor = self.calculate_hydraulic_factor()

            # 获取参数
            alpha = self.params['hydraulic_alpha']
            sigma = self.params['smoothing_sigma']

            # 计算河道影响势场
            logger.info("开始构建河道要素势场...")
            v_river = phi_r * (1 + hydraulic_factor * alpha)  # alpha为水力特性影响系数
            v_river[self.river_mask] = np.max(v_river)  # 河道内风险设为最大值

            # 平滑处理
            v_smoothed = gaussian_filter(v_river, sigma=sigma) if sigma > 0 else v_river

            # 归一化
            v_min, v_max = np.min(v_smoothed), np.max(v_smoothed)
            v_normalized = (v_smoothed - v_min) / (v_max - v_min) if v_max != v_min else np.zeros_like(v_smoothed)
            v_normalized[np.isnan(v_normalized)] = 0

            # 保存结果
            self._save_result(v_normalized, "river_potential_field.tif", "河道要素势场")

            logger.info("河道要素势场构建完成")
            return v_normalized
        except Exception as e:
            logger.error(f"构建河道要素势场时出错: {str(e)}")
            raise

    def visualize_results(self, river_field: np.ndarray):
        """可视化河道要素势场结果"""
        if not hasattr(self, 'river_coords') or len(self.river_coords) == 0:
            logger.warning("没有河道坐标数据，无法叠加河道网络")
            has_river_network = False
        else:
            has_river_network = True

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            extent = self._get_raster_extent()

            # 河道要素势场
            im = ax.imshow(river_field, cmap='Blues', origin='upper', extent=extent)
            ax.set_title('河道要素势场')
            plt.colorbar(im, ax=ax, label='风险值 [0-1]')

            # 叠加河道网络（如果有）
            if has_river_network and hasattr(self, 'river_data'):
                self.river_data.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.7, label='河道')
                ax.legend()

            plt.tight_layout()
            output_path = os.path.join(self.output_dir, "river_risk_visualization.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"可视化结果已保存至: {output_path}")
        except Exception as e:
            logger.error(f"可视化结果时出错: {str(e)}")
            raise

    def _get_raster_extent(self) -> tuple:
        """获取栅格范围（左, 右, 下, 上）"""
        if not hasattr(self, 'ref_transform') or not hasattr(self, 'ref_shape'):
            raise ValueError("参考栅格信息未初始化")

        left = self.ref_transform[2]
        bottom = self.ref_transform[5] + self.ref_shape[0] * self.ref_transform[4]
        right = left + self.ref_shape[1] * self.ref_transform[0]
        top = self.ref_transform[5]
        return (left, right, bottom, top)

    def _save_result(self, data: np.ndarray, filename: str, desc: str):
        """保存结果栅格"""
        if not hasattr(self, 'ref_crs') or not hasattr(self, 'ref_transform') or not hasattr(self, 'ref_shape'):
            raise ValueError("参考栅格信息未初始化")

        try:
            path = os.path.join(self.output_dir, filename)
            dtype = 'float32'
            with rasterio.open(
                    path,
                    'w',
                    driver='GTiff',
                    height=self.ref_shape[0],
                    width=self.ref_shape[1],
                    count=1,
                    dtype=dtype,
                    crs=self.ref_crs,
                    transform=self.ref_transform,
                    nodata=np.nan
            ) as dst:
                dst.write(data.astype(dtype), 1)

            logger.info(f"{desc} 已保存至: {path}")
        except Exception as e:
            logger.error(f"保存结果栅格时出错: {str(e)}")
            raise


def main():
    """主函数：配置路径并运行模型"""
    # 配置数据路径
    DEM_PATH = r"C:\Users\Administrator\Desktop\参数文件\郑州DEM_拼接裁剪.tiff"
    RIVER_GEOJSON_PATH = r"C:\Users\Administrator\Desktop\河道分析结果_20250531_170314\river_network.geojson"
    DISTANCE_TIF_PATH = r"C:\Users\Administrator\Desktop\河道分析结果_20250531_170314\distance_to_rivers.tif"
    OUTPUT_DIR = r"C:\Users\Administrator\Desktop\输出结果\河道要素势场"

    # 自定义参数
    params = {
        'distance_decay_max': 1500.0,  # 增大距离衰减范围
        'hydraulic_alpha': 1.5  # 增大水力特性影响系数
    }

    # 初始化模型
    # 示例1：同时使用GeoJSON和距离栅格
    model = RiverPotentialField(
        dem_path=DEM_PATH,
        output_dir=OUTPUT_DIR,
        params=params,
        river_geojson_path=RIVER_GEOJSON_PATH,
        distance_tif_path=DISTANCE_TIF_PATH
    )

    # 示例2：仅使用GeoJSON
    # model = RiverPotentialField(
    #     dem_path=DEM_PATH,
    #     output_dir=OUTPUT_DIR,
    #     params=params,
    #     river_geojson_path=RIVER_GEOJSON_PATH
    # )

    # 示例3：仅使用距离栅格（无法计算水力特性因子）
    # model = RiverPotentialField(
    #     dem_path=DEM_PATH,
    #     output_dir=OUTPUT_DIR,
    #     params=params,
    #     distance_tif_path=DISTANCE_TIF_PATH
    # )

    try:
        # 加载DEM数据（用于设置参考坐标系）
        logger.info(f"开始加载DEM数据: {DEM_PATH}")
        model.dem = model.load_raster_data(model.dem_path)
        model.dem = np.where(np.isnan(model.dem), np.nanmean(model.dem), model.dem)
        logger.info(f"DEM数据加载完成，尺寸: {model.dem.shape}")

        # 加载河道数据
        logger.info("开始加载河道数据...")
        model.load_river_data()
        logger.info("河道数据加载完成")

        # 计算河道要素势场
        logger.info("开始计算河道要素势场...")
        river_field = model.build_river_potential_field()
        logger.info("河道要素势场计算完成")

        # 可视化结果
        logger.info("开始生成可视化结果...")
        model.visualize_results(river_field)
        logger.info("可视化结果生成完成")

        logger.info("河道要素势场计算模型运行完成！")
    except Exception as e:
        logger.error(f"模型运行失败：{str(e)}")


if __name__ == "__main__":
    main()