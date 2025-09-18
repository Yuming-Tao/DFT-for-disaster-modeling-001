import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SoilSigmoidPotential:
    """基于Sigmoid函数的土壤渗透势场模型"""

    def __init__(self, permeability_raster_path: str, output_dir: str,
                 p_min: float = 0.1, p_max: float = 0.9, sigmoid_k: float = 5.0, sigmoid_x0: float = 0.3):
        """初始化模型参数

        Args:
            permeability_raster_path: 渗透率栅格路径（OSM处理结果）
            output_dir: 结果输出目录
            p_min: 最低风险概率（默认0.1）
            p_max: 最高风险概率（默认0.9）
            sigmoid_k: Sigmoid斜率参数（默认5.0，越大曲线越陡峭）
            sigmoid_x0: Sigmoid中心位置（默认0.3，对应概率0.5的标准化渗透率值）
        """
        self.permeability_path = permeability_raster_path
        self.output_dir = output_dir
        self.p_min = p_min
        self.p_max = p_max
        self.k = sigmoid_k
        self.x0 = sigmoid_x0
        self.ref_crs = None
        self.ref_transform = None
        self.ref_shape = None
        self.resolution = None
        os.makedirs(output_dir, exist_ok=True)

    def load_permeability_data(self) -> np.ndarray:
        """加载渗透率数据并标准化到[0,1]"""
        with rasterio.open(self.permeability_path) as src:
            data = src.read(1).astype(np.float32)
            self.ref_crs = src.crs
            self.ref_transform = src.transform
            self.ref_shape = (src.height, src.width)
            self.resolution = abs(src.transform[0])

            # 处理无效值（负值/NaN设为0）
            data = np.where(data <= 0, 0, data)
            data = np.where(np.isnan(data), 0, data)

            # 标准化到[0,1]区间
            if np.max(data) == np.min(data):
                normalized = np.zeros_like(data)
            else:
                normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            return normalized

    def sigmoid_mapping(self, normalized_ks: np.ndarray) -> np.ndarray:
        """Sigmoid函数非线性概率映射"""
        # 核心公式：P = p_min + (p_max-p_min) * (1 / (1 + exp(-k*(x0 - x))))
        linear_term = self.k * (self.x0 - normalized_ks)
        probability = self.p_min + (self.p_max - self.p_min) * (1.0 / (1.0 + np.exp(-linear_term)))
        return np.clip(probability, 0, 1)  # 确保概率在[0,1]

    def build_potential_field(self, sigma: float = 2.0) -> np.ndarray:
        """构建完整势场（标准化+非线性映射+空间平滑）"""
        # 1. 加载并标准化渗透率
        normalized_ks = self.load_permeability_data()
        logger.info(f"标准化渗透率范围: [{np.min(normalized_ks):.2f}, {np.max(normalized_ks):.2f}]")

        # 2. 应用Sigmoid概率映射
        prob_field = self.sigmoid_mapping(normalized_ks)
        logger.info(f"原始概率范围: [{np.min(prob_field):.2f}, {np.max(prob_field):.2f}]")

        # 3. 空间平滑（高斯滤波模拟邻域扩散）
        smoothed_field = gaussian_filter(prob_field, sigma=sigma)
        logger.info(f"平滑后概率范围: [{np.min(smoothed_field):.2f}, {np.max(smoothed_field):.2f}]")

        return smoothed_field

    def save_result(self, data: np.ndarray, filename: str = "soil_sigmoid_potential.tif") -> None:
        """保存结果栅格"""
        path = os.path.join(self.output_dir, filename)
        with rasterio.open(
                path, 'w',
                driver='GTiff',
                height=self.ref_shape[0],
                width=self.ref_shape[1],
                count=1,
                dtype='float32',
                crs=self.ref_crs,
                transform=self.ref_transform,
                nodata=-9999
        ) as dst:
            dst.write(data, 1)
        logger.info(f"势场已保存至: {path}")

    def visualize_results(self, data: np.ndarray, output_path: str = "sigmoid_potential_visual.png") -> None:
        """可视化结果（带概率色阶）"""
        # 确保输出路径包含目录信息
        if not os.path.dirname(output_path):
            output_path = os.path.join(self.output_dir, output_path)

        fig, ax = plt.subplots(figsize=(10, 8))
        extent = (
            self.ref_transform[2],
            self.ref_transform[2] + self.ref_shape[1] * self.resolution,
            self.ref_transform[5] - self.ref_shape[0] * self.resolution,
            self.ref_transform[5]
        )

        im = ax.imshow(
            data,
            cmap='viridis',
            origin='upper',
            extent=extent,
            vmin=0, vmax=1
        )
        ax.set_title('Soil Infiltration Risk Potential Field (Sigmoid Mapping)')
        plt.colorbar(im, ax=ax, label='Risk Probability [0-1]')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"可视化图已保存至: {output_path}")


def main():
    """主函数：执行完整流程"""
    # 配置路径（请替换为实际路径）
    PERMEABILITY_PATH = r"C:\Users\Administrator\Desktop\输出结果\osm_analysis\soil_permeability.tif"
    OUTPUT_DIR = r"C:\Users\Administrator\Desktop\输出结果\soil_sigmoid_potential"

    # 初始化模型（可调整Sigmoid参数）
    model = SoilSigmoidPotential(
        permeability_raster_path=PERMEABILITY_PATH,
        output_dir=OUTPUT_DIR,
        p_min=0.2,  # 最低概率（高渗透率区域）
        p_max=0.8,  # 最高概率（低渗透率区域）
        sigmoid_k=4.0,  # 斜率参数（降低k值使曲线更平缓）
        sigmoid_x0=0.4  # 中心位置（向右移动对应更严格的风险阈值）
    )

    try:
        # 计算势场
        potential_field = model.build_potential_field(sigma=1.5)

        # 保存结果
        model.save_result(potential_field)

        # 可视化
        model.visualize_results(potential_field)

        logger.info("Sigmoid势场构建完成！")

    except Exception as e:
        logger.error(f"运行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()