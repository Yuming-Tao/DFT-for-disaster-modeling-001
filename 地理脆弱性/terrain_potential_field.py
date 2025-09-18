import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform
from scipy.ndimage import gaussian_filter, generic_filter
import matplotlib.pyplot as plt


class ComprehensivePotentialField:
    """综合地理要素势场计算模型"""

    def __init__(self, dem_path: str, output_dir: str):
        self.dem_path = dem_path
        self.output_dir = output_dir
        self.transform = None
        self.crs = None
        self.ref_shape = None
        os.makedirs(output_dir, exist_ok=True)

    def load_and_preprocess_dem(self):
        """加载并预处理DEM（洼地填充+坐标信息提取）"""
        with rasterio.open(self.dem_path) as src:
            dem = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.ref_shape = (src.height, src.width)
            dem = np.where(dem == src.nodatavals[0], np.nan, dem)

            # 洼地填充（3x3均值滤波）
            filled_dem = generic_filter(dem, np.nanmean, size=3)
            filled_dem = np.nan_to_num(filled_dem, nan=np.nanmean(filled_dem))
        return filled_dem

    def calculate_terrain_factors(self, dem):
        """计算地形风险因子（海拔和坡度）"""
        z_min, z_max = np.nanmin(dem), np.nanmax(dem)
        phi_z = (z_max - dem) / (z_max - z_min + 1e-8)  # 海拔风险因子
        phi_z[np.isnan(phi_z)] = 0

        xres = abs(self.transform[0])
        yres = abs(self.transform[4])
        dx = np.gradient(dem, xres, axis=1)
        dy = np.gradient(dem, yres, axis=0)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope_deg = np.degrees(slope_rad)
        phi_s = (np.nanmax(slope_deg) - slope_deg) / (np.nanmax(slope_deg) - np.nanmin(slope_deg) + 1e-8)  # 坡度抑制因子
        phi_s[np.isnan(phi_s)] = 0

        return phi_z, phi_s

    def calculate_flow_accumulation(self, dem):
        """计算汇水面积并转换为势场因子"""
        rows, cols = dem.shape
        flow_dir = self._calculate_flow_direction(dem)
        accumulation = self._calculate_flow_accumulation(dem, flow_dir)
        phi_a = np.log10(accumulation + 1)  # 汇水面积对数转换
        phi_a[np.isnan(phi_a)] = 0
        return phi_a

    def _calculate_flow_direction(self, dem):
        """D8水流方向计算"""
        rows, cols = dem.shape
        flow_dir = np.full((rows, cols), -1, dtype=np.int8)
        dir_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                current = dem[i, j]
                if np.isnan(current):
                    continue

                max_drop = 0
                best_dir = -1
                for d_idx, (di, dj) in enumerate(dir_offsets):
                    ni, nj = i + di, j + dj
                    neighbor = dem[ni, nj]
                    if np.isnan(neighbor):
                        continue
                    drop = current - neighbor
                    if drop > max_drop:
                        max_drop = drop
                        best_dir = d_idx
                if best_dir != -1:
                    flow_dir[i, j] = best_dir
        return flow_dir

    def _calculate_flow_accumulation(self, dem, flow_dir):
        """汇水面积累积计算"""
        rows, cols = dem.shape
        accumulation = np.ones_like(dem, dtype=np.float32)
        valid_indices = np.where(~np.isnan(dem.flatten()))[0]
        elevations = dem.flatten()[valid_indices]
        sorted_indices = valid_indices[np.argsort(-elevations)]  # 从高到低排序
        dir_offsets = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int8)

        for idx in sorted_indices:
            i = idx // cols
            j = idx % cols
            dir_idx = flow_dir[i, j]
            if dir_idx == -1:
                continue

            di, dj = dir_offsets[dir_idx]
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                accumulation[ni, nj] += accumulation[i, j]
        return accumulation

    def build_comprehensive_potential(self, phi_z, phi_s, phi_a, sigma=1.0):
        """构建综合势场（地形+汇水面积耦合）"""
        # 物理机制耦合公式：V = phi_Z * phi_A * (1 - phi_S) （参考文档公式）
        combined = phi_z * phi_a * (1 - phi_s)
        combined[np.isnan(combined)] = 0

        # 高斯平滑
        smoothed = gaussian_filter(combined, sigma=sigma)

        # 归一化到[0,1]
        v_min, v_max = np.nanmin(smoothed), np.nanmax(smoothed)
        normalized = (smoothed - v_min) / (v_max - v_min + 1e-8)
        normalized[np.isnan(normalized)] = 0
        return normalized

    def save_result(self, data, filename):
        """保存综合势场为GeoTIFF"""
        output_path = os.path.join(self.output_dir, filename)
        with rasterio.open(
                output_path, 'w', driver='GTiff',
                height=self.ref_shape[0], width=self.ref_shape[1],
                count=1, dtype=np.float32, crs=self.crs,
                transform=self.transform, nodata=0.0
        ) as dst:
            dst.write(data, 1)

    def visualize_result(self, data, title="综合地理要素势场"):
        """可视化综合势场"""
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = (
            self.transform[2],
            self.transform[2] + self.transform[0] * self.ref_shape[1],
            self.transform[5] + self.transform[4] * self.ref_shape[0],
            self.transform[5]
        )
        im = ax.imshow(data, cmap='viridis', origin='upper', extent=extent)
        plt.colorbar(im, label='综合风险值 [0-1]')
        ax.set_title(title)
        plt.savefig(os.path.join(self.output_dir, "comprehensive_potential.png"), dpi=300)
        plt.close()


def main():
    # 配置路径
    DEM_PATH = r"C:\Users\Administrator\Desktop\参数文件\郑州DEM_拼接裁剪.tiff"
    OUTPUT_DIR = r"C:\Users\Administrator\Desktop\输出结果\地形要素势场"  # 用户指定输出目录

    # 初始化模型
    model = ComprehensivePotentialField(DEM_PATH, OUTPUT_DIR)

    # 1. 加载并预处理DEM
    filled_dem = model.load_and_preprocess_dem()

    # 2. 计算地形风险因子
    phi_z, phi_s = model.calculate_terrain_factors(filled_dem)

    # 3. 计算汇水面积势场因子
    phi_a = model.calculate_flow_accumulation(filled_dem)

    # 4. 构建综合势场（扩散系数sigma=1.0）
    comprehensive_potential = model.build_comprehensive_potential(phi_z, phi_s, phi_a, sigma=1.0)

    # 5. 保存结果
    model.save_result(comprehensive_potential, "comprehensive_potential.tif")

    # 6. 可视化
    model.visualize_result(comprehensive_potential)

    print(f"综合势场已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()