import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from datetime import datetime
import os

# 基础字体设置
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# 创建静态输出文件夹
os.makedirs("static_output", exist_ok=True)


class SignalDataVisualizer:
    def __init__(self):
        # 数据文件路径（根据实际路径调整）
        self.heat_data_path = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\结果数据\人口热力.csv"
        self.travel_data_path = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\结果数据\人口出行.csv"
        self.grid_shp_path = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"

        # 加载并预处理数据
        self.heat_df = self.load_heat_data()
        self.travel_df = self.load_travel_data()
        self.grid_gdf = self.load_grid_data()

        # 修正：计算实际有效的唯一OD对数量
        self.valid_od_pairs = self.calculate_valid_od_pairs()

    # -------------------------- 数据加载方法（强化清洗） --------------------------
    def load_heat_data(self):
        """加载并清洗人口热力数据"""
        try:
            df = pd.read_csv(self.heat_data_path)
            print(f"成功加载人口热力数据，共{len(df)}行")
        except FileNotFoundError:
            print(f"错误：未找到人口热力数据文件：{self.heat_data_path}")
            return pd.DataFrame()

        # 强制转换all_pop为数值类型
        if 'all_pop' in df.columns:
            df['all_pop'] = pd.to_numeric(df['all_pop'], errors='coerce')
            invalid_count = df['all_pop'].isna().sum()
            if invalid_count > 0:
                print(f"过滤{invalid_count}条'all_pop'列非数值数据")
                df = df.dropna(subset=['all_pop'])
        else:
            print("警告：人口热力数据中未找到'all_pop'列！")
            return pd.DataFrame()

        # 统一tid为字符串类型（去除空格）
        if 'tid' in df.columns:
            df['tid'] = df['tid'].astype(str).str.strip()  # 关键：去除空格避免格式差异
            print("人口热力数据'tid'列已转为字符串并去除空格")
        else:
            print("警告：人口热力数据中未找到'tid'列！")
            return pd.DataFrame()

        # 时段格式处理
        if 'time' in df.columns:
            df['time'] = df['time'].str.replace('24:00', '00:00')
            df[['start_str', 'end_str']] = df['time'].str.split('-', expand=True)
            invalid_split = df[['start_str', 'end_str']].isna().any(axis=1).sum()
            if invalid_split > 0:
                print(f"过滤{invalid_split}条无法拆分的时段数据")
                df = df[~df[['start_str', 'end_str']].isna().any(axis=1)]

            # 解析时间并过滤无效时段
            df['start_dt'] = pd.to_datetime(df['start_str'], format='%H:%M', errors='coerce')
            df['end_dt'] = pd.to_datetime(df['end_str'], format='%H:%M', errors='coerce')
            invalid_range = (df['start_dt'] >= df['end_dt']) | df[['start_dt', 'end_dt']].isna().any(axis=1)
            if invalid_range.sum() > 0:
                print(f"过滤{invalid_range.sum()}条无效时段（起始≥结束或格式错误）")
                df = df[~invalid_range]

            df['time'] = df['start_dt'].dt.time
            df['time_str'] = df['start_dt'].dt.strftime('%H:%M')
        else:
            print("警告：人口热力数据中未找到'time'列！")
            return pd.DataFrame()

        # 日期格式转换
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['date', 'time'])
        print(f"清洗后保留有效人口热力数据行数：{len(df)}")
        return df

    def load_travel_data(self):
        """加载并清洗人口出行数据（强化OD字段清洗）"""
        try:
            df = pd.read_csv(self.travel_data_path)
            print(f"成功加载人口出行数据，共{len(df)}行")
        except FileNotFoundError:
            print(f"错误：未找到人口出行数据文件：{self.travel_data_path}")
            return pd.DataFrame()

        # 强制转换all_pop为数值类型
        if 'all_pop' in df.columns:
            df['all_pop'] = pd.to_numeric(df['all_pop'], errors='coerce')
            invalid_count = df['all_pop'].isna().sum()
            if invalid_count > 0:
                print(f"过滤{invalid_count}条'all_pop'列非数值数据")
                df = df.dropna(subset=['all_pop'])
        else:
            print("警告：人口出行数据中未找到'all_pop'列！")
            return pd.DataFrame()

        # 清洗OD字段：转为字符串+去除空格+过滤空值
        for col in ['o_tid', 'd_tid']:
            if col in df.columns:
                # 关键：去除空格+过滤空字符串
                df[col] = df[col].astype(str).str.strip()
                df = df[df[col] != '']  # 过滤空字符串
                print(f"出行数据{col}列已转为字符串并清洗，剩余有效行数：{len(df)}")
            else:
                print(f"警告：出行数据中未找到{col}列！")
                return pd.DataFrame()

        # 日期格式转换
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['date'])
        print(f"清洗后保留有效人口出行数据行数：{len(df)}")
        return df

    # -------------------------- 修正：计算有效唯一OD对 --------------------------
    def calculate_valid_od_pairs(self):
        """计算实际存在的唯一OD对（基于去重逻辑，确保≤出行数据行数）"""
        if len(self.travel_df) == 0:
            print("无有效出行数据，无法计算OD对数量")
            return 0

        # 核心修正：用drop_duplicates获取实际存在的唯一组合
        unique_od = self.travel_df[['o_tid', 'd_tid']].drop_duplicates()
        valid_count = len(unique_od)

        # 校验逻辑：唯一OD对数量必须≤出行数据行数
        if valid_count > len(self.travel_df):
            print(f"警告：计算后唯一OD对数量（{valid_count}）仍大于出行数据行数（{len(self.travel_df)}），可能存在数据异常")
        else:
            print(f"===== 有效唯一OD对数量（实际存在的起点-终点组合）：{valid_count} 个 =====")
        return valid_count

    def load_grid_data(self):
        """加载网格数据（强化格式统一）"""
        try:
            gdf = gpd.read_file(self.grid_shp_path)
            print(f"成功加载网格数据，共{len(gdf)}个网格")
        except FileNotFoundError:
            print(f"错误：未找到网格数据文件：{self.grid_shp_path}")
            return None

        # 修复列名并统一为字符串类型（去除空格）
        if 'TID' in gdf.columns:
            gdf = gdf.rename(columns={'TID': 'tid'})
            gdf['tid'] = gdf['tid'].astype(str).str.strip()  # 关键：去除空格
            print("网格数据列名'TID'已重命名为'tid'（转为字符串并去除空格）")
        elif 'tid' in gdf.columns:
            gdf['tid'] = gdf['tid'].astype(str).str.strip()  # 去除空格
            print("网格数据'tid'列已转为字符串并去除空格")
        else:
            print("警告：网格数据中未找到'TID'或'tid'列，可能导致关联失败")
            print(f"网格数据现有列名：{gdf.columns.tolist()}")

        # 检查投影
        if gdf.crs.is_geographic:
            print(f"警告：网格数据为地理坐标系（{gdf.crs}），建议后续转换为投影坐标系")
        return gdf

    # -------------------------- 辅助方法 --------------------------
    def _project_centroid(self, gdf):
        """投影转换计算质心"""
        if gdf.crs is None:
            print("警告：网格数据无坐标系信息，默认假设为WGS84（EPSG:4326）")
            gdf = gdf.set_crs(epsg=4326)

        if gdf.crs.is_geographic:
            # 转换为郑州所在UTM投影（37N带，EPSG:32637）
            projected = gdf.to_crs(epsg=32637)
            projected['centroid'] = projected.geometry.centroid
            # 转换回原地理坐标系
            gdf['centroid'] = projected['centroid'].to_crs(gdf.crs)
        else:
            gdf['centroid'] = gdf.geometry.centroid
        return gdf

    # -------------------------- 静态可视化方法 --------------------------
    def plot_grid_heatmap(self, target_date=None, target_time=None, save_static=True):
        """绘制网格人口热力图"""
        if self.grid_gdf is None or len(self.heat_df) == 0:
            print("网格数据或人口热力数据不足，无法绘制热力图")
            return

        df = self.heat_df.copy()
        if target_date:
            target_dt = pd.to_datetime(target_date)
            df = df[df['date'] == target_dt]
            if len(df) == 0:
                print(f"未找到{target_date}的人口热力数据")
                return

        if target_time:
            try:
                target_t = pd.to_datetime(target_time, format='%H:%M').time()
                df = df[df['time'] == target_t]
                if len(df) == 0:
                    print(f"未找到起始时间为{target_time}的人口热力数据")
                    return
            except Exception as e:
                print(f"时间格式错误：{str(e)}，请使用'HH:MM'格式")
                return

        # 合并网格数据
        merged_gdf = self.grid_gdf.merge(df, on='tid', how='left')
        merged_gdf['all_pop'] = merged_gdf['all_pop'].fillna(0).astype(float)
        merged_gdf = merged_gdf[merged_gdf['all_pop'] > 0]
        if len(merged_gdf) == 0:
            print("该时段无有效人口数据")
            return

        # 投影转换
        merged_gdf = self._project_centroid(merged_gdf)

        # 绘制静态热力图
        fig, ax = plt.subplots(figsize=(12, 10))
        q95 = merged_gdf['all_pop'].quantile(0.95)
        merged_gdf.plot(
            column='all_pop',
            cmap='YlOrRd',
            ax=ax,
            legend=True,
            legend_kwds={'label': 'Resident Count'},
            vmin=0,
            vmax=q95
        )
        ax.set_title(f'Zhengzhou Grid Population Heatmap ({target_date or "All Dates"} {target_time or "All Periods"})')
        ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存静态图片
        if save_static:
            static_path = f"static_output/heatmap_{target_date or 'all'}_{target_time or 'all'}.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"静态热力图已保存：{static_path}")

        # 显示图表
        plt.show()

    def plot_time_trend(self, grid_id=None, save_static=True):
        """绘制人口时间趋势图"""
        if len(self.heat_df) == 0:
            print("无有效人口热力数据，无法绘制趋势图")
            return

        df = self.heat_df.copy()
        if grid_id:
            grid_id_str = str(grid_id).strip()  # 去除空格
            df = df[df['tid'] == grid_id_str]
            if len(df) == 0:
                print(f"未找到网格{grid_id}的数据")
                return
            title = f'Resident Population Trend (Grid {grid_id})'
            suffix = f'_grid{grid_id}'
        else:
            df = df.groupby(['date', 'time'])['all_pop'].sum().reset_index()
            title = 'Resident Population Trend (Citywide)'
            suffix = '_citywide'

        # 合并日期和时间
        df['datetime'] = df.apply(
            lambda row: datetime.combine(row['date'], row['time']),
            axis=1
        )
        df = df.sort_values('datetime')

        # 绘制趋势图
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=df, x='datetime', y='all_pop', marker='o', markersize=5)
        plt.title(title)
        plt.xlabel('Date and Time')
        plt.ylabel('Resident Count')
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存静态图片
        if save_static:
            static_path = f"static_output/trend{suffix}.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"静态趋势图已保存：{static_path}")

        # 显示图表
        plt.show()

    def plot_od_flow(self, top_n=1000, target_date=None, save_static=True):
        """绘制OD流图（基于有效OD对数量动态调整）"""
        if self.grid_gdf is None or len(self.travel_df) == 0:
            print("网格数据或出行数据不足，无法绘制OD流图")
            return

        # 动态调整TOP N（不超过有效OD对数量）
        actual_top_n = min(top_n, self.valid_od_pairs)
        if actual_top_n < top_n:
            print(f"提示：有效OD对数量不足{top_n}，实际显示前{actual_top_n}条")

        # 提示占比（基于有效OD对）
        if self.valid_od_pairs > 0:
            ratio = actual_top_n / self.valid_od_pairs * 100
            print(f"当前显示TOP {actual_top_n} OD流（占有效OD对的{ratio:.1f}%）")

        df = self.travel_df.copy()
        if target_date:
            target_dt = pd.to_datetime(target_date)
            df = df[df['date'] == target_dt]
            if len(df) == 0:
                print(f"未找到{target_date}的出行数据")
                return
            title_suffix = f' ({target_date})'
        else:
            title_suffix = ' (All Dates)'

        # 聚合OD流（按流量排序）
        od_flow = df.groupby(['o_tid', 'd_tid'])['all_pop'].sum().reset_index()
        od_flow = od_flow.sort_values('all_pop', ascending=False).head(actual_top_n)

        if len(od_flow) == 0:
            print("无有效OD流数据")
            return

        # 处理网格质心
        grid_centers = self.grid_gdf.copy()
        grid_centers = self._project_centroid(grid_centers)
        grid_centers['lon'] = grid_centers['centroid'].x
        grid_centers['lat'] = grid_centers['centroid'].y
        grid_centers = grid_centers[['tid', 'lon', 'lat']]

        # 合并OD与网格坐标
        od_flow_merged = od_flow.merge(
            grid_centers.rename(columns={'tid': 'o_tid', 'lon': 'o_lon', 'lat': 'o_lat'}),
            on='o_tid', how='left'
        ).merge(
            grid_centers.rename(columns={'tid': 'd_tid', 'lon': 'd_lon', 'lat': 'd_lat'}),
            on='d_tid', how='left'
        )
        # 调试输出
        print(f"OD流合并前：{len(od_flow)} 条，合并后：{len(od_flow_merged)} 条")
        print(f"空值统计（o_lon/d_lon）：\n{od_flow_merged[['o_lon', 'd_lon']].isna().sum()}")
        od_flow = od_flow_merged.dropna(subset=['o_lon', 'd_lon'])

        if len(od_flow) == 0:
            print("未找到有效OD流的网格坐标匹配（检查tid是否一致）")
            return

        # 优化线宽计算
        max_flow = od_flow['all_pop'].max()
        min_flow = od_flow['all_pop'].min()
        od_flow['line_width'] = ((od_flow['all_pop'] - min_flow) / (max_flow - min_flow + 1e-8)) * 9 + 1

        # 绘制静态OD流图
        fig, ax = plt.subplots(figsize=(14, 12))
        self.grid_gdf.boundary.plot(ax=ax, linewidth=0.3, color='gray', alpha=0.5)
        for _, row in od_flow.iterrows():
            ax.plot(
                [row['o_lon'], row['d_lon']],
                [row['o_lat'], row['d_lat']],
                color='blue',
                alpha=0.6,
                linewidth=row['line_width']
            )
        ax.set_title(f'Zhengzhou OD Flows{title_suffix} (Top {len(od_flow)})')
        ax.axis('off')
        plt.tight_layout()

        # 保存静态图片
        if save_static:
            static_path = f"static_output/od_flow_{target_date or 'all'}_top{len(od_flow)}.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"静态OD流图已保存：{static_path}")

        # 显示图表
        plt.show()

    def plot_hourly_distribution(self, save_static=True):
        """绘制时段人口分布对比"""
        if self.grid_gdf is None or len(self.heat_df) == 0:
            print("网格数据或人口热力数据不足，无法绘制对比图")
            return

        # 选择数据量最多的日期
        date_counts = self.heat_df['date'].value_counts()
        if date_counts.empty:
            print("无有效日期数据")
            return
        typical_date = date_counts.idxmax()
        print(f"选择数据量最多的日期进行对比：{typical_date.date()}")
        df = self.heat_df[self.heat_df['date'] == typical_date]

        # 关键时段筛选
        key_times = ['07:30', '12:00', '18:30', '22:00']
        available_times = []
        for t in key_times:
            try:
                time_obj = pd.to_datetime(t, format='%H:%M').time()
                if time_obj in df['time'].unique():
                    available_times.append(time_obj)
            except:
                continue
        if len(available_times) < 2:
            print("有效时段不足，无法绘制对比图")
            return

        # 合并网格数据
        merged_gdf = self.grid_gdf.merge(df, on='tid', how='left')
        merged_gdf['all_pop'] = merged_gdf['all_pop'].fillna(0).astype(float)
        merged_gdf = merged_gdf[merged_gdf['all_pop'] > 0]
        if len(merged_gdf) == 0:
            print("该日期无有效人口数据")
            return

        # 投影转换
        merged_gdf = self._project_centroid(merged_gdf)

        # 绘图布局
        n = len(available_times)
        rows = (n + 1) // 2
        cols = min(2, n)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        # 统一色阶范围
        global_q95 = merged_gdf['all_pop'].quantile(0.95)

        for i, time in enumerate(available_times):
            temp_df = merged_gdf[merged_gdf['time'] == time]
            if temp_df.empty:
                continue

            temp_df.plot(
                column='all_pop',
                cmap='YlOrRd',
                ax=axes[i],
                legend=True,
                legend_kwds={'label': 'Resident Count'},
                vmin=0,
                vmax=global_q95
            )
            axes[i].set_title(f'Population Distribution at {time.strftime("%H:%M")}')
            axes[i].axis('off')

        plt.suptitle(f'Population Distribution Comparison on {typical_date.date()}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存静态图片
        if save_static:
            static_path = f"static_output/hourly_compare_{typical_date.date()}.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"静态时段对比图已保存：{static_path}")

        # 显示图表
        plt.show()


def main():
    visualizer = SignalDataVisualizer()

    # 生成人口热力图
    print("\n生成人口热力图...")
    if not visualizer.heat_df.empty:
        sample_date = visualizer.heat_df['date'].min().strftime('%Y-%m-%d')
        visualizer.plot_grid_heatmap(target_date=sample_date, target_time='18:00')

    # 生成时间趋势图
    print("\n生成时间趋势图...")
    visualizer.plot_time_trend()

    # 生成OD流图（默认显示TOP 1000，自动适配有效OD对数量）
    print("\n生成OD流图...")
    if not visualizer.travel_df.empty:
        sample_date = visualizer.travel_df['date'].min().strftime('%Y-%m-%d')
        visualizer.plot_od_flow(top_n=1000, target_date=sample_date)

    # 生成时段对比图
    print("\n生成时段对比图...")
    visualizer.plot_hourly_distribution()

    print("\n所有可视化操作已完成！")


if __name__ == "__main__":
    main()
