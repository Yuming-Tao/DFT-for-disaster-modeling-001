import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from typing import Dict, List, Optional, Tuple


class OSMDataProcessor:
    """OSM Data Processor for soil permeability analysis"""

    # Mapping of land use types to permeability coefficients (Ks) in mm/h
    LANDUSE_MAPPING: Dict[str, Tuple[float, float]] = {
        'forest': (10.0, 30.0),  # Forest: High permeability
        'grassland': (5.0, 15.0),  # Grassland: Medium permeability
        'farmland': (2.0, 8.0),  # Farmland: Medium-low permeability
        'residential': (0.1, 2.0),  # Residential: Low permeability
        'industrial': (0.01, 0.5),  # Industrial: Extremely low permeability
        'water': (0.0, 0.0),  # Water body: No permeability
        'wetland': (0.05, 1.0),  # Wetland: Low permeability
        'park': (3.0, 10.0),  # Park: Medium permeability
        'construction': (0.1, 1.0),  # Construction site: Low permeability
        'bare_ground': (1.0, 5.0),  # Bare ground: Medium-low permeability
        'default': (1.0, 5.0)  # Default value
    }

    # Supported OSM fields and types
    SUPPORTED_FIELDS: List[str] = ['landuse', 'fclass']  # Supported attribute fields
    SUPPORTED_TYPES: set = set(LANDUSE_MAPPING.keys())  # Valid land use types

    def __init__(self, osm_folder: str, boundary_path: str):
        """Initialize the processor

        Args:
            osm_folder: Path to OSM feature folder
            boundary_path: Path to regional boundary file (GeoJSON/Shapefile)
        """
        self.osm_folder = osm_folder
        self.boundary = self._load_boundary(boundary_path)
        self.landuse_data = None
        self.crs = self._get_crs()
        self.output_crs = "EPSG:3857"  # Uniform output to Web Mercator for visualization

    def _load_boundary(self, path: str) -> Optional[gpd.GeoDataFrame]:
        """Load regional boundary"""
        if not os.path.exists(path):
            print(f"Boundary file not found: {path}")
            return None
        try:
            gdf = gpd.read_file(path)
            print(f"Successfully loaded boundary file: {path} (Feature count: {len(gdf)})")
            return gdf
        except Exception as e:
            print(f"Failed to load boundary file: {str(e)}")
            return None

    def _get_crs(self) -> str:
        """Automatically get coordinate reference system"""
        if self.boundary is not None:
            return self.boundary.crs.to_string()
        return "EPSG:4326"

    def _load_osm_layers(self) -> List[gpd.GeoDataFrame]:
        """Load all valid OSM layers"""
        layers = []
        for file in os.listdir(self.osm_folder):
            if file.lower().endswith('.shp'):  # Case-insensitive check
                file_path = os.path.join(self.osm_folder, file)
                try:
                    gdf = gpd.read_file(file_path)
                    # Check for supported fields and polygon geometry
                    if (any(field in gdf.columns for field in self.SUPPORTED_FIELDS) and
                            gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any()):
                        gdf = gdf.to_crs(self.crs)
                        layers.append(gdf)
                        print(f"Loaded layer: {file} (Fields: {gdf.columns.tolist()}, Features: {len(gdf)})")
                except Exception as e:
                    print(f"Skipped invalid file {file}: {str(e)}")
        return layers

    def extract_landuse_features(self) -> gpd.GeoDataFrame:
        """Extract and clean land use data"""
        layers = self._load_osm_layers()
        if not layers:
            raise ValueError("No valid layers found in OSM folder (requires Polygon features and supported fields)")

        landuse_layers = []
        for layer in layers:
            # Auto-match supported fields
            field = next((f for f in self.SUPPORTED_FIELDS if f in layer.columns), None)
            if not field:
                continue  # Skip layers without valid fields

            # Filter valid types and handle missing values
            layer['landuse_type'] = layer[field].fillna('default').astype(str).apply(
                lambda x: x if x in self.SUPPORTED_TYPES else 'default'
            )
            landuse_layers.append(layer[['landuse_type', 'geometry']])

        if not landuse_layers:
            raise ValueError("No valid land use types found (requires forest/residential etc.)")

        # Concatenate and clip data
        landuse = pd.concat(landuse_layers, ignore_index=True)
        if self.boundary is not None:
            # Compatibility with geopandas new versions
            if hasattr(self.boundary.geometry, 'union_all'):
                landuse = gpd.clip(landuse, self.boundary.geometry.union_all())
            else:
                landuse = gpd.clip(landuse, self.boundary.geometry.unary_union)

        self.landuse_data = landuse.dropna(subset=['landuse_type']).reset_index(drop=True)
        print(f"Land use data extracted: {len(self.landuse_data)} features")
        print(f"Detected land use types: {self.landuse_data['landuse_type'].unique()}")
        return self.landuse_data

    def calculate_permeability(self, method: str = 'median') -> gpd.GeoDataFrame:
        """Calculate soil permeability coefficients"""
        if self.landuse_data is None:
            raise ValueError("Please extract land use features first")

        # Map permeability ranges
        self.landuse_data[['ks_min', 'ks_max']] = self.landuse_data['landuse_type'].apply(
            lambda x: pd.Series(self.LANDUSE_MAPPING.get(x, self.LANDUSE_MAPPING['default']))
        )

        # Calculate final values
        if method == 'median':
            self.landuse_data['ks_value'] = (self.landuse_data['ks_min'] + self.landuse_data['ks_max']) / 2
        elif method == 'mean':
            self.landuse_data['ks_value'] = self.landuse_data[['ks_min', 'ks_max']].mean(axis=1)
        else:
            self.landuse_data['ks_value'] = self.landuse_data['ks_min']  # Default to min

        print(f"Permeability calculation completed (Method: {method})")
        return self.landuse_data

    def rasterize_data(self, output_path: str = "soil_permeability.tif") -> None:
        """Rasterize soil permeability data"""
        if self.landuse_data is None:
            raise ValueError("Please calculate permeability coefficients first")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        # Determine raster extent and resolution
        if self.boundary is None:
            bounds = self.landuse_data.total_bounds
        else:
            bounds = self.boundary.total_bounds

        # Default 30m resolution
        resolution = 30.0
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)

        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': self.crs,
            'transform': rasterio.transform.from_bounds(*bounds, width, height)
        }

        # Perform rasterization
        out_arr = np.zeros((profile['height'], profile['width']), dtype=np.float32)
        shapes = ((geom, val) for geom, val in zip(
            self.landuse_data.to_crs(profile['crs']).geometry,
            self.landuse_data['ks_value']
        ))

        features.rasterize(shapes, out=out_arr, transform=profile['transform'], fill=0)

        # Save to file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(out_arr, 1)
        print(f"Rasterized result saved to: {output_path}")

    def visualize_results(self, output_dir: str, show_basemap: bool = False):
        """Generate multi-dimensional visualization results"""
        os.makedirs(output_dir, exist_ok=True)
        if self.landuse_data is None or len(self.landuse_data) == 0:
            print("No valid land use data for visualization")
            return

        # Reproject data
        landuse_web = self.landuse_data.to_crs(self.output_crs)

        # Explicit boundary check
        boundary_web = None
        if self.boundary is not None and not self.boundary.empty:
            boundary_web = self.boundary.to_crs(self.output_crs)

        # 1. Land use type distribution map
        plt.figure(figsize=(12, 8))
        ax = landuse_web.plot(
            column='landuse_type',
            cmap='Set3',
            legend=True,
            alpha=0.9,
            ax=plt.gca(),
            legend_kwds={'loc': 'lower right', 'bbox_to_anchor': (1.3, 0)}
        )
        if boundary_web is not None:
            boundary_web.boundary.plot(ax=ax, color='red', linewidth=1.5)

        plt.title("Spatial Distribution of Land Use Types", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "landuse_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Permeability coefficient thematic map
        plt.figure(figsize=(12, 8))
        ax = landuse_web.plot(
            column='ks_value',
            cmap='viridis',
            legend=True,
            legend_kwds={'label': 'Permeability Coefficient (mm/h)', 'shrink': 0.8},
            alpha=0.8,
            ax=plt.gca()
        )
        if boundary_web is not None:
            boundary_web.boundary.plot(ax=ax, color='black', linewidth=1.2)

        # Add basemap
        if show_basemap:
            try:
                ctx.add_basemap(ax, crs=self.output_crs, source=ctx.providers.CartoDB.Positron)
            except Exception as e:
                print(f"Basemap loading failed: {str(e)}")

        plt.title("Soil Permeability Coefficient Inference Results", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "permeability_map.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Land use type distribution bar chart
        plt.figure(figsize=(10, 6))
        type_counts = self.landuse_data['landuse_type'].value_counts()
        type_counts.plot(kind='bar', color='skyblue')
        plt.title("Distribution of Land Use Types", fontsize=14)
        plt.xlabel("Land Use Type")
        plt.ylabel("Number of Features")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "landuse_counts.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Permeability coefficient distribution
        plt.figure(figsize=(10, 6))
        self.landuse_data['ks_value'].plot(kind='hist', bins=20, color='lightgreen', alpha=0.7)
        plt.title("Distribution of Permeability Coefficients", fontsize=14)
        plt.xlabel("Permeability Coefficient (mm/h)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "permeability_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Permeability by land use type
        plt.figure(figsize=(12, 6))
        landuse_ks = self.landuse_data.groupby('landuse_type')['ks_value'].mean().sort_values()
        landuse_ks.plot(kind='barh', color='orange')
        plt.title("Average Permeability by Land Use Type", fontsize=14)
        plt.xlabel("Average Permeability Coefficient (mm/h)")
        plt.ylabel("Land Use Type")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "permeability_by_landuse.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All visualization results saved to: {output_dir}")


def main():
    """主函数"""
    # 配置参数（请根据实际路径修改）
    OSM_FOLDER = r"C:\Users\Administrator\Desktop\参数文件\河南省OSM数据"  # OSM要素文件夹
    BOUNDARY_PATH = r"C:\Users\Administrator\Desktop\参数文件\郑州市_市.geojson"  # 区域边界
    OUTPUT_DIR = r"C:\Users\Administrator\Desktop"  # 输出目录

    # Initialize processor
    processor = OSMDataProcessor(
        osm_folder=OSM_FOLDER,
        boundary_path=BOUNDARY_PATH
    )

    try:
        # 1. Extract land use data
        processor.extract_landuse_features()

        # 2. Calculate permeability coefficients
        processor.calculate_permeability(method='median')

        # 3. Rasterize results
        processor.rasterize_data(os.path.join(OUTPUT_DIR, "soil_permeability.tif"))

        # 4. Generate visualizations
        processor.visualize_results(OUTPUT_DIR, show_basemap=False)

        print("\nProcessing completed!")
        print(f"Results saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()