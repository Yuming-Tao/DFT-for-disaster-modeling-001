import pandas as pd
import logging
import time
import argparse
from data_loader import fetch_enterprise_data, generate_mock_relations
from supply_chain import calculate_vulnerability
from integrator import integrate_results
from visualizer import visualize_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('analysis.log')]
)
logger = logging.getLogger(__name__)


def main(use_cache=True, force_refresh=False):
    start_time = time.time()
    logger.info("Analysis started.")

    try:
        # 1. 加载企业数据
        enterprises = fetch_enterprise_data(use_cache=use_cache)
        logger.info(f"Enterprises loaded: {len(enterprises)}")

        # 2. 生成供应链关系
        relations = generate_mock_relations(enterprises, use_cache=use_cache)
        logger.info(f"Relations generated: {len(relations)}")

        # 3. 计算脆弱性（仅供应链）
        vulnerability = calculate_vulnerability(enterprises, relations)

        # 4. 整合到网格
        grid_data = integrate_results(enterprises, vulnerability)
        if grid_data.empty:
            raise ValueError("Integrated data is empty!")

        # 5. 可视化
        viz_path = visualize_results(grid_data)
        if viz_path:
            logger.info(f"Visualization saved: {viz_path}")
        grid_data.to_csv("vulnerability_results.csv", index=False)
        logger.info("Results saved to: vulnerability_results.csv")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

    logger.info(f"Analysis finished in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply Chain Vulnerability Analysis (Simplified)")
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh cache')
    args = parser.parse_args()
    main(use_cache=not args.no_cache, force_refresh=args.force_refresh)