#!/usr/bin/env python3
"""
Example usage of PlotAnalysis class
"""

import os
import sys
from collections import defaultdict

# Add the scripts directory to Python path
sys.path.append('/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts')

from plot_analysis import PlotAnalysis

class MockDataset:
    """Mock dataset for demonstration"""
    def __init__(self):
        self.metadata = self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock dataset structure"""
        return {
            "APN": {
                "K024": {
                    "data": [
                        {
                            "class": [
                                {"class_name": "elephant", "class_id": 1},
                                {"class_name": "lion", "class_id": 2}
                            ]
                        }
                    ],
                    "ckp": {
                        "ckp_1": {
                            "train": [
                                {"class": [{"class_name": "elephant", "class_id": 1}]},
                                {"class": [{"class_name": "lion", "class_id": 2}]},
                                {"class": [{"class_name": "elephant", "class_id": 1}]}
                            ],
                            "val": [
                                {"class": [{"class_name": "lion", "class_id": 2}]},
                                {"class": [{"class_name": "elephant", "class_id": 1}]}
                            ]
                        },
                        "ckp_2": {
                            "train": [
                                {"class": [{"class_name": "giraffe", "class_id": 3}]},
                                {"class": [{"class_name": "elephant", "class_id": 1}]}
                            ],
                            "val": [
                                {"class": [{"class_name": "giraffe", "class_id": 3}]}
                            ]
                        }
                    }
                }
            }
        }

def main():
    """Main function demonstrating PlotAnalysis usage"""
    
    # 1. 配置
    config = {
        "plot_analysis": {
            "ckp_piechart": True,      # 生成饼图
            "count_histogram": True     # 生成直方图
        }
    }
    
    # 2. 创建模拟数据集
    dataset = MockDataset()
    
    # 3. 设置输出路径
    analysis_path = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/analysis_output"
    os.makedirs(analysis_path, exist_ok=True)
    
    # 4. 创建分析器
    analyzer = PlotAnalysis(config, dataset, analysis_path)
    
    # 5. 运行分析
    print("🚀 Starting plot analysis...")
    analyzer.run()
    print("✅ Analysis completed!")
    
    # 6. 显示生成的文件
    print("\n📁 Generated files:")
    for root, dirs, files in os.walk(analysis_path):
        for file in files:
            if file.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, file), analysis_path)
                print(f"  📊 {rel_path}")

if __name__ == "__main__":
    main()
