#!/usr/bin/env python3
"""
简单的per-class accuracy绘图示例 - 无需命令行参数！

这个脚本展示了如何直接在Python代码中调用API函数来绘制per-class accuracy图表。
完全不需要命令行参数，直接import并调用即可。
"""

import plot_per_class_from_json

# 示例1: 从字典直接画图
def example_plot_from_dict():
    """从字典直接画图的示例"""
    print("=== 示例1: 从字典直接画图 ===")
    
    # 准备测试数据（这些数据通常来自extract_per_class_from_completed.py的输出）
    metrics_dict = {
        'class_names': ['Acer rubrum', 'Quercus alba', 'Betula nigra', 'Pinus strobus'],
        'per_class_accuracy': [0.85, 0.72, 0.91, 0.68],
        'samples_per_class': [100, 80, 120, 95],
        'overall_accuracy': 0.790,
        'balanced_accuracy': 0.790,
        'dataset': 'ENO_C05',
        'method': 'bioclip2',
        'checkpoint': 'ckp_1'
    }
    
    # 直接调用API函数画图
    fig = plot_per_class_from_json.plot_from_dict(
        metrics_dict, 
        title="ENO_C05 Dataset - BioCLIP2 Model",
        output_path="example_single_plot.png",
        show=False  # 设为False避免阻塞，只保存图片
    )
    
    print("✅ 单个数据集的图表已保存到: example_single_plot.png")


# 示例2: 从JSON字符串画图
def example_plot_from_json_string():
    """从JSON字符串画图的示例"""
    print("\n=== 示例2: 从JSON字符串画图 ===")
    
    json_string = '''
    {
        "class_names": ["Acer rubrum", "Quercus alba", "Betula nigra"],
        "per_class_accuracy": [0.78, 0.81, 0.85],
        "samples_per_class": [90, 85, 110],
        "overall_accuracy": 0.813,
        "balanced_accuracy": 0.813,
        "dataset": "MAD_MAD05",
        "method": "openclip",
        "checkpoint": "ckp_2"
    }
    '''
    
    # 直接调用API函数
    fig = plot_per_class_from_json.plot_from_json_string(
        json_string,
        title="MAD_MAD05 Dataset - OpenCLIP Model",
        output_path="example_json_plot.png",
        show=False
    )
    
    print("✅ JSON字符串的图表已保存到: example_json_plot.png")


# 示例3: 比较多个方法
def example_comparison_plot():
    """比较多个方法的示例"""
    print("\n=== 示例3: 比较多个方法 ===")
    
    # 方法1的数据
    method1_metrics = {
        'class_names': ['Acer rubrum', 'Quercus alba', 'Betula nigra'],
        'per_class_accuracy': [0.85, 0.72, 0.91],
        'samples_per_class': [100, 80, 120],
        'overall_accuracy': 0.827,
        'balanced_accuracy': 0.827
    }
    
    # 方法2的数据
    method2_metrics = {
        'class_names': ['Acer rubrum', 'Quercus alba', 'Betula nigra'],
        'per_class_accuracy': [0.78, 0.81, 0.85],
        'samples_per_class': [100, 80, 120],
        'overall_accuracy': 0.813,
        'balanced_accuracy': 0.813
    }
    
    # 方法3的数据
    method3_metrics = {
        'class_names': ['Acer rubrum', 'Quercus alba', 'Betula nigra'],
        'per_class_accuracy': [0.82, 0.75, 0.88],
        'samples_per_class': [100, 80, 120],
        'overall_accuracy': 0.817,
        'balanced_accuracy': 0.817
    }
    
    # 比较多个方法 - 柱状图
    fig_bar = plot_per_class_from_json.plot_comparison_from_dicts(
        [method1_metrics, method2_metrics, method3_metrics],
        labels=['BioCLIP2', 'OpenCLIP', 'CLIP-ViT'],
        title="ENO_C05 Dataset - Method Comparison",
        output_path="example_comparison_bar.png",
        plot_type='bar',
        show=False
    )
    
    # 比较多个方法 - 折线图
    fig_line = plot_per_class_from_json.plot_comparison_from_dicts(
        [method1_metrics, method2_metrics, method3_metrics],
        labels=['BioCLIP2', 'OpenCLIP', 'CLIP-ViT'],
        title="ENO_C05 Dataset - Method Comparison",
        output_path="example_comparison_line.png",
        plot_type='line',
        show=False
    )
    
    print("✅ 方法比较图表已保存:")
    print("   - 柱状图: example_comparison_bar.png")
    print("   - 折线图: example_comparison_line.png")


# 示例4: 从实际的JSON文件读取并画图
def example_plot_from_actual_json():
    """从实际JSON文件画图的示例（如果存在的话）"""
    print("\n=== 示例4: 从实际JSON文件画图 ===")
    
    import os
    import glob
    
    # 查找实际的JSON文件
    json_files = glob.glob("extracted_metrics/*.json")
    
    if json_files:
        print(f"找到 {len(json_files)} 个JSON文件:")
        for f in json_files[:3]:  # 只显示前3个
            print(f"  - {f}")
        
        # 使用第一个文件作为示例
        first_file = json_files[0]
        fig = plot_per_class_from_json.plot_from_json_files(
            [first_file],
            title="Real Dataset Example",
            output_path="example_real_data.png",
            show=False
        )
        print(f"✅ 实际数据图表已保存到: example_real_data.png")
        
        # 如果有多个文件，比较前两个
        if len(json_files) >= 2:
            fig_comp = plot_per_class_from_json.plot_from_json_files(
                json_files[:2],
                labels=[f"Method {i+1}" for i in range(2)],
                title="Real Data Comparison",
                output_path="example_real_comparison.png",
                plot_type='line',
                show=False
            )
            print("✅ 实际数据比较图已保存到: example_real_comparison.png")
    else:
        print("没有找到JSON文件，请先运行extract_per_class_from_completed.py")


def main():
    """主函数 - 运行所有示例"""
    print("🎯 ICICLE-Benchmark Per-Class Accuracy 绘图示例")
    print("=" * 50)
    print("这些示例展示了如何直接在Python代码中调用API函数")
    print("完全不需要命令行参数！")
    print("=" * 50)
    
    # 运行所有示例
    example_plot_from_dict()
    example_plot_from_json_string()
    example_comparison_plot()
    example_plot_from_actual_json()
    
    print("\n" + "=" * 50)
    print("🎉 所有示例运行完成！")
    print("生成的图片文件:")
    print("  - example_single_plot.png")
    print("  - example_json_plot.png") 
    print("  - example_comparison_bar.png")
    print("  - example_comparison_line.png")
    print("  - example_real_data.png (如果有JSON文件)")
    print("  - example_real_comparison.png (如果有多个JSON文件)")
    print("\n💡 在Jupyter Notebook中使用时，将show=True即可直接显示图表")


if __name__ == '__main__':
    main()
