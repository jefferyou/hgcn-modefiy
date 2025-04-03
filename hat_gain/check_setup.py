import os
import shutil
import importlib
import sys


def check_directory_structure():
    """检查目录结构并创建必要的目录"""
    directories = [
        "models",
        "utils",
        "results",
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
        else:
            print(f"目录已存在: {directory}")

    # 确保有__init__.py文件
    for directory in ["models", "utils"]:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass
            print(f"创建文件: {init_file}")
        else:
            print(f"文件已存在: {init_file}")


def check_implementation_files():
    """检查所有实现文件是否存在"""
    files_to_check = [
        ("utils/util.py", "双曲空间操作工具"),
        ("utils/hyperbolic_gain_layer.py", "Hyperbolic GAIN聚合器"),
        ("utils/neigh_samplers.py", "邻居采样器"),
        ("utils/prediction.py", "边预测层"),
        ("utils/hyperbolic_gain_sampling.py", "图采样工具"),
        ("models/base_hgattn.py", "基础HAT模型"),
        ("models/enhanced_hyperbolic_gain.py", "增强的Hyperbolic GAIN模型"),
        ("test_model_init.py", "模型初始化测试脚本"),
        ("train_enhanced_model.py", "训练脚本")
    ]

    all_files_exist = True
    missing_files = []

    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"找到: {file_path} - {description}")
        else:
            print(f"缺失: {file_path} - {description} ⚠️")
            all_files_exist = False
            missing_files.append((file_path, description))

    return all_files_exist, missing_files


def check_imports():
    """检查是否可以导入关键模块"""
    modules_to_check = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("networkx", "NetworkX")
    ]

    all_imports_ok = True

    for module_name, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"成功导入: {module_name} - {description}")
        except ImportError:
            print(f"无法导入: {module_name} - {description} ⚠️")
            all_imports_ok = False

    return all_imports_ok


def try_import_project_modules():
    """尝试导入项目自己的模块"""
    # 确保当前目录在路径中
    sys.path.insert(0, os.getcwd())

    modules_to_check = [
        ("utils.util", "双曲空间操作工具"),
        ("utils.hyperbolic_gain_layer", "Hyperbolic GAIN聚合器"),
        ("utils.neigh_samplers", "邻居采样器"),
        ("utils.prediction", "边预测层"),
        ("models.enhanced_hyperbolic_gain", "增强的Hyperbolic GAIN模型")
    ]

    all_imports_ok = True

    for module_name, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"成功导入项目模块: {module_name} - {description}")
        except ImportError as e:
            print(f"无法导入项目模块: {module_name} - {description} ⚠️")
            print(f"  错误: {e}")
            all_imports_ok = False

    return all_imports_ok


def main():
    print("=" * 50)
    print("Hyperbolic GAIN项目设置检查工具")
    print("=" * 50)

    # 检查基本目录结构
    print("\n[1/4] 检查目录结构...")
    check_directory_structure()

    # 检查所有实现文件
    print("\n[2/4] 检查实现文件...")
    all_files_exist, missing_files = check_implementation_files()
    if not all_files_exist:
        print("\n缺少以下文件:")
        for file_path, description in missing_files:
            print(f"  - {file_path} ({description})")

    # 检查依赖库
    print("\n[3/4] 检查依赖库...")
    all_imports_ok = check_imports()
    if not all_imports_ok:
        print("\n缺少一些必要的依赖库。请使用以下命令安装:")
        print("  pip install tensorflow numpy networkx")

    # 检查项目模块
    print("\n[4/4] 尝试导入项目模块...")
    project_imports_ok = try_import_project_modules()

    # 总结
    print("\n" + "=" * 50)
    print("检查总结:")
    if all_files_exist and all_imports_ok and project_imports_ok:
        print("✅ 所有检查通过！项目设置正确。")
        print("您可以运行以下命令来测试模型初始化:")
        print("  python test_model_init.py")
    else:
        print("❌ 检查失败。请解决上述问题后重试。")

    print("=" * 50)


if __name__ == "__main__":
    main()