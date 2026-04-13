import os
import importlib
import pkgutil
import langchain_classic

def search_class_in_package(package, class_name):   #检查langchain旧版本的类是否在langchain_classic中
    print(f"开始在 {package.__name__} 中深度搜索 {class_name}...\n")
    package_path = os.path.dirname(package.__file__)   #获取包的物理路径
    for loader, module_name, is_pkg in pkgutil.walk_packages([package_path], package.__name__ + "."):   #递归遍历所有子模块
        try:
            module = importlib.import_module(module_name)   #动态导入模块
            if hasattr(module, class_name):   #检查模块中是否有我们要找的类
                print(f"✨ 找到了！")
                print(f"模块路径: {module_name}")
                print(f"导入语句: from {module_name} import {class_name}")
                return
        except Exception:   #忽略掉一些因为缺失依赖而无法导入的模块
            continue
    print(f"❌ 搜遍了整个 {package.__name__} 也没找到 {class_name}。")
    print("这通常意味着该类不在这个包里，或者你的版本过旧。")
search_class_in_package(langchain_classic, "EnsembleRetriever")   #执行搜索