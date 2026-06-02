import subprocess

with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        package = line.strip()
        # 跳过空行和注释
        if not package or package.startswith("#"):
            continue
            
        try:
            print(f"正在安装: {package} ...")
            # 使用 pip 逐个安装，如果失败会抛出异常被 except 捕获
            subprocess.check_call(["pip", "install", package])
            print(f"✅ {package} 安装成功！\n")
        except subprocess.CalledProcessError:
            print(f"❌ {package} 安装失败，已跳过！\n")
            continue