import os
import platform
import matplotlib.font_manager as fm
import warnings

def setup_chinese_font():
    """
    自动配置中文字体，优先检测Linux系统路径下的中文字体（如Noto Sans CJK）
    确保图表和界面中的中文字符正常显示
    """
    system = platform.system()
    
    # 字体搜索路径
    font_search_paths = []
    
    # 根据系统添加字体搜索路径
    if system == "Linux":
        # Linux系统常见中文字体路径
        linux_font_paths = [
            "/usr/share/fonts/truetype/noto",  # Noto字体
            "/usr/share/fonts/opentype/noto",  # Noto OpenType字体
            "/usr/share/fonts/truetype/wqy",   # 文泉驿字体
            "/usr/share/fonts/truetype/arphic", # 文鼎字体
            "/usr/share/fonts/chinese",        # 中文字体目录
            "/usr/share/fonts",                # 通用字体目录
            "/usr/local/share/fonts",          # 本地安装字体
            "/app/.fonts",                     # Docker容器常见字体路径
        ]
        font_search_paths.extend(linux_font_paths)
    elif system == "Windows":
        # Windows系统字体路径
        windows_font_paths = [
            "C:\\Windows\\Fonts",
            os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Microsoft", "Windows", "Fonts")
        ]
        font_search_paths.extend(windows_font_paths)
    elif system == "Darwin":  # macOS
        mac_font_paths = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts")
        ]
        font_search_paths.extend(mac_font_paths)
    
    # 添加当前工作目录的fonts文件夹
    font_search_paths.append(os.path.join(os.getcwd(), "fonts"))
    
    # 字体名称优先级（从高到低）
    font_priority = [
        "Noto Sans CJK SC",      # Google Noto Sans 简体中文
        "Noto Sans CJK TC",      # Google Noto Sans 繁体中文
        "Source Han Sans SC",    # 思源黑体 简体中文
        "Source Han Sans TC",    # 思源黑体 繁体中文
        "WenQuanYi Micro Hei",   # 文泉驿微米黑
        "WenQuanYi Zen Hei",     # 文泉驿正黑
        "Microsoft YaHei",       # 微软雅黑
        "SimHei",                # 黑体
        "SimSun",                # 宋体
        "FangSong",              # 仿宋
        "KaiTi",                 # 楷体
        "Arial Unicode MS",      # Arial Unicode
    ]
    
    # 查找可用的中文字体
    available_fonts = []
    for font_name in font_priority:
        try:
            # 尝试通过字体管理器查找
            font_path = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
            if font_path and os.path.exists(font_path):
                available_fonts.append((font_name, font_path))
                print(f"找到字体: {font_name} - {font_path}")
        except:
            pass
    
    # 如果在字体管理器中没找到，尝试在文件系统中搜索
    if not available_fonts:
        for font_dir in font_search_paths:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        file_lower = file.lower()
                        # 检查常见中文字体文件扩展名和名称模式
                        if any(ext in file_lower for ext in ['.ttf', '.otf', '.ttc']):
                            # 检查文件名是否包含中文字体特征
                            for font_pattern in ['notosanscjk', 'sourcehansans', 'wqy', 'msyh', 'simhei', 'simsun']:
                                if font_pattern in file_lower:
                                    font_path = os.path.join(root, file)
                                    font_name = os.path.splitext(file)[0]
                                    available_fonts.append((font_name, font_path))
                                    print(f"从文件系统找到字体: {font_name} - {font_path}")
                                    break
    
    # 配置matplotlib
    if available_fonts:
        # 选择优先级最高的字体
        selected_font_name, selected_font_path = available_fonts[0]
        
        # 添加字体到matplotlib字体管理器
        if selected_font_path not in fm.findSystemFonts():
            fm.fontManager.addfont(selected_font_path)
            font_prop = fm.FontProperties(fname=selected_font_path)
            selected_font_name = font_prop.get_name()
        
        # 设置matplotlib字体
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = [selected_font_name]
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"已设置中文字体: {selected_font_name}")
        return selected_font_name, selected_font_path
    else:
        # 回退到默认配置
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        warnings.warn("未找到中文字体，使用默认字体配置。图表中的中文可能显示为方框。")
        print("警告: 未找到中文字体，使用默认配置")
        return None, None

def check_font_installation():
    """
    检查系统中已安装的字体，并打印相关信息
    """
    system = platform.system()
    print(f"系统平台: {system}")
    print("已安装的中文字体:")
    
    # 获取所有字体
    fonts = fm.findSystemFonts()
    chinese_fonts = []
    
    # 常见中文字体名称关键词
    chinese_keywords = ['cjk', 'chinese', 'china', 'sc', 'tc', 'hc', 'gb', 'big5', 
                        'noto', 'source', 'han', 'wqy', 'wenquanyi', 'msyh', 
                        'simhei', 'simsun', 'fangsong', 'kaiti', 'yahei', 'heiti']
    
    for font_path in fonts:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name().lower()
            
            # 检查是否包含中文字体关键词
            if any(keyword in font_name for keyword in chinese_keywords):
                chinese_fonts.append((font_prop.get_name(), font_path))
        except:
            continue
    
    # 打印找到的中文字体
    if chinese_fonts:
        for font_name, font_path in sorted(chinese_fonts, key=lambda x: x[0]):
            print(f"  {font_name} - {font_path}")
    else:
        print("  未检测到中文字体")
    
    return chinese_fonts

if __name__ == "__main__":
    # 测试字体配置
    print("=== 中文字体配置测试 ===")
    check_font_installation()
    print("\n=== 自动配置中文字体 ===")
    font_name, font_path = setup_chinese_font()
    if font_name:
        print(f"配置成功: {font_name}")
    else:
        print("配置失败，使用默认字体")