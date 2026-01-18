"""
数据完整性检查与修复页面
集成到Streamlit多页面架构中
"""
import streamlit as st
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_integrity_checker import show_data_integrity_ui

# 页面配置
st.set_page_config(
    page_title="数据完整性检查",
    page_icon="🔍",
    layout="wide"
)

# 显示界面
show_data_integrity_ui()
