import streamlit as st
import streamlit_authenticator as stauth
from config import load_config

def check_authentication():
    """
    检查用户是否已认证
    返回 (authenticated, user_name, username)
    """
    config = load_config()
    
    # 从配置中获取认证信息
    auth_config = config.get('auth', {})
    usernames = auth_config.get('usernames', {})
    cookie_name = auth_config.get('cookie', {}).get('name', 'snk_cookie')
    cookie_key = auth_config.get('cookie', {}).get('key', 'your_random_secret_key_here')
    cookie_expiry_days = auth_config.get('cookie', {}).get('expiry_days', 7)
    
    if not usernames:
        # 如果没有配置用户，默认允许访问
        return True, '管理员', 'shawn'
    
    # 创建认证器
    authenticator = stauth.Authenticate(
        credentials={'usernames': usernames},
        cookie_name=cookie_name,
        key=cookie_key,
        cookie_expiry_days=cookie_expiry_days
    )
    
    # 检查认证状态
    if st.session_state.get('authentication_status') is None:
        # 显示登录表单
        authenticator.login()
        if st.session_state.get('authentication_status'):
            return st.session_state.get('authentication_status', False), st.session_state.get('name'), st.session_state.get('username')
        elif st.session_state.get('authentication_status') is False:
            st.error('用户名/密码不正确')
            return False, None, None
        else:
            st.warning('请输入用户名和密码')
            return False, None, None
    
    return st.session_state.get('authentication_status', False), st.session_state.get('name'), st.session_state.get('username')

def show_logout_button():
    """
    显示登出按钮
    """
    if 'authenticator' in st.session_state:
        st.session_state['authenticator'].logout('登出', 'sidebar')

def protect_page():
    """
    保护页面，只有认证用户才能访问
    """
    authenticated, user_name, username = check_authentication()
    if not authenticated:
        st.stop()