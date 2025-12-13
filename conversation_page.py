import streamlit as st
from openai import OpenAI  # openai 2.x 正确导入方式
from typing import List, Dict
import pydantic  # 显式导入pydantic，避免隐式依赖问题

# 验证pydantic版本（调试用，可删除）
st.sidebar.caption(f"Pydantic版本：{pydantic.__version__}")
st.sidebar.caption(f"OpenAI版本：2.11.0")

# ===================== 核心类（适配openai 2.11.0） =====================
class AIModelClient:
    def __init__(self, api_key: str, base_url: str, default_model: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.supported_models = ["gpt-4o", "deepseek-chat", "glm-4", "gemini-3-pro-preview", "doubao-seed-1-6-250615"]
        # 初始化当前模型（校验是否在支持列表）
        self.current_model = default_model if default_model in self.supported_models else "gpt-4o"
        # 对话上下文（用列表保存历史消息）
        self.conversation_history: List[Dict[str, str]] = [ {"role": "system", "content": "你需要使用周易、梅花易数、塔罗牌等书籍跟历史算命结果，通过用户输入或选择相关信息的方式，产出用户的算命结果，为用户的感情，事业，学习，风水选择等方面提供指导和心理安慰。"}]
        # 初始化OpenAI客户端（openai 2.x 标准写法）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=100.0  # 显式指定float类型，避免类型报错
        )

    def switch_model(self, new_model: str) -> bool:
        """切换当前使用的模型"""
        if new_model in self.supported_models:
            self.current_model = new_model
            return True
        return False

    def send_message(self, user_message: str, keep_context: bool = True) -> str:
        """发送消息到当前模型，返回响应内容（适配openai 2.11.0）"""
        # 新增用户消息到上下文
        user_msg = {"role": "user", "content": user_message.strip()}  # 去除首尾空格
        if keep_context:
            self.conversation_history.append(user_msg)
        else:
            self.conversation_history = [self.conversation_history[0], user_msg]

        try:
            # 调用当前选中的模型（openai 2.x 标准调用）
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=self.conversation_history,
                timeout=100.0,
                temperature=0.7  # 新增可选参数，提升兼容性
            )
            # 提取响应内容（openai 2.x 正确取值）
            assistant_content = response.choices[0].message.content.strip()
            assistant_msg = {"role": "assistant", "content": assistant_content}
            # 保留助手响应到上下文
            if keep_context:
                self.conversation_history.append(assistant_msg)
            return assistant_content
        except Exception as e:
            # 详细报错信息，方便排查
            error_info = f"模型调用失败：{str(e)}\n错误类型：{type(e).__name__}"
            return error_info

    def clear_history(self):
        """清空对话上下文"""
        self.conversation_history = [self.conversation_history[0]]

# ===================== Streamlit页面配置 =====================
st.set_page_config(
    page_title="知命阁",
    #page_icon="🤖",
    layout="wide"
)

# 侧边栏：配置API信息 + 模型选择
with st.sidebar:
    st.title("配置中心")
    #API密钥和BaseURL输入（可隐藏，避免泄露）
    api_key = st.text_input(
        "API Key",
        value="sk-mfyuzP5LaqpQ3XT6gKGWpqSyFv75vCG8r4JTAI6gPZff8vGa",
        type="password"
    )
    base_url = st.text_input(
        "Base URL",
        value="https://yunwu.ai/v1",
        help="请确保该地址支持openai 2.x接口规范"
    )

    # 模型选择
    st.divider()
    st.subheader("模型切换")
    supported_models = ["gpt-4o", "deepseek-chat", "glm-4", "gemini-3-pro-preview", "doubao-seed-1-6-250615"]
    selected_model = st.selectbox("选择模型", supported_models)

    # 清空历史按钮
    clear_btn = st.button("🗑️ 清空对话历史", type="secondary")

# ===================== 初始化客户端（Streamlit会话态） =====================
# 用st.session_state保存客户端实例，避免刷新页面丢失状态
if "ai_client" not in st.session_state:
    # 初始化时校验API Key和Base URL非空
    if api_key and base_url:
        st.session_state.ai_client = AIModelClient(api_key=api_key, base_url=base_url)
    else:
        st.session_state.ai_client = None
        st.sidebar.error("⚠️ 请填写有效的API Key和Base URL！")

# 切换模型（如果用户选择的模型和当前不一致）
if st.session_state.ai_client and selected_model != st.session_state.ai_client.current_model:
    switch_success = st.session_state.ai_client.switch_model(selected_model)
    if switch_success:
        st.toast(f"✅ 已切换到模型：{selected_model}", icon="🔄")
    else:
        st.toast(f"❌ 切换失败！不支持模型：{selected_model}", icon="⚠️")

# 清空历史（点击按钮触发）
if clear_btn and st.session_state.ai_client:
    st.session_state.ai_client.clear_history()
    st.toast("🗑️ 已清空对话历史", icon="✅")

# ===================== 聊天界面 =====================
st.title("AI算命助手")
if st.session_state.ai_client:
    st.caption(f"当前使用模型：{st.session_state.ai_client.current_model}")
else:
    st.warning("⚠️ 请先在左侧配置中心填写有效的API Key和Base URL！")

# 展示历史对话（从上下文列表中读取）
if st.session_state.ai_client:
    # 过滤掉system角色的消息，只展示用户和助手的对话
    for msg in st.session_state.ai_client.conversation_history:
        if msg["role"] in ["user", "assistant"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# 用户输入框（仅当客户端初始化成功时显示）
if st.session_state.ai_client and (user_input := st.chat_input("请输入你的问题...")):
    # 展示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用模型并展示响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.ai_client.send_message(user_input, keep_context=True)
        st.markdown(response)