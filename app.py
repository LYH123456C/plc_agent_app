import streamlit as st
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from knowledge import load_chunks_and_vectors, search
from tools import check_scl_syntax

# 1. 网页基础设置
st.set_page_config(page_title="PLC 智能编程助手", page_icon="🤖", layout="wide")
st.title("🤖  PLC  编程助手")
# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.write("当前模型：GLM-4-Air")
    st.divider()  # 画一条分割线
    uploaded_file = st.file_uploader("上传需求文档 (.txt)", type=["txt"])
    if st.button("🗑️ 清空对话记忆"):
        st.session_state.messages = [
            {"role": "system", "content": """你是一个顶级的工业自动化 PLC 编程专家。你的任务是准确理解用户的控制需求，并生成无懈可击的代码。
                                            要求：
                                            1. 默认生成西门子 SCL 代码;如果用户指定“三菱”或“欧姆龙”,你必须输出纯正的 ST 语言：
                                                【绝对禁止】使用 AT %IX 或 AT %QX 这种西门子绝对地址！
                                                必须直接使用 X0, X1, Y0, M0 等软元件名称作为变量名！
                                                例如:X0 : BOOL; Y0 : BOOL;。
                                            2. 必须完整输出 VAR...END_VAR 变量声明区和逻辑控制区。
                                            3. 代码必须有详细的中文注释。
                                            4. 在最终回复前，你必须调用 check_scl_syntax 工具进行语法自检，如果有错必须自己修改。
                                            5.【输出规范】：直接输出完整的 SCL/ST 代码，不要客套话。但是，在代码块输出完毕后，请用 2-3 句话简要说明这段代码的核心逻辑和所使用的关键指令（如定时器、状态机等），方便工程师快速理解。
                                            6.[强制要求]:只要用户的程序涉及多步流程控制(如分拣、打包、红绿灯),你【必须】使用CASE step OF 状态机语法编写,严禁使用多个IF堆砌逻辑!如果不用CASE,将视为严重语法错误！"""}

        ]
        st.rerun()
st.caption("基于 GLM-4 与 RAG 知识库构建，具备代码自检能力")


 # 2. 初始化 API (兼容本地和云端)
load_dotenv()
API_KEY = os.environ.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 缓存知识库，避免重复加载
# 3. 动态生成知识库（云端没有 vectors.npy，需要现场生成）
@st.cache_resource
def init_knowledge():
    import numpy as np
    content = ""
    docs_dir = "docs"
    # 动态读取 docs 文件夹下的所有 txt 文件
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            if file.endswith(".txt"):
                with open(os.path.join(docs_dir, file), "r", encoding="utf-8") as f:
                    content += f.read() + "\n\n"
    
    # 现场切分
    chunks = [content[i:i+300] for i in range(0, len(content), 300)]
    
    # 现场算向量
    vectors = []
    for chunk in chunks:
        resp = client.embeddings.create(model="embedding-3", input=chunk)
        vectors.append(resp.data[0].embedding)
        
    return chunks, np.array(vectors)

chunks, vectors = init_knowledge()

# 3. 工具定义
tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "check_scl_syntax",
            "description": "检查生成的 SCL 代码是否符合基本语法，找出遗漏的括号或分号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "需要检查的 SCL 代码块"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# 4. 聊天记忆初始化
if "messages" not in st.session_state:
    st.session_state.messages = [
    {"role": "system", "content": "你是一个全能 PLC 编程专家。用户让你写西门子代码时，输出 SCL 代码；用户让你写三菱或欧姆龙代码时，输出 ST 代码。你生成代码后，必须调用 check_scl_syntax 工具检查。在最终回复时，必须完整输出包含 VAR 声明、逻辑控制、详细注释的完整代码。"}
]
    

# 5. 渲染历史对话
for msg in st.session_state.messages:
    if msg["role"] != "system" and msg["role"] != "tool":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. 接收用户输入
user_prompt = st.chat_input("请输入你的控制需求")
file_prompt = None

# 如果用户上传了文件，就读取内容
if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    # 把文件内容加上一句指令，打包成问题
    file_prompt = f"请根据上传的需求文档生成代码。"
    if st.sidebar.button("根据文档生成代码"):
        st.session_state.trigger_file = True

# 判断是手动输入还是文档触发
if user_prompt:
    prompt = user_prompt
elif getattr(st.session_state, 'trigger_file', False) and file_prompt:
    prompt = file_prompt
    st.session_state.trigger_file = False
else:
    prompt = None

if prompt:
    # 下面的代码保持你原来的样子，不要动
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 检索知识库
    results = search(prompt, chunks, vectors, top_k=1)
    context = results[0]
    
    full_prompt = f"请参考以下 SCL 语法知识：\n\n{context}\n\n用户需求：{prompt}"
    st.session_state.messages.append({"role": "user", "content": full_prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("AI 正在查阅手册、编写代码并自检..."):
            try:
                # 第一次调用模型（带上工具）
                response = client.chat.completions.create(
                    model="glm-4-air",  # 确认这里换成了 air
                    messages=st.session_state.messages,
                    tools=tools_definition,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    st.info("🔧 检测到代码，AI 正在进行语法自检...")
                    
                    tool_call = response_message.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    tool_result = check_scl_syntax(args.get("code", ""))
                    
                    # 必须用 model_dump() 转成字典，防止页面崩溃
                    st.session_state.messages.append(response_message.model_dump())
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    
                    # 第二次调用模型
                    final_response = client.chat.completions.create(
                        model="glm-4-air", # 确认这里换成了 air
                        messages=st.session_state.messages
                    )
                    final_reply = final_response.choices[0].message.content
                else:
                    final_reply = response_message.content
                
                # 核心修改：清理 Markdown 标记，用代码框显示
                clean_code = final_reply.replace("```scl", "").replace("```st", "").replace("```", "").strip()
                st.code(clean_code, language="pascal")

                # 一键下载代码功能
                st.download_button(
                    label="💾 下载这段代码到本地",
                    data=clean_code,
                    file_name="plc_auto_code.txt",
                    mime="text/plain"
                )
                
                # 保存记忆
                st.session_state.messages[-1]["content"] = prompt 
                st.session_state.messages.append({"role": "assistant", "content": clean_code})
                
            except Exception as e:
                st.error(f"出错了：{e}")