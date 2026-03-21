"""
美联储演讲分析 Agent
功能：分析美联储官员的演讲稿，提取鹰鸽信号，输出结构化分析结果
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import config


def create_fed_speech_agent():
    """创建美联储演讲分析 Agent"""

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    # 分析 prompt —— 这是项目的核心
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位资深的美联储政策分析师。你的任务是分析美联储官员的演讲内容，
提取关于货币政策方向的关键信号。

请严格按照以下 JSON 格式输出分析结果，不要包含任何其他文字：

{{
    "speaker": "演讲者姓名和职位",
    "date": "演讲日期（如果能从文本中判断）",
    "hawkish_dovish_score": 一个从-10到10的整数，-10表示极度鸽派，10表示极度鹰派,
    "key_signals": [
        "关键信号1",
        "关键信号2",
        "关键信号3"
    ],
    "rate_implications": {{
        "direction": "raise / hold / cut",
        "confidence": 一个从0到100的整数表示置信度,
        "reasoning": "简要推理过程"
    }},
    "notable_phrases": [
        "值得关注的原话或措辞"
    ],
    "comparison_to_previous": "与此前表态相比的措辞变化（如果无法判断则填'无法判断'）"
}}"""),
        ("human", "请分析以下美联储官员的演讲内容：\n\n{speech_text}")
    ])

    # 组装 chain: prompt -> LLM -> 解析 JSON
    chain = prompt | llm | JsonOutputParser()

    return chain


def analyze_speech(speech_text: str) -> dict:
    """
    分析一段美联储演讲
    
    Args:
        speech_text: 演讲文本内容
    
    Returns:
        dict: 结构化的分析结果
    """
    chain = create_fed_speech_agent()
    return config.invoke_with_retry(chain, {"speech_text": speech_text})
