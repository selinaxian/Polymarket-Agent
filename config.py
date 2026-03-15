import os
import time
import re
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini 模型配置
GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.1  # 低温度 = 更确定性的分析结果

# 重试配置
MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


def invoke_with_retry(chain, inputs: dict, retries: int = MAX_RETRIES) -> dict:
    """带自动重试的 LLM chain 调用，遇到 429 限流自动等待重试"""
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                if attempt < retries - 1:
                    # 尝试从错误信息提取建议等待时间
                    match = re.search(r'retryDelay.*?(\d+)', err)
                    wait = int(match.group(1)) + 5 if match else RETRY_WAIT
                    print(f"  API 限流，等待 {wait}s (重试 {attempt+1}/{retries})...")
                    time.sleep(wait)
                    continue
            raise
