"""
关税/贸易政策新闻抓取模块
数据来源：白宫官网、USTR、Google News RSS（聚合路透社/Bloomberg 等）
"""

import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from datetime import datetime
import re
from html import unescape

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# 关税/贸易关键词，用于过滤
TARIFF_KEYWORDS = [
    "tariff", "tariffs", "trade war", "trade deal", "trade agreement",
    "import duty", "duties", "section 301", "section 232",
    "trade deficit", "trade surplus", "trade policy", "trade representative",
    "customs", "htsus", "retaliatory", "retaliation",
    "wto", "usmca", "nafta",
]

COUNTRY_KEYWORDS = [
    "china", "eu", "european union", "canada", "mexico", "japan",
    "korea", "taiwan", "vietnam", "india", "uk", "britain",
]


def _fetch(url: str) -> requests.Response:
    resp = requests.get(url, headers=BASE_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


# ============================================================
# 白宫官网：总统行政命令 + 声明
# ============================================================

def get_whitehouse_actions(pages: int = 2) -> list[dict]:
    """
    抓取白宫总统行政命令列表，过滤关税/贸易相关条目

    Args:
        pages: 抓取页数（每页约 10 条）

    Returns:
        list[dict]: date, title, url, category
    """
    results = []
    for page in range(1, pages + 1):
        url = f"https://www.whitehouse.gov/presidential-actions/page/{page}/"
        try:
            soup = BeautifulSoup(_fetch(url).text, "html.parser")
        except requests.HTTPError:
            break

        for item in soup.select("ul.wp-block-post-template > li"):
            title_el = item.select_one("h2.wp-block-post-title a")
            date_el = item.select_one(".wp-block-post-date")
            cat_el = item.select_one(".wp-block-post-terms")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            # 只保留贸易/关税相关
            if not _is_tariff_related(title):
                continue

            results.append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "title": title,
                "url": title_el["href"],
                "category": cat_el.get_text(strip=True) if cat_el else "",
                "source": "White House",
            })

    return results


def get_whitehouse_statements(pages: int = 2) -> list[dict]:
    """抓取白宫声明/新闻稿，过滤关税/贸易相关"""
    results = []
    for page in range(1, pages + 1):
        url = f"https://www.whitehouse.gov/briefing-room/statements-releases/page/{page}/"
        try:
            soup = BeautifulSoup(_fetch(url).text, "html.parser")
        except requests.HTTPError:
            break

        for item in soup.select("ul.wp-block-post-template > li"):
            title_el = item.select_one("h2.wp-block-post-title a")
            date_el = item.select_one(".wp-block-post-date")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            if not _is_tariff_related(title):
                continue

            results.append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "title": title,
                "url": title_el["href"],
                "category": "Statement",
                "source": "White House",
            })

    return results


# ============================================================
# USTR：美国贸易代表办公室公告
# ============================================================

def get_ustr_releases(limit: int = 20) -> list[dict]:
    """
    抓取 USTR 新闻公告，过滤关税/贸易相关

    Args:
        limit: 最大返回条数
    """
    results = []
    url = "https://ustr.gov/about-us/policy-offices/press-office/press-releases"

    try:
        soup = BeautifulSoup(_fetch(url).text, "html.parser")
    except Exception:
        return results

    main = soup.select_one("#main-content") or soup
    # USTR 页面结构：日期文本 + 链接交替出现
    for link in main.select("a[href*='/press-releases/']"):
        title = link.get_text(strip=True)
        if not title or len(title) < 10:
            continue

        href = link.get("href", "")
        if not href.startswith("http"):
            href = "https://ustr.gov" + href

        # 从链接路径提取日期 (e.g. /press-releases/2025/march/...)
        date_match = re.search(r"/press-releases/(\d{4})/(\w+)/", href)
        date_str = f"{date_match.group(2).title()} {date_match.group(1)}" if date_match else ""

        if not _is_tariff_related(title):
            continue

        results.append({
            "date": date_str,
            "title": title,
            "url": href,
            "category": "USTR Press Release",
            "source": "USTR",
        })

        if len(results) >= limit:
            break

    return results


# ============================================================
# Google News RSS：聚合路透社/Bloomberg/主流媒体的关税新闻
# ============================================================

def get_trade_news_rss(query: str = "tariff trade policy", limit: int = 15) -> list[dict]:
    """
    从 Google News RSS 获取关税/贸易相关新闻

    聚合路透社、Bloomberg、CNBC、WSJ 等主流来源。

    Args:
        query: 搜索词
        limit: 返回条数

    Returns:
        list[dict]: date, title, url, source, summary
    """
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = _fetch(rss_url)
    except Exception:
        return []

    root = ET.fromstring(resp.content)
    results = []

    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        pub_date = item.findtext("pubDate", "")
        source_el = item.find("source")
        source_name = source_el.text if source_el is not None else ""
        description = item.findtext("description", "")
        # 清理 HTML 标签
        summary = unescape(re.sub(r"<[^>]+>", "", description)).strip()

        results.append({
            "date": _parse_rss_date(pub_date),
            "title": title,
            "url": link,
            "source": source_name,
            "summary": summary[:300],
            "category": "News",
        })

        if len(results) >= limit:
            break

    return results


def get_trump_tariff_posts(limit: int = 10) -> list[dict]:
    """
    获取 Trump 关税相关动态（通过 Google News RSS 间接获取报道）

    X/Twitter 需要 API 认证无法直接抓取，
    但主流媒体会报道 Trump 的关税相关推文/声明，通过 RSS 聚合获取。
    """
    return get_trade_news_rss(
        query="Trump tariff announcement",
        limit=limit,
    )


# ============================================================
# 聚合函数：一次获取所有来源
# ============================================================

def get_all_tariff_news(limit_per_source: int = 10) -> list[dict]:
    """
    从所有来源聚合关税/贸易相关新闻，按日期排序

    Returns:
        list[dict]: 合并去重后的新闻列表
    """
    all_news = []

    # 白宫
    all_news.extend(get_whitehouse_actions(pages=2))
    all_news.extend(get_whitehouse_statements(pages=2))

    # USTR
    all_news.extend(get_ustr_releases(limit=limit_per_source))

    # Google News RSS（覆盖路透社/Bloomberg 等）
    all_news.extend(get_trade_news_rss("tariff trade policy", limit=limit_per_source))
    all_news.extend(get_trump_tariff_posts(limit=5))

    # 按标题去重
    seen = set()
    unique = []
    for n in all_news:
        key = n["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(n)

    return unique


# ============================================================
# 辅助函数
# ============================================================

def _is_tariff_related(text: str) -> bool:
    """判断文本是否与关税/贸易相关"""
    t = text.lower()
    return any(k in t for k in TARIFF_KEYWORDS + COUNTRY_KEYWORDS + ["trade", "tariff", "import", "export"])


def _parse_rss_date(date_str: str) -> str:
    """解析 RSS 日期格式 -> YYYY-MM-DD"""
    try:
        # RFC 822: "Sat, 22 Mar 2025 07:00:00 GMT"
        dt = datetime.strptime(date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            dt = datetime.strptime(date_str.strip()[:25], "%a, %d %b %Y %H:%M:%S")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return date_str


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=== 白宫行政命令（关税相关）===\n")
    wh_actions = get_whitehouse_actions(pages=3)
    for a in wh_actions[:5]:
        print(f"  [{a['date']}] {a['title']}")
        print(f"    {a['url']}")
    print(f"  共 {len(wh_actions)} 条\n")

    print("=== 白宫声明（关税相关）===\n")
    wh_stmts = get_whitehouse_statements(pages=3)
    for s in wh_stmts[:5]:
        print(f"  [{s['date']}] {s['title']}")
    print(f"  共 {len(wh_stmts)} 条\n")

    print("=== USTR 公告 ===\n")
    ustr = get_ustr_releases()
    for u in ustr[:5]:
        print(f"  [{u['date']}] {u['title']}")
    print(f"  共 {len(ustr)} 条\n")

    print("=== Google News RSS（关税新闻）===\n")
    news = get_trade_news_rss("tariff trade policy", limit=8)
    for n in news[:5]:
        print(f"  [{n['date']}] {n['title']}")
        print(f"    来源: {n['source']}")
    print(f"  共 {len(news)} 条\n")

    print("=== Trump 关税动态 ===\n")
    trump = get_trump_tariff_posts(limit=5)
    for t in trump[:5]:
        print(f"  [{t['date']}] {t['title']}")
        print(f"    来源: {t['source']}")
    print(f"  共 {len(trump)} 条\n")

    print("=== 聚合汇总 ===")
    all_news = get_all_tariff_news()
    print(f"  总计: {len(all_news)} 条不重复新闻")
    sources = {}
    for n in all_news:
        sources[n["source"]] = sources.get(n["source"], 0) + 1
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {cnt} 条")
