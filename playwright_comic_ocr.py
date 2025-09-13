# playwright_comic_ocr.py (with Scroll-to-Bottom and Ad domains NOT filtered)

import asyncio
import base64
import json
import logging
import time
from typing import Dict, List, Optional

import aiohttp
from playwright.async_api import ElementHandle, Page, Playwright, async_playwright
from rich.progress import Progress

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONCURRENT_TASKS = 20

class PlaywrightComicOCR:
    def __init__(self, server_url: str, headless: bool, min_image_size: int, batch_size: int, debug: bool = False):
        self.server_url = server_url
        self.headless = headless
        self.min_image_size = min_image_size
        self.batch_size = batch_size
        self.debug = debug
        self.playwright: Optional[Playwright] = None
        self.browser = None
        self.context = None
        self.semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
        self.load_all_keywords = [
            "load all pages", "load all images", "view all", "show all",
            "long strip", "webtoon mode", "阅读全话"
        ]
        # 根据您的要求，广告过滤功能已被移除

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # 增加视口尺寸，有助于加载更多内容
            viewport={'width': 1920, 'height': 1080}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()

    # <-- NEW FUNCTION: 模拟用户平滑滚动到底部 -->
    async def _scroll_to_bottom_smoothly(self, page: Page):
        """
        平滑地、分段地滚动页面到底部，以确保所有懒加载的图片都被触发。
        """
        logger.info("模拟用户滚动浏览，以确保所有图片都已触发加载...")
        last_height = await page.evaluate('document.body.scrollHeight')
        
        while True:
            # 每次滚动当前视口的高度
            await page.evaluate('window.scrollBy(0, window.innerHeight);')
            # 等待一小段时间让新内容加载
            await asyncio.sleep(0.5)
            
            new_height = await page.evaluate('document.body.scrollHeight')
            if new_height == last_height:
                # 如果滚动后页面高度没有变化，说明已经到底部了
                break
            last_height = new_height
        
        logger.info("✅ 已滚动到页面底部。")
        # 滚动到底部后，再等待网络空闲一次
        await page.wait_for_load_state('networkidle', timeout=15000)


    async def _click_load_all_button(self, page: Page):
        # ... 此函数逻辑不变 ...
        logger.info("正在搜索 '加载全部' 类型的按钮...")
        for keyword in self.load_all_keywords:
            try:
                button = page.get_by_text(keyword, exact=False)
                if await button.is_visible(timeout=2000):
                    button_text = await button.text_content()
                    logger.info(f"✅ 找到按钮: '{button_text.strip()}', 准备点击...")
                    await button.click(timeout=5000)
                    logger.info("...点击完成，等待页面加载新内容...")
                    await page.wait_for_load_state('networkidle', timeout=30000)
                    return True # 返回True表示成功点击
            except Exception:
                continue
        logger.info("未找到可点击的 '加载全部' 按钮。")
        return False # 返回False表示未点击
        
    # ... 其他辅助函数 (_screenshot_element, _wait_for_image_to_load, _process_single_image) 保持不变 ...
    async def _screenshot_element(self, element: ElementHandle) -> Optional[str]:
        try: return base64.b64encode(await element.screenshot(type='png')).decode('utf-8')
        except Exception as e: logger.debug(f"元素截图失败: {e}"); return None
    async def _wait_for_image_to_load(self, element: ElementHandle, timeout: int = 15, retries: int = 1) -> bool:
        src = await element.get_attribute('src'); src_for_log = src[:100] if src else 'N/A'
        for attempt in range(retries + 1):
            try:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if await element.evaluate('img => img.complete && typeof img.naturalWidth !== "undefined" && img.naturalWidth > 0'):
                        if attempt > 0: logger.info(f"  -> ✅ 重试成功: 图片 {src_for_log} 已加载。")
                        return True
                    if time.time() - start_time <= 0.5 and attempt == 0: logger.info(f"⏳ 正在等待图片加载完成: {src_for_log}...")
                    await asyncio.sleep(0.5)
                raise asyncio.TimeoutError("单次尝试超时")
            except asyncio.TimeoutError:
                if attempt < retries: logger.warning(f"  -> ⚠️ 第 {attempt + 1}/{retries + 1} 次尝试加载图片超时，将在2秒后重试: {src_for_log}"); await asyncio.sleep(2)
                else: logger.error(f"❌ 图片加载最终失败 (共尝试 {retries + 1} 次): {src_for_log}"); return False
        return False
    async def _process_single_image(self, element: ElementHandle, index: int, safe_page_title: str) -> Optional[Dict]:
        async with self.semaphore:
            try:
                if not await self._wait_for_image_to_load(element): return None
                img_info = await element.evaluate(f"""img => ({{src: img.src, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight, visible: img.getBoundingClientRect().width > 0 && img.getBoundingClientRect().height > 0}})""")
                if not (img_info['visible'] and img_info['naturalWidth'] >= self.min_image_size and img_info['naturalHeight'] >= self.min_image_size): return None
                logger.info(f"  -> [并发任务] 发现有效图片 #{index} (尺寸: {img_info['naturalWidth']}x{img_info['naturalHeight']}), 准备截图...")
                base64_data = await self._screenshot_element(element)
                if base64_data:
                    img_info['base64_data'] = base64_data; img_info['filename'] = f"image_{index:03d}_{safe_page_title}.png"; img_info['original_index'] = index
                    return img_info
                return None
            except Exception as e: logger.warning(f"处理图片元素 #{index} 时出错: {e}"); return None
            finally: await element.dispose()


    async def extract_images_from_page(self, page: Page) -> List[Dict]:
        # ... 此函数内部逻辑不变 ...
        page_title = await page.title()
        safe_page_title = page_title.replace('/', '-').replace('\\', '-').strip()[:30]
        logger.info("正在扫描页面上的所有图片元素...")
        all_img_elements = await page.query_selector_all('img')
        logger.info(f"初步扫描到 {len(all_img_elements)} 个<img>标签。开始并行处理...")
        tasks = [self._process_single_image(element, i, safe_page_title) for i, element in enumerate(all_img_elements)]
        results = await asyncio.gather(*tasks)
        successful_results = [res for res in results if res is not None]
        sorted_results = sorted(successful_results, key=lambda x: x['original_index'])
        logger.info(f"✅ 页面图片提取完成，共成功提取 {len(sorted_results)} 张有效图片。")
        return sorted_results

    async def process_page(self, url: str, wait_time: int) -> Dict:
        page = await self.context.new_page()
        try:
            await page.goto(url, timeout=90000, wait_until='networkidle')
            await asyncio.sleep(wait_time)
            
            # <-- MODIFIED: 集成新的滚动加载逻辑 -->
            # 1. 尝试点击 "加载全部" 按钮
            clicked = await self._click_load_all_button(page)
            
            # 2. 无论是否点击成功，都执行一次滚动到底部的操作，确保万无一失
            await self._scroll_to_bottom_smoothly(page)
            
            images = await self.extract_images_from_page(page)
            # ... 后续逻辑不变 ...
            if not images: return {"url": url, "error": "未找到或未能成功加载任何符合条件的图片"}
            ocr_request = { "images": [{"filename": img["filename"], "data": img["base64_data"]} for img in images], "config": {"ocr_enabled": True, "debug": self.debug}}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/process_batch", json=ocr_request, timeout=180) as response:
                    if response.status == 200:
                        ocr_data = await response.json()
                        img_results_map = {res['filename']: res for res in ocr_data.get('results', [])}
                        page_results = []
                        for img in images:
                            result = img_results_map.get(img['filename'], {})
                            page_results.append({ "filename": img['filename'], "width": img['naturalWidth'], "height": img['naturalHeight'], "bubble_count": result.get('bubble_count', 0), "boxes": result.get('boxes', []), "ocr_results": result.get('ocr_results', [])})
                        return {"url": url, "image_count": len(images), "bubble_count": ocr_data.get('total_bubbles', 0), "results": page_results}
                    else:
                        return {"url": url, "error": f"OCR服务器错误: {response.status}"}
        except Exception as e:
            logger.error(f"处理页面 {url} 时发生错误: {e}")
            return {"url": url, "error": str(e)}
        finally:
            await page.close()

    async def process_multiple_pages(self, urls: List[str], output_file: str, wait_time: int) -> Dict:
        # ... 此函数内部逻辑不变 ...
        all_results, failed_pages, total_images, total_bubbles = [], [], 0, 0
        with Progress() as progress:
            task = progress.add_task("[cyan]处理中...", total=len(urls))
            for url in urls:
                progress.update(task, description=f"[cyan]处理页面: {url[:80]}...")
                result = await self.process_page(url, wait_time)
                if "error" in result: failed_pages.append(result)
                else: all_results.append(result); total_images += result.get('image_count', 0); total_bubbles += result.get('bubble_count', 0)
                progress.advance(task)
        stats = {"total_pages": len(urls), "processed_pages": len(all_results), "total_images": total_images, "total_bubbles": total_bubbles, "failed_pages": [p['url'] for p in failed_pages]}
        final_output = {"stats": stats, "results": all_results}
        with open(output_file, 'w', encoding='utf-8') as f: json.dump(final_output, f, ensure_ascii=False, indent=2)
        return final_output