# playwright_comic_ocr.py (with Smart Image Wait)

import asyncio
import base64
import json
import logging
import time # <-- 新增导入
from typing import Dict, List, Optional

import aiohttp
from playwright.async_api import ElementHandle, Page, Playwright, async_playwright
from rich.progress import Progress

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        self.load_all_keywords = [
            "load all pages", "load all images", "view all", "show all",
            "long strip", "webtoon mode", "阅读全话"
        ]

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()

    async def _click_load_all_button(self, page: Page):
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
                    await asyncio.sleep(3)
                    logger.info("✅ 新内容加载完成。")
                    return True
            except Exception:
                continue
        logger.info("未找到可点击的 '加载全部' 按钮。")
        return False

    async def _screenshot_element(self, element: ElementHandle) -> Optional[str]:
        try:
            image_bytes = await element.screenshot(type='png')
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.debug(f"元素截图失败: {e}")
            return None

    # <-- NEW FUNCTION: 智能等待单个图片加载 -->
    async def _wait_for_image_to_load(self, element: ElementHandle, timeout: int = 15) -> bool:
        """
        检查并等待单个图片元素加载完成。
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            is_loaded = await element.evaluate(
                'img => img.complete && typeof img.naturalWidth !== "undefined" && img.naturalWidth > 0'
            )
            if is_loaded:
                return True
            
            # 仅在第一次进入等待时打印消息
            if time.time() - start_time <= 0.5:
                src = await element.get_attribute('src')
                logger.info(f"⏳ 正在等待图片加载完成: {src[:100] if src else 'N/A'}...")

            await asyncio.sleep(0.5) # 每半秒检查一次

        src = await element.get_attribute('src')
        logger.warning(f"⚠️ 图片加载超时 (超过 {timeout} 秒): {src[:100] if src else 'N/A'}")
        return False

    # <-- MODIFIED: 集成了智能等待的图片提取逻辑 -->
    async def extract_images_from_page(self, page: Page) -> List[Dict]:
        """
        获取页面上所有符合条件的图片元素，并确保它们都已完全加载。
        """
        page_title = await page.title()
        safe_page_title = page_title.replace('/', '-').replace('\\', '-').strip()[:30]

        logger.info("正在扫描页面上的所有图片元素...")
        all_img_elements = await page.query_selector_all('img')
        logger.info(f"初步扫描到 {len(all_img_elements)} 个<img>标签。")
        
        extracted = []
        image_index = 0

        for element in all_img_elements:
            try:
                # 1. 智能等待图片加载完成
                is_loaded = await self._wait_for_image_to_load(element)
                
                if not is_loaded:
                    continue # 如果加载失败，跳过此图片

                # 2. 获取图片信息并进行尺寸过滤
                img_info = await element.evaluate(
                    f"""
                    img => ({{
                        src: img.src,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        visible: img.getBoundingClientRect().width > 0 && img.getBoundingClientRect().height > 0
                    }})
                    """
                )

                if not (img_info['visible'] and 
                        img_info['naturalWidth'] >= self.min_image_size and 
                        img_info['naturalHeight'] >= self.min_image_size):
                    continue # 如果图片不可见或尺寸不符，跳过
                
                logger.info(f"  -> 发现有效图片 (尺寸: {img_info['naturalWidth']}x{img_info['naturalHeight']}), 准备截图...")

                # 3. 截图并保存
                base64_data = await self._screenshot_element(element)
                if base64_data:
                    img_info['base64_data'] = base64_data
                    img_info['filename'] = f"image_{image_index:03d}_{safe_page_title}.png"
                    extracted.append(img_info)
                    image_index += 1

            except Exception as e:
                logger.warning(f"处理某个图片元素时出错: {e}")
            finally:
                # 释放元素句柄，防止内存泄漏
                await element.dispose()
        
        logger.info(f"✅ 页面图片提取完成，共成功提取 {len(extracted)} 张有效图片。")
        return extracted

    async def process_page(self, url: str, wait_time: int) -> Dict:
        page = await self.context.new_page()
        try:
            # 增加页面加载超时和使用 networkidle
            await page.goto(url, timeout=90000, wait_until='networkidle')
            await asyncio.sleep(wait_time)
            
            await self._click_load_all_button(page)
            
            # 调用更新后的、带智能等待的提取函数
            images = await self.extract_images_from_page(page)

            if not images:
                return {"url": url, "error": "未找到或未能成功加载任何符合条件的图片"}

            ocr_request = {
                "images": [{"filename": img["filename"], "data": img["base64_data"]} for img in images],
                "config": {"ocr_enabled": True, "debug": self.debug}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/process_batch", json=ocr_request, timeout=180) as response:
                    if response.status == 200:
                        ocr_data = await response.json()
                        img_results_map = {res['filename']: res for res in ocr_data.get('results', [])}
                        page_results = []
                        for img in images:
                            result = img_results_map.get(img['filename'], {})
                            page_results.append({
                                "filename": img['filename'], "width": img['naturalWidth'], "height": img['naturalHeight'],
                                "bubble_count": result.get('bubble_count', 0), "boxes": result.get('boxes', []),
                                "ocr_results": result.get('ocr_results', [])
                            })
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
                if "error" in result:
                    failed_pages.append(result)
                else:
                    all_results.append(result)
                    total_images += result.get('image_count', 0)
                    total_bubbles += result.get('bubble_count', 0)
                progress.advance(task)
        stats = {"total_pages": len(urls), "processed_pages": len(all_results), "total_images": total_images, "total_bubbles": total_bubbles, "failed_pages": [p['url'] for p in failed_pages]}
        final_output = {"stats": stats, "results": all_results}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        return final_output