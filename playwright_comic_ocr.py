# playwright_comic_ocr.py

import asyncio
import base64
import json
import logging
from typing import Dict, List, Optional

import aiohttp
from playwright.async_api import ElementHandle, Page, Playwright, async_playwright
from rich.progress import Progress

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlaywrightComicOCR:
    # <-- MODIFIED: 添加了 debug 参数
    def __init__(self, server_url: str, headless: bool, min_image_size: int, batch_size: int, debug: bool = False):
        self.server_url = server_url
        self.headless = headless
        self.min_image_size = min_image_size
        self.batch_size = batch_size
        self.debug = debug # <-- NEW: 保存debug状态
        self.playwright: Optional[Playwright] = None
        self.browser = None
        self.context = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()

    async def _screenshot_element(self, element: ElementHandle) -> Optional[str]:
        try:
            image_bytes = await element.screenshot(type='png')
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.debug(f"元素截图失败: {e}")
            return None

    async def extract_images_from_page(self, page: Page) -> List[Dict]:
        await page.wait_for_load_state('networkidle', timeout=60000)
        await asyncio.sleep(2)
        
        # --- 核心修改在这里 ---
        # 1. 首先，在循环外部获取一次页面标题
        page_title = await page.title()
        
        # 2. 清理标题，使其适合作为文件名
        safe_page_title = page_title.replace('/', '-').replace('\\', '-').strip()[:30] # 增加长度并清理更多非法字符

        image_infos = await page.evaluate(f"""
        () => {{
            return Array.from(document.querySelectorAll('img'))
                .filter(img => img.complete && img.naturalWidth >= {self.min_image_size} && img.naturalHeight >= {self.min_image_size} && img.src)
                .map((img, index) => ({{
                    index: index, src: img.src,
                    naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight,
                }}));
        }}
        """)
        
        extracted = []
        for info in image_infos:
            element = await page.query_selector(f'img[src="{info["src"]}"]')
            if not element: continue
            
            base64_data = await self._screenshot_element(element)
            if base64_data:
                info['base64_data'] = base64_data
                
                # 3. 在循环内部，使用我们已经获取并处理好的标题字符串
                info['filename'] = f"image_{info['index']:03d}_{safe_page_title}.png"
                
                extracted.append(info)
            await element.dispose()
            
        return extracted

    async def process_page(self, url: str, wait_time: int) -> Dict:
        page = await self.context.new_page()
        try:
            await page.goto(url, timeout=90000)
            await asyncio.sleep(wait_time)
            
            images = await self.extract_images_from_page(page)
            if not images:
                return {"url": url, "error": "未找到符合条件的图片"}

            # <-- MODIFIED: 在发送到服务器的config中加入debug标志
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