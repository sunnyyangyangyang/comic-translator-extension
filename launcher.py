# launcher.py

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import select

console = Console()

class ComicOCRLauncher:
    def __init__(self):
        self.flask_process = None

    def start_flask_server(self, script_path="flask_server.py"):
        if self.is_server_running():
            console.print("[green]✓ Flask OCR服务器已在运行中[/green]")
            return True
            
        if not os.path.exists(script_path):
            console.print(f"[red]错误: Flask脚本未找到: {script_path}[/red]")
            return False
            
        console.print("[yellow]正在启动Flask OCR服务器... (请查看下方实时日志)[/yellow]")
        
        # 使用 universal_newlines=True (等同于 text=True) 确保是文本流
        self.flask_process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # 使用行缓冲
        )
        
        # 实时打印子进程的输出，直到服务器启动或超时
        start_time = time.time()
        timeout = 20 # 20秒超时
        
        console.print("[bold cyan]----- Flask Server 日志 (启动中) -----[/bold cyan]")
        
        while time.time() - start_time < timeout:
            # 使用 select 监控 stdout 和 stderr 是否有数据可读
            ready_to_read, _, _ = select.select([self.flask_process.stdout, self.flask_process.stderr], [], [], 0.1)
            
            for stream in ready_to_read:
                line = stream.readline()
                if line:
                    # 去掉末尾的换行符并打印
                    console.print(f"[dim blue]Flask:[/dim blue] {line.strip()}", highlight=False)

            # 检查服务器是否已经健康
            if self.is_server_running():
                console.print("[bold cyan]-------------------------------------[/bold cyan]")
                console.print("[green]✓ Flask OCR服务器启动成功[/green]")
                
                # 创建一个线程来继续打印后续的日志 (可选，但推荐)
                import threading
                def log_stream(stream, prefix):
                    for line in iter(stream.readline, ''):
                        console.print(f"{prefix} {line.strip()}", highlight=False)
                    stream.close()

                threading.Thread(target=log_stream, args=(self.flask_process.stdout, "[dim blue]Flask (stdout):[/dim blue]"), daemon=True).start()
                threading.Thread(target=log_stream, args=(self.flask_process.stderr, "[red]Flask (stderr):[/red]"), daemon=True).start()

                return True
        
        # 如果超时
        console.print("[bold cyan]-------------------------------------[/bold cyan]")
        console.print("[red]✗ Flask服务器启动超时[/red]")
        self.stop_flask_server()
        return False

    def stop_flask_server(self):
        if self.flask_process:
            console.print("[yellow]正在停止Flask服务器...[/yellow]")
            self.flask_process.terminate()
            try: self.flask_process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.flask_process.kill()
            self.flask_process = None
            console.print("[green]✓ Flask服务器已停止[/green]")

    @staticmethod
    def is_server_running():
        try:
            return requests.get("http://localhost:5000/health", timeout=1).status_code == 200
        except requests.RequestException:
            return False

    def signal_handler(self, signum, frame):
        console.print("\n[yellow]收到中断信号，正在优雅关闭...[/yellow]")
        self.stop_flask_server()
        sys.exit(0)

@click.group()
def cli():
    """漫画OCR处理系统命令行工具"""
    pass

@cli.command()
@click.option('--headless', is_flag=True, default=False, help='使用无头模式运行浏览器')
@click.option('--urls-file', type=str, required=True, help='包含待处理URL的JSON文件路径')
@click.option('--output-dir', default='./output', help='结果输出目录')
@click.option('--batch-size', default=3, help='批处理大小 (当前版本暂未使用)')
@click.option('--min-image-size', default=512, help='要提取的图片最小尺寸 (宽度或高度)')
@click.option('--wait-time', default=3, help='每个页面加载后的额外等待时间 (秒)')
@click.option('--debug', is_flag=True, default=False, help='开启Debug模式，保存带检测框的图片') # <-- NEW
def process(headless, urls_file, output_dir, batch_size, min_image_size, wait_time, debug): # <-- MODIFIED
    """启动Playwright批量处理漫画页面"""
    launcher = ComicOCRLauncher()
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)

    try:
        if not launcher.start_flask_server(): return
        if not os.path.exists(urls_file):
            console.print(f"[red]错误: URL文件未找到: {urls_file}[/red]"); return
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = json.load(f).get('urls', [])
        if not urls:
            console.print("[red]错误: URL文件中未找到任何URL[/red]"); return

        table = Table(title="✨ 处理任务配置 ✨")
        table.add_column("参数", style="cyan"); table.add_column("值", style="magenta")
        table.add_row("URL数量", str(len(urls)))
        table.add_row("无头模式", "是" if headless else "否")
        table.add_row("Debug模式", "[bold green]开启[/bold green]" if debug else "关闭") # <-- NEW
        table.add_row("输出目录", output_dir)
        table.add_row("最小图片尺寸", str(min_image_size))
        console.print(table)

        from playwright_comic_ocr import PlaywrightComicOCR
        
        async def run_processing():
            # <-- MODIFIED: 传递debug参数
            async with PlaywrightComicOCR(
                server_url="http://localhost:5000", headless=headless,
                min_image_size=min_image_size, batch_size=batch_size, debug=debug
            ) as ocr_processor:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "ocr_results.json")
                results = await ocr_processor.process_multiple_pages(
                    urls=urls, output_file=output_file, wait_time=wait_time
                )
                console.print(Panel(
                    f"[green]✅ 处理完成！[/green]\n"
                    f"总计页面: {results['stats']['total_pages']}\n"
                    f"成功页面: {results['stats']['processed_pages']}\n"
                    f"识别图片: {results['stats']['total_images']}\n"
                    f"识别气泡: {results['stats']['total_bubbles']}\n"
                    f"[bold yellow]结果已保存至: {output_file}[/bold yellow]",
                    title="📊 结果统计"
                ))

        asyncio.run(run_processing())
    finally:
        launcher.stop_flask_server()

@cli.command()
@click.option('--output', default='urls.json', help='生成的URL配置文件路径')
def create_urls_file(output):
    """创建一个URL配置文件的模板"""
    template = {"description": "请在此处填入你要处理的漫画章节URL列表", "urls": ["https://example-comic-site.com/manga/chapter-1", "https://example-comic-site.com/manga/chapter-2"]}
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓ URL配置文件模板已创建: {output}[/green]")

if __name__ == "__main__":
    cli()