# launcher.py (ULTIMATE FIX version)

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import time
import threading

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class ComicOCRLauncher:
    def __init__(self):
        self.flask_process = None

    def _get_nvidia_lib_paths(self):
        """
        [健壮版] 自动查找当前Python虚拟环境中的NVIDIA CUDA库路径。
        可以处理多个 site-packages 目录。
        """
        try:
            import site
            site_packages_paths = site.getsitepackages()
            
            if not site_packages_paths:
                console.print("[dim yellow]警告: `site.getsitepackages()` 未返回任何路径。[/dim yellow]")
                return None

            console.print(f"[dim]侦测到 site-packages 路径: {site_packages_paths}[/dim]")
            
            # 使用集合避免重复路径
            lib_paths = set()

            for path in site_packages_paths:
                nvidia_base_dir = os.path.join(path, 'nvidia')
                if not os.path.isdir(nvidia_base_dir):
                    continue # 如果此 site-packages 没有 nvidia 目录，则跳过

                for package_name in os.listdir(nvidia_base_dir):
                    package_dir = os.path.join(nvidia_base_dir, package_name)
                    if os.path.isdir(package_dir):
                        lib_dir = os.path.join(package_dir, 'lib')
                        if os.path.isdir(lib_dir):
                            lib_paths.add(lib_dir)
            
            if not lib_paths:
                return None

            return os.pathsep.join(sorted(list(lib_paths)))
            
        except Exception as e:
            console.print(f"[yellow]警告: 自动查找NVIDIA库路径时发生错误: {e}[/yellow]")
            return None

    # ... 其他所有函数 (start_flask_server, stop_flask_server, etc.) 
    # 都与上一个版本完全相同，无需修改。
    # 我在这里重新粘贴一遍以确保完整性。
    
    def start_flask_server(self, script_path="flask_server.py"):
        if self.is_server_running():
            console.print("[green]✓ Flask OCR服务器已在运行中[/green]")
            return True
        if not os.path.exists(script_path):
            console.print(f"[red]错误: Flask脚本未找到: {script_path}[/red]")
            return False
        console.print("[yellow]正在启动Flask OCR服务器... (请查看下方实时日志)[/yellow]")
        
        env = os.environ.copy()
        nvidia_paths = self._get_nvidia_lib_paths()
        
        if nvidia_paths:
            console.print(f"[dim green]自动检测到NVIDIA库，正在准备 LD_LIBRARY_PATH...[/dim green]")
            console.print(f"[dim]  -> Paths: {nvidia_paths.replace(os.pathsep, ' ')}[/dim]")
            existing_ld_path = env.get('LD_LIBRARY_PATH')
            if existing_ld_path:
                env['LD_LIBRARY_PATH'] = f"{nvidia_paths}{os.pathsep}{existing_ld_path}"
            else:
                env['LD_LIBRARY_PATH'] = nvidia_paths
        else:
            console.print("[dim yellow]未在虚拟环境中找到NVIDIA库, 将使用系统默认路径。这可能会导致CUDA失败。[/dim yellow]")

        self.flask_process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            # [关键] 在Linux/macOS上, 启动一个新的进程组
            preexec_fn=os.setsid if sys.platform != "win32" else None
        )
        
        
        start_time = time.time()
        timeout = 20
        console.print("[bold cyan]----- Flask Server 日志 (启动中) -----[/bold cyan]")
        while time.time() - start_time < timeout:
            ready_to_read, _, _ = select.select([self.flask_process.stdout, self.flask_process.stderr], [], [], 0.1)
            for stream in ready_to_read:
                line = stream.readline()
                if line: console.print(f"[dim blue]Flask:[/dim blue] {line.strip()}", highlight=False)
            if self.is_server_running():
                console.print("[bold cyan]-------------------------------------[/bold cyan]")
                console.print("[green]✓ Flask OCR服务器启动成功[/green]")
                def log_stream(stream, prefix):
                    for line in iter(stream.readline, ''):
                        console.print(f"{prefix} {line.strip()}", highlight=False)
                    stream.close()
                threading.Thread(target=log_stream, args=(self.flask_process.stdout, "[dim blue]Flask (stdout):[/dim blue]"), daemon=True).start()
                threading.Thread(target=log_stream, args=(self.flask_process.stderr, "[red]Flask (stderr):[/red]"), daemon=True).start()
                return True
        console.print("[bold cyan]-------------------------------------[/bold cyan]")
        console.print("[red]✗ Flask服务器启动超时[/red]")
        self.stop_flask_server()
        return False

    def stop_flask_server(self):
        """[加强版] 确保Flask服务器及其所有子进程都被彻底终结。"""
        if self.flask_process:
            console.print("[yellow]正在强制停止Flask服务器...[/yellow]")
            
            # 获取进程ID
            pid = self.flask_process.pid
            
            try:
                # 根据操作系统使用不同的、更强大的终结方法
                if sys.platform == "win32":
                    # 在Windows上, 使用 taskkill 强制杀死进程树
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], check=True, capture_output=True)
                else:
                    # 在Linux/macOS上, 使用 os.killpg 杀死整个进程组
                    # 这会杀死由Popen启动的shell及其所有子进程
                    import os
                    import signal
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                
                self.flask_process.wait(timeout=5) # 等待进程状态更新
                console.print("[green]✓ Flask服务器已确认停止[/green]")

            except (ProcessLookupError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # 如果进程已经自己退出了，或者出了其他问题，也认为是停止了
                console.print(f"[dim yellow]服务器进程可能已自行退出或终结失败: {e}[/dim yellow]")
            
            self.flask_process = None

    @staticmethod
    def is_server_running():
        try: return requests.get("http://localhost:5000/health", timeout=1).status_code == 200
        except requests.RequestException: return False

    def signal_handler(self, signum, frame):
        console.print("\n[yellow]收到中断信号，正在优雅关闭...[/yellow]")
        self.stop_flask_server()
        sys.exit(0)

# CLI commands remain the same
@click.group()
def cli(): pass

@cli.command()
@click.option('--headless', is_flag=True, default=False, help='使用无头模式运行浏览器')
@click.option('--urls-file', type=str, required=True, help='包含待处理URL的JSON文件路径')
@click.option('--output-dir', default='./output', help='结果输出目录')
@click.option('--batch-size', default=3, help='批处理大小 (当前版本暂未使用)')
@click.option('--min-image-size', default=512, help='要提取的图片最小尺寸 (宽度或高度)')
@click.option('--wait-time', default=3, help='每个页面加载后的额外等待时间 (秒)')
@click.option('--debug', is_flag=True, default=False, help='开启Debug模式，保存带检测框的图片')
def process(headless, urls_file, output_dir, batch_size, min_image_size, wait_time, debug):
    launcher = ComicOCRLauncher()
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    try:
        if not launcher.start_flask_server(): return
        if not os.path.exists(urls_file): console.print(f"[red]错误: URL文件未找到: {urls_file}[/red]"); return
        with open(urls_file, 'r', encoding='utf-8') as f: urls = json.load(f).get('urls', [])
        if not urls: console.print("[red]错误: URL文件中未找到任何URL[/red]"); return
        table = Table(title="✨ 处理任务配置 ✨")
        table.add_column("参数", style="cyan"); table.add_column("值", style="magenta")
        table.add_row("URL数量", str(len(urls)))
        table.add_row("无头模式", "是" if headless else "否")
        table.add_row("Debug模式", "[bold green]开启[/bold green]" if debug else "关闭")
        table.add_row("输出目录", output_dir)
        table.add_row("最小图片尺寸", str(min_image_size))
        console.print(table)
        from playwright_comic_ocr import PlaywrightComicOCR
        async def run_processing():
            async with PlaywrightComicOCR(server_url="http://localhost:5000", headless=headless, min_image_size=min_image_size, batch_size=batch_size, debug=debug) as ocr_processor:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "ocr_results.json")
                results = await ocr_processor.process_multiple_pages(urls=urls, output_file=output_file, wait_time=wait_time)
                console.print(Panel(f"[green]✅ 处理完成！[/green]\n总计页面: {results['stats']['total_pages']}\n成功页面: {results['stats']['processed_pages']}\n识别图片: {results['stats']['total_images']}\n识别气泡: {results['stats']['total_bubbles']}\n[bold yellow]结果已保存至: {output_file}[/bold yellow]", title="📊 结果统计"))
        asyncio.run(run_processing())
    finally:
        launcher.stop_flask_server()

@cli.command()
@click.option('--output', default='urls.json', help='生成的URL配置文件路径')
def create_urls_file(output):
    template = {"description": "请在此处填入你要处理的漫画章节URL列表", "urls": ["https://example-comic-site.com/manga/chapter-1", "https://example-comic-site.com/manga/chapter-2"]}
    with open(output, 'w', encoding='utf-8') as f: json.dump(template, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓ URL配置文件模板已创建: {output}[/green]")

if __name__ == "__main__":
    cli()