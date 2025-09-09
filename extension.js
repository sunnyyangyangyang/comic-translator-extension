// ==UserScript==
// @name         漫画OCR自动处理
// @name:en      Comic OCR Auto Processor
// @namespace    http://tampermonkey.net/
// @version      1.4
// @description  通过Canvas直接提取页面图片数据进行OCR，规避反盗链，速度更快。
// @description:en  Directly extracts image data from the page using Canvas for OCR, bypassing anti-hotlinking and improving speed.
// @author       You & AI Assistant
// @match        *://*/*
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_addStyle
// @grant        GM_xmlhttpRequest
// @connect      localhost
// @connect      127.0.0.1
// @run-at       document-idle
// ==/UserScript==

(function() {
    'use strict';

    // ... [样式定义代码保持不变] ...
    GM_addStyle(`
        /* 浮动控制面板样式 */
        #comic-ocr-panel {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 280px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 10000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
            transition: all 0.3s ease;
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
        }

        #comic-ocr-panel.minimized {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            overflow: hidden;
            padding: 0;
            background: #764ba2;
        }

        #comic-ocr-panel.minimized:before {
            content: '🎨';
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
            font-size: 24px;
            cursor: pointer;
        }

        #comic-ocr-panel .panel-header {
            padding: 12px 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px 12px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: move;
            user-select: none;
        }

        #comic-ocr-panel.minimized .panel-header,
        #comic-ocr-panel.minimized .panel-content {
            display: none;
        }


        #comic-ocr-panel .panel-title {
            font-weight: bold;
            font-size: 14px;
        }

        #comic-ocr-panel .minimize-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }

        #comic-ocr-panel .panel-content {
            padding: 16px;
        }

        #comic-ocr-panel .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 12px;
        }

        #comic-ocr-panel .info-label {
            opacity: 0.8;
        }

        #comic-ocr-panel .info-value {
            font-weight: bold;
        }

        #comic-ocr-panel .button-row {
            display: flex;
            gap: 8px;
            margin: 12px 0;
        }

        #comic-ocr-panel button {
            flex: 1;
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            background: rgba(255,255,255,0.2);
            color: white;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s ease;
        }

        #comic-ocr-panel button:hover:not(:disabled) {
            background: rgba(255,255,255,0.3);
            transform: translateY(-1px);
        }

        #comic-ocr-panel button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        #comic-ocr-panel .status {
            margin-top: 12px;
            padding: 8px;
            border-radius: 6px;
            font-size: 11px;
            text-align: center;
            transition: all 0.3s ease;
            word-wrap: break-word;
        }

        #comic-ocr-panel .status.success {
            background: rgba(76, 175, 80, 0.3);
            border: 1px solid rgba(76, 175, 80, 0.5);
        }

        #comic-ocr-panel .status.error {
            background: rgba(244, 67, 54, 0.3);
            border: 1px solid rgba(244, 67, 54, 0.5);
        }

        #comic-ocr-panel .status.info {
            background: rgba(33, 150, 243, 0.3);
            border: 1px solid rgba(33, 150, 243, 0.5);
        }

        #comic-ocr-panel .config-section {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }

        #comic-ocr-panel .config-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 11px;
        }

        #comic-ocr-panel .config-item label {
            flex: 1;
            opacity: 0.9;
        }

        #comic-ocr-panel .config-item input {
            width: 60px;
            padding: 4px;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 11px;
        }

        #comic-ocr-panel .config-item input[type="checkbox"] {
            width: auto;
            margin: 0;
            height: 16px;
        }

        /* 高亮检测到的图片 */
        .comic-image-detected {
            outline: 3px solid #4CAF50 !important;
            outline-offset: 2px;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5) !important;
            transition: all 0.3s ease !important;
        }

        .comic-image-processing {
            outline: 3px solid #FF9800 !important;
            outline-offset: 2px;
            box-shadow: 0 0 10px rgba(255, 152, 0, 0.5) !important;
        }

        .comic-image-processed {
            outline: 3px solid #2196F3 !important;
            outline-offset: 2px;
            box-shadow: 0 0 10px rgba(33, 150, 243, 0.5) !important;
        }

        /* 结果显示窗口 */
        #comic-ocr-results {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80%;
            max-width: 600px;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            z-index: 10001;
            overflow: hidden;
            display: none;
            flex-direction: column;
        }

        #comic-ocr-results .results-header {
            background: #667eea;
            color: white;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }

        #comic-ocr-results .results-header h3 {
            margin: 0;
            font-size: 1.2em;
        }

        #comic-ocr-results .results-header button {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }

        #comic-ocr-results .results-content {
            padding: 20px;
            max-height: 70vh;
            overflow-y: auto;
            color: #333;
        }

        #comic-ocr-results .result-item {
            margin-bottom: 16px;
            padding: 12px;
            background: #f5f5f5;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        #comic-ocr-results .result-item-header {
            font-weight: bold;
            margin-bottom: 8px;
            color: #667eea;
            word-break: break-all;
        }

        #comic-ocr-results .result-text {
            line-height: 1.6;
            word-wrap: break-word;
            margin-bottom: 5px;
        }

        .results-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 10000;
            display: none;
        }
    `);

    const CONFIG = {
        SERVER_URL: GM_getValue('server_url', 'http://localhost:5000'),
        MIN_IMAGE_SIZE: GM_getValue('min_image_size', 512),
        AUTO_MODE: GM_getValue('auto_mode', false),
        BATCH_SIZE: GM_getValue('batch_size', 5),
        CHECK_INTERVAL: GM_getValue('check_interval', 3000),
    };

    class ComicOCRProcessor {
        constructor() {
            this.detectedImages = new Map();
            this.isProcessing = false;
            this.isScanning = false;
            this.observer = null;
            this.checkTimer = null;
            this.stats = {
                totalImages: 0,
                validImages: 0,
                processedImages: 0,
                totalBubbles: 0
            };
            this.init();
        }

        init() {
            console.log('🎨 Comic OCR Processor v1.4 Initializing...');
            this.createUI();
            this.bindEvents();
            setTimeout(() => {
                this.scanAllImages();
                if (CONFIG.AUTO_MODE) {
                    this.startAutoMode();
                }
            }, 1500);
        }

        createUI() {
            const panel = document.createElement('div');
            panel.id = 'comic-ocr-panel';
            panel.innerHTML = `
                <div class="panel-header">
                    <div class="panel-title">🎨 漫画OCR</div>
                    <button class.title="最小化" class="minimize-btn" id="minimize-btn">−</button>
                </div>
                <div class="panel-content">
                    <div class="info-row">
                        <span class="info-label">总图片:</span>
                        <span class="info-value" id="total-images">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">符合条件:</span>
                        <span class="info-value" id="valid-images">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">已处理:</span>
                        <span class="info-value" id="processed-images">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">文字框:</span>
                        <span class="info-value" id="total-bubbles">0</span>
                    </div>
                    <div class="button-row">
                        <button id="scan-btn">🔍 扫描</button>
                        <button id="process-btn">⚡ 处理</button>
                        <button id="clear-btn">🗑️ 清除</button>
                    </div>
                    <div class="button-row">
                        <button id="results-btn" disabled>📋 查看结果</button>
                        <button id="settings-btn">⚙️ 设置</button>
                    </div>
                    <div class="config-section" id="config-section" style="display: none;">
                        <div class="config-item">
                            <label for="min-size">最小尺寸:</label>
                            <input type="number" id="min-size" value="${CONFIG.MIN_IMAGE_SIZE}" min="100" max="2048">
                        </div>
                        <div class="config-item">
                            <label for="auto-mode">自动模式:</label>
                            <input type="checkbox" id="auto-mode" ${CONFIG.AUTO_MODE ? 'checked' : ''}>
                        </div>
                        <div class="config-item">
                            <label for="batch-size">批次大小:</label>
                            <input type="number" id="batch-size" value="${CONFIG.BATCH_SIZE}" min="1" max="20">
                        </div>
                    </div>
                    <div id="status" class="status" style="display: none;"></div>
                </div>
            `;
            document.body.appendChild(panel);

            const overlay = document.createElement('div');
            overlay.className = 'results-overlay';
            overlay.id = 'results-overlay';
            document.body.appendChild(overlay);

            const resultsWindow = document.createElement('div');
            resultsWindow.id = 'comic-ocr-results';
            resultsWindow.innerHTML = `
                <div class="results-header">
                    <h3>OCR识别结果</h3>
                    <button id="close-results" title="关闭">✖</button>
                </div>
                <div class="results-content" id="results-content"><p>暂无结果</p></div>
            `;
            document.body.appendChild(resultsWindow);

            this.makeDraggable(panel);
        }

        bindEvents() {
            const panel = document.getElementById('comic-ocr-panel');
            panel.addEventListener('click', (e) => {
                if (panel.classList.contains('minimized')) {
                     panel.classList.remove('minimized');
                }
            });

            document.getElementById('minimize-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                panel.classList.add('minimized');
            });

            document.getElementById('scan-btn').addEventListener('click', () => this.scanAllImages());
            document.getElementById('process-btn').addEventListener('click', () => this.processAllImages());
            document.getElementById('clear-btn').addEventListener('click', () => this.clearAll());
            document.getElementById('results-btn').addEventListener('click', () => this.showResults());
            document.getElementById('settings-btn').addEventListener('click', () => this.toggleSettings());

            document.getElementById('min-size').addEventListener('change', (e) => {
                CONFIG.MIN_IMAGE_SIZE = parseInt(e.target.value, 10) || 512;
                GM_setValue('min_image_size', CONFIG.MIN_IMAGE_SIZE);
                this.scanAllImages(); // Re-scan with new settings
            });
            document.getElementById('auto-mode').addEventListener('change', (e) => {
                CONFIG.AUTO_MODE = e.target.checked;
                GM_setValue('auto_mode', CONFIG.AUTO_MODE);
                if (CONFIG.AUTO_MODE) this.startAutoMode();
                else this.stopAutoMode();
            });
            document.getElementById('batch-size').addEventListener('change', (e) => {
                CONFIG.BATCH_SIZE = parseInt(e.target.value, 10) || 5;
                GM_setValue('batch_size', CONFIG.BATCH_SIZE);
            });

            document.getElementById('close-results').addEventListener('click', () => this.hideResults());
            document.getElementById('results-overlay').addEventListener('click', () => this.hideResults());
        }

        makeDraggable(element) {
            let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
            const header = element.querySelector('.panel-header');
            header.onmousedown = dragMouseDown;

            function dragMouseDown(e) {
                e.preventDefault();
                pos3 = e.clientX;
                pos4 = e.clientY;
                document.onmouseup = closeDragElement;
                document.onmousemove = elementDrag;
            }

            function elementDrag(e) {
                e.preventDefault();
                pos1 = pos3 - e.clientX;
                pos2 = pos4 - e.clientY;
                pos3 = e.clientX;
                pos4 = e.clientY;
                element.style.top = (element.offsetTop - pos2) + "px";
                element.style.left = (element.offsetLeft - pos1) + "px";
                element.style.right = 'auto';
            }

            function closeDragElement() {
                document.onmouseup = null;
                document.onmousemove = null;
            }
        }

        async scanAllImages() {
            if (this.isScanning) {
                return;
            }
            this.isScanning = true;
            this.showStatus('info', '🔍 正在扫描页面图片...');
            try {
                const images = document.querySelectorAll('img');
                const promises = Array.from(images).map(img => this.checkImage(img));
                await Promise.allSettled(promises);
                this.updateStats();
                this.showStatus('success', `✅ 扫描完成: ${this.stats.validImages} 张图片符合条件`);
            } catch (error) {
                this.showStatus('error', '扫描出错，请查看控制台');
            } finally {
                this.isScanning = false;
            }
        }

        async checkImage(imgElement) {
            const imgSrc = imgElement.src || imgElement.dataset.src;
            if (!imgSrc || !imgSrc.startsWith('http')) return;
            const existing = this.detectedImages.get(imgSrc);
            if (existing && existing.width > 0) return;
            await this.waitForImageLoad(imgElement);
            const dimensions = this.getImageDimensions(imgElement);
            const imageInfo = {
                src: imgSrc,
                element: imgElement,
                width: dimensions.width,
                height: dimensions.height,
                isValid: this.isValidComicImage(imgSrc, dimensions),
                processed: false,
                ocrResults: []
            };
            this.detectedImages.set(imgSrc, imageInfo);
            if (imageInfo.isValid) {
                imgElement.classList.add('comic-image-detected');
            } else {
                 imgElement.classList.remove('comic-image-detected');
            }
        }

        waitForImageLoad(img) {
            return new Promise((resolve) => {
                if (img.complete && img.naturalWidth !== 0) return resolve();
                img.onload = () => resolve();
                img.onerror = () => resolve();
            });
        }

        getImageDimensions(imgElement) {
            return {
                width: imgElement.naturalWidth || imgElement.width,
                height: imgElement.naturalHeight || imgElement.height
            };
        }

        isValidComicImage(src, dims) {
            const sizeValid = dims.width >= CONFIG.MIN_IMAGE_SIZE || dims.height >= CONFIG.MIN_IMAGE_SIZE;
            if (!sizeValid) return false;
            const validExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
            const lowerSrc = src.toLowerCase();
            try {
                const url = new URL(src);
                 const hasValidExtension = validExtensions.some(ext => url.pathname.endsWith(ext));
                 if(!hasValidExtension) return false;
            } catch(e) {}
            const excludePatterns = ['avatar', 'logo', 'icon', 'banner', 'header', 'footer', 'ads', 'thumb', 'profile', 'sprite', 'button'];
            return !excludePatterns.some(pattern => lowerSrc.includes(pattern));
        }

        async processAllImages() {
            if (this.isProcessing) {
                this.showStatus('error', '⚠️ 正在处理中，请稍候...');
                return;
            }
            const validImages = Array.from(this.detectedImages.values()).filter(img => img.isValid && !img.processed);
            if (validImages.length === 0) {
                this.showStatus('error', '❌ 没有找到未处理的有效图片');
                return;
            }
            this.isProcessing = true;
            document.getElementById('process-btn').disabled = true;
            this.showStatus('info', `⚡ 开始处理 ${validImages.length} 张图片...`);

            try {
                for (let i = 0; i < validImages.length; i += CONFIG.BATCH_SIZE) {
                    const batch = validImages.slice(i, i + CONFIG.BATCH_SIZE);
                    await this.processBatch(batch);
                    const processedCount = Math.min(i + CONFIG.BATCH_SIZE, validImages.length);
                    this.showStatus('info', `⚡ 已处理 ${processedCount}/${validImages.length} 张图片`);
                }
                this.updateStats();
                document.getElementById('results-btn').disabled = false;
                this.showStatus('success', `✅ 全部处理完成！共识别 ${this.stats.totalBubbles} 个文字框`);
            } catch (error) {
                this.showStatus('error', `❌ 处理失败: ${error.message}`);
            } finally {
                this.isProcessing = false;
                this.updateStats();
            }
        }

        async processBatch(images) {
            const imagePromises = images.map(async (img) => {
                // ... (这部分代码不变)
                try {
                    img.element.classList.remove('comic-image-detected');
                    img.element.classList.add('comic-image-processing');
                    const base64Data = await this.convertImageToBase64(img.element);
                    return {
                        filename: this.extractFilename(img.src),
                        width: img.width,
                        height: img.height,
                        data: base64Data
                    };
                } catch (error) {
                    img.element.classList.remove('comic-image-processing');
                    console.error(`处理图片 ${img.src} 失败:`, error);
                    return { error: error.message, filename: this.extractFilename(img.src) };
                }
            });

            const results = await Promise.all(imagePromises);
            const imageDataList = results.filter(r => !r.error);

            if (imageDataList.length === 0) {
                images.forEach(img => img.processed = true);
                this.updateStats();
                return;
            }

            const requestData = { images: imageDataList, config: { ocr_enabled: true } };

            // ==========================================================
            // ===== v1.6 核心修改：增强网络请求日志 ============
            // ==========================================================
            return new Promise((resolve, reject) => {
                console.log(`[ComicOCR] 即将发送 POST 请求到 ${CONFIG.SERVER_URL}/process_batch`, {
                    dataSize: JSON.stringify(requestData).length,
                    imageCount: imageDataList.length
                });

                GM_xmlhttpRequest({
                    method: 'POST',
                    url: `${CONFIG.SERVER_URL}/process_batch`,
                    headers: { 'Content-Type': 'application/json' },
                    data: JSON.stringify(requestData),
                    timeout: 180000,
                    onload: (response) => {
                        console.log('[ComicOCR] GM_xmlhttpRequest onload 触发。', {
                            status: response.status,
                            statusText: response.statusText,
                            responseTextLength: response.responseText ? response.responseText.length : 0,
                            responseHeaders: response.responseHeaders
                        });

                        try {
                            // 检查响应文本是否为空或无效
                            if (!response.responseText) {
                                throw new Error('服务器返回了空响应。可能是CORS预检失败或服务器崩溃。');
                            }
                            if (response.status >= 400) throw new Error(`服务器错误: ${response.status} ${response.statusText}`);

                            const result = JSON.parse(response.responseText);
                            if (!result.success) throw new Error(result.message || '处理失败');

                            images.forEach((img) => {
                                const res = result.results.find(r => r.filename === this.extractFilename(img.src));
                                if (res) {
                                    img.element.classList.remove('comic-image-processing');
                                    img.element.classList.add('comic-image-processed');
                                    img.processed = true;
                                    if (!res.error) {
                                        img.ocrResults = res.ocr_results || [];
                                        this.stats.totalBubbles += res.bubble_count || 0;
                                    }
                                }
                            });
                            this.updateStats();
                            resolve(result);
                        } catch (error) {
                            // 将错误传递给onerror处理
                            reject(error);
                        }
                    },
                    onerror: (error) => {
                        console.error('[ComicOCR] GM_xmlhttpRequest onerror 触发。', error);
                        reject(new Error('网络请求失败, 请检查本地服务和浏览器控制台网络面板。'));
                    },
                    ontimeout: () => {
                        console.error('[ComicOCR] GM_xmlhttpRequest ontimeout 触发。');
                        reject(new Error('请求超时'));
                    }
                });
            });
        }
    }

        convertImageToBase64(imgElement) {
            return new Promise((resolve, reject) => {
                try {
                    if (!imgElement.complete || imgElement.naturalWidth === 0) {
                        imgElement.onload = () => resolve(this.convertImageElementToBase64(imgElement));
                        imgElement.onerror = () => reject(new Error('图片加载失败: ' + imgElement.src));
                        return;
                    }
                    resolve(this.convertImageElementToBase64(imgElement));
                } catch (error) {
                    console.error('Canvas 操作失败', error);
                    reject(error);
                }
            });
        }

        convertImageElementToBase64(imgElement) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imgElement.naturalWidth;
            canvas.height = imgElement.naturalHeight;
            ctx.drawImage(imgElement, 0, 0, imgElement.naturalWidth, imgElement.naturalHeight);
            const dataURL = canvas.toDataURL('image/png');
            const base64 = dataURL.substring(dataURL.indexOf(',') + 1);
            if (!base64) {
                 throw new Error('无法从Canvas生成Base64数据');
            }
            return base64;
        }

        extractFilename(src) {
            try {
                return new URL(src).pathname.split('/').pop() || 'image';
            } catch {
                return `image_${Date.now()}`;
            }
        }

        updateStats() {
            this.stats.totalImages = this.detectedImages.size;
            this.stats.validImages = Array.from(this.detectedImages.values()).filter(img => img.isValid).length;
            this.stats.processedImages = Array.from(this.detectedImages.values()).filter(img => img.processed).length;

            document.getElementById('total-images').textContent = this.stats.totalImages;
            document.getElementById('valid-images').textContent = this.stats.validImages;
            document.getElementById('processed-images').textContent = this.stats.processedImages;
            document.getElementById('total-bubbles').textContent = this.stats.totalBubbles;

            const hasUnprocessed = Array.from(this.detectedImages.values()).some(img => img.isValid && !img.processed);
            document.getElementById('process-btn').disabled = this.isProcessing || !hasUnprocessed;
        }

        clearAll() {
            document.querySelectorAll('.comic-image-detected, .comic-image-processing, .comic-image-processed')
                .forEach(img => img.classList.remove('comic-image-detected', 'comic-image-processing', 'comic-image-processed'));
            this.detectedImages.clear();
            this.stats = { totalImages: 0, validImages: 0, processedImages: 0, totalBubbles: 0 };
            this.updateStats();
            document.getElementById('results-btn').disabled = true;
            this.showStatus('success', '🗑️ 缓存已清除, 请重新扫描');
        }

        showResults() {
            const processedImages = Array.from(this.detectedImages.values()).filter(img => img.processed);
            if (processedImages.length === 0) {
                this.showStatus('error', '❌ 暂无处理结果');
                return;
            }
            let resultsHTML = '';
            let totalResults = 0;
            processedImages.forEach((img, imgIndex) => {
                if (img.ocrResults && img.ocrResults.length > 0) {
                    resultsHTML += `<div class="result-item">
                        <div class="result-item-header">📷 图片 ${imgIndex + 1}: ${this.extractFilename(img.src)}</div>`;
                    img.ocrResults.forEach(result => {
                        resultsHTML += `<div class="result-text">
                            <strong>#${result.box_number}:</strong> ${result.ocr_text.replace(/\n/g, '<br>')}
                        </div>`;
                        totalResults++;
                    });
                    resultsHTML += `</div>`;
                }
            });
            if (totalResults === 0) {
                resultsHTML = '<p>所有已处理的图片均未识别到文字内容。</p>';
            }
            document.getElementById('results-content').innerHTML = resultsHTML;
            document.getElementById('results-overlay').style.display = 'block';
            document.getElementById('comic-ocr-results').style.display = 'flex';
        }

        hideResults() {
            document.getElementById('results-overlay').style.display = 'none';
            document.getElementById('comic-ocr-results').style.display = 'none';
        }

        toggleSettings() {
            const configSection = document.getElementById('config-section');
            configSection.style.display = configSection.style.display === 'none' ? 'block' : 'none';
        }

        showStatus(type, message, duration = 4000) {
            const statusEl = document.getElementById('status');
            statusEl.className = `status ${type}`;
            statusEl.textContent = message;
            statusEl.style.display = 'block';
            if (duration > 0) {
                setTimeout(() => {
                    if (statusEl.textContent === message) {
                       statusEl.style.display = 'none';
                    }
                }, duration);
            }
        }

        startAutoMode() {
            this.stopAutoMode();
            this.showStatus('info', '🚀 自动模式已启动', 2000);
            this.observer = new MutationObserver((mutationsList) => {
                for (const mutation of mutationsList) {
                    if (mutation.type === 'childList') {
                        mutation.addedNodes.forEach(node => {
                            if (node.nodeType === 1) {
                                const images = node.matches('img') ? [node] : node.querySelectorAll('img');
                                images.forEach(img => this.checkImage(img).then(() => this.updateStats()));
                            }
                        });
                    }
                }
            });
            this.observer.observe(document.body, { childList: true, subtree: true });
            this.checkTimer = setInterval(() => this.scanAllImages(), CONFIG.CHECK_INTERVAL);
        }

        stopAutoMode() {
            if (this.observer) {
                this.observer.disconnect();
                this.observer = null;
            }
            if (this.checkTimer) {
                clearInterval(this.checkTimer);
                this.checkTimer = null;
            }
        }
    }

    new ComicOCRProcessor();

})();