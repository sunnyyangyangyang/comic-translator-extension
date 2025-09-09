#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask服务器 - 接收Chrome插件发送的图片并处理OCR (带详细日志)
与您现有的OCR脚本集成，实现内存处理，不保存任何文件
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
from PIL import Image
import json
import traceback
import requests
import os # 推荐导入，用于后续优化

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# 全局变量存储模型
model_session = None
model_loaded = False

# OCR配置
OCR_CONFIG = {
    "api_url": "http://127.0.0.1:1234/v1/chat/completions",
    "model_name": "qwen2.5vl-3b",
    "system_prompt": "You are a professional comic book text recognition engine. Your task is to accurately extract any and all text from the provided image bubble. Return only the transcribed text no space !"
}

def load_model(model_path='comic-speech-bubble-detector.onnx'):
    """加载ONNX模型"""
    global model_session, model_loaded
    
    try:
        providers = ort.get_available_providers()
        print(f"可用的执行提供者: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ 使用CUDA GPU")
            execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'DmlExecutionProvider' in providers:
            print("✅ 使用DirectML GPU")
            execution_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            print("⚠️ 使用CPU")
            execution_providers = ['CPUExecutionProvider']
            
        model_session = ort.InferenceSession(model_path, providers=execution_providers)
        model_loaded = True
        print("✅ 模型加载成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        model_loaded = False
        return False

def preprocess_image(image, target_size=1024):
    """预处理图片"""
    h, w, _ = image.shape
    scale = min(target_size / h, target_size / w)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (resized_w, resized_h))
    
    padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top_pad, left_pad = (target_size - resized_h) // 2, (target_size - resized_w) // 2
    padded_image[top_pad:top_pad+resized_h, left_pad:left_pad+resized_w] = resized_image
    
    input_image = padded_image[:, :, ::-1] / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    return np.expand_dims(input_image, axis=0).astype(np.float32), scale, left_pad, top_pad

def postprocess_outputs(outputs, scale, left_pad, top_pad, conf_threshold=0.5):
    """后处理模型输出"""
    predictions = np.squeeze(outputs).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions_over_conf = predictions[scores > conf_threshold]
    scores_over_conf = scores[scores > conf_threshold]
    
    if len(predictions_over_conf) == 0:
        return [], []
    
    boxes = predictions_over_conf[:, :4]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 0] = (boxes[:, 0] - left_pad) / scale
    boxes[:, 1] = (boxes[:, 1] - top_pad) / scale
    boxes[:, 2] /= scale
    boxes[:, 3] /= scale
    
    return boxes, scores_over_conf

def merge_boxes_v2(boxes, y_thresh=10, x_overlap_ratio=0.95):
    """合并重叠的框"""
    if len(boxes) < 2:
        return boxes
    
    while True:
        merged_in_pass = False
        merged_boxes = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            current_box = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                box2 = boxes[j]
                x1_start, x1_end = current_box[0], current_box[0] + current_box[2]
                x2_start, x2_end = box2[0], box2[0] + box2[2]
                intersection_width = max(0, min(x1_end, x2_end) - max(x1_start, x2_start))
                min_width = min(current_box[2], box2[2])
                if min_width == 0:
                    continue
                is_horizontally_aligned = (intersection_width / min_width) > x_overlap_ratio
                y1_start, y1_end = current_box[1], current_box[1] + current_box[3]
                y2_start, y2_end = box2[1], box2[1] + box2[3]
                vertical_gap = max(y1_start, y2_start) - min(y1_end, y2_end)
                is_vertically_close = (vertical_gap < y_thresh)
                
                if is_horizontally_aligned and is_vertically_close:
                    new_x = min(x1_start, x2_start)
                    new_y = min(y1_start, y2_start)
                    new_w = max(x1_end, x2_end) - new_x
                    new_h = max(y1_end, y2_end) - new_y
                    current_box = [new_x, new_y, new_w, new_h]
                    used[j] = True
                    merged_in_pass = True
            merged_boxes.append(current_box)
            used[i] = True
        boxes = merged_boxes
        if not merged_in_pass:
            break
            
    return [[int(coord) for coord in box] for box in boxes]

def sort_boxes_manga_style(boxes):
    """按漫画阅读顺序排序框"""
    if not boxes:
        return []
    return sorted(boxes, key=lambda box: (box[1], -box[0]))

def detect_speech_bubbles_memory(image_data, overlap_ratio=0.2):
    """在内存中检测语音气泡"""
    if not model_loaded:
        raise Exception("模型未加载")
    
    # 解码base64图片
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    original_height, original_width, _ = image_cv.shape
    print(f"  - 图像解码成功, 尺寸: {original_width}x{original_height}") # <-- 新增日志
    all_boxes, all_scores = [], []
    
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    # 对长图使用滑动窗口
    if original_height > original_width * 1.5:
        print("  - 检测到长图, 启用滑动窗口模式") # <-- 新增日志
        window_size = original_width
        stride = int(window_size * (1 - overlap_ratio))
        
        for y_start in range(0, original_height, stride):
            y_end = min(y_start + window_size, original_height)
            if y_end - y_start < window_size * 0.2:
                continue
            tile = image_cv[y_start:y_end, :, :]
            input_tensor, scale, left_pad, top_pad = preprocess_image(tile)
            outputs = model_session.run([output_name], {input_name: input_tensor})[0]
            boxes, scores = postprocess_outputs(outputs, scale, left_pad, top_pad)
            
            for box in boxes:
                box[1] += y_start
                all_boxes.append(box)
            all_scores.extend(scores)
    else:
        # 常规图片单次推理
        print("  - 常规图片, 进行单次推理") # <-- 新增日志
        input_tensor, scale, left_pad, top_pad = preprocess_image(image_cv)
        outputs = model_session.run([output_name], {input_name: input_tensor})[0]
        all_boxes, all_scores = postprocess_outputs(outputs, scale, left_pad, top_pad)
    
    print(f"  - 模型原始输出: {len(all_boxes)} 个候选框") # <-- 新增日志

    # NMS处理
    nms_boxes = []
    if len(all_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=0.5, nms_threshold=0.45)
        if len(indices) > 0:
            nms_boxes = [all_boxes[i] for i in indices]
    print(f"  - NMS处理后: {len(nms_boxes)} 个框") # <-- 新增日志
    
    # 合并框
    merged_boxes = merge_boxes_v2(nms_boxes, y_thresh=10, x_overlap_ratio=0.95)
    print(f"  - 文本框合并后: {len(merged_boxes)} 个框") # <-- 新增日志

    # 按漫画阅读顺序排序
    sorted_boxes = sort_boxes_manga_style(merged_boxes)
    print(f"  - 排序后的最终框: {sorted_boxes}") # <-- 新增日志
    
    return image_cv, sorted_boxes

def perform_ocr_memory(image, boxes, padding_pixels=2):
    """在内存中执行OCR"""
    headers = {"Content-Type": "application/json"}
    ocr_results = []
    img_h, img_w = image.shape[:2]

    print(f"  - 开始对 {len(boxes)} 个文本框进行OCR") # <-- 新增日志
    for i, current_box in enumerate(boxes):
        box_num = i + 1
        x, y, w, h = current_box
        print(f"    -> 处理框 {box_num}/{len(boxes)}: [x={x}, y={y}, w={w}, h={h}]") # <-- 新增日志

        # 添加安全的填充
        x1 = max(0, x - padding_pixels)
        y1 = max(0, y - padding_pixels)
        x2 = min(img_w, x + w + padding_pixels)
        y2 = min(img_h, y + h + padding_pixels)

        if not (x2 > x1 and y2 > y1):
            ocr_results.append({"box_number": box_num, "ocr_text": "[Invalid box size]"})
            continue

        cropped_bubble = image[y1:y2, x1:x2]
        print(f"    - 已裁剪气泡, 尺寸: {cropped_bubble.shape[1]}x{cropped_bubble.shape[0]}") # <-- 新增日志
        _, buffer = cv2.imencode('.png', cropped_bubble)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": OCR_CONFIG["model_name"],
            "max_tokens": 512,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": OCR_CONFIG["system_prompt"]},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Extract all text from this image. Return only the raw text content without any explanations or labels."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]
        }
        
        # 打印部分payload信息用于调试，避免打印完整的base64图像数据
        print(f"    - 发送请求到OCR服务 (API: {OCR_CONFIG['api_url']}, Model: {OCR_CONFIG['model_name']})") # <-- 新增日志
        
        try:
            response = requests.post(OCR_CONFIG["api_url"], headers=headers, data=json.dumps(payload), timeout=30) # 增加超时
            response.raise_for_status()
            result_json = response.json()
            ocr_text = result_json['choices'][0]['message']['content'].strip()
            print(f"    - OCR 成功, 结果: '{ocr_text}'") # <-- 新增日志
            ocr_results.append({"box_number": box_num, "ocr_text": ocr_text})
        except Exception as e:
            print(f"    - ❌ OCR失败 - 框 {box_num}: {e}") # <-- 修改日志
            ocr_results.append({"box_number": box_num, "ocr_text": f"[Error: {str(e)}]"})

    return ocr_results

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })

@app.before_request
def log_request_info():
    """在每个请求处理前打印日志"""
    print(f"\n>>> 收到请求: {request.method} {request.path}")
    if request.data:
        print(f"    - 请求体大小: {len(request.data)} bytes")
    if request.headers:
        print(f"    - Content-Type: {request.headers.get('Content-Type')}")
        print(f"    - Origin: {request.headers.get('Origin')}")

@app.route('/process_batch', methods=['POST'])
def process_batch():
    """批量处理图片端点"""
    try:
        if not model_loaded:
            return jsonify({
                "success": False,
                "message": "模型未加载，请先加载模型"
            }), 500

        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({
                "success": False,
                "message": "请求数据无效"
            }), 400

        images = data['images']
        config = data.get('config', {})
        
        print(f"\n======================================================\n收到批量处理请求: {len(images)} 张图片") # <-- 修改日志
        
        all_results = []
        
        for idx, img_data in enumerate(images):
            filename = img_data.get('filename', f'image_{idx}')
            try:
                print(f"\n--- 处理图片 {idx + 1}/{len(images)}: {filename} ---") # <-- 修改日志
                
                # 检测语音气泡
                image, boxes = detect_speech_bubbles_memory(
                    img_data['data'], 
                    config.get('overlap_ratio', 0.2)
                )
                
                print(f"  - [检测阶段完成] 共找到 {len(boxes)} 个气泡")
                
                # 执行OCR
                if boxes and config.get('ocr_enabled', True):
                    ocr_results = perform_ocr_memory(image, boxes)
                    print(f"  - [OCR阶段完成] 共返回 {len(ocr_results)} 个结果")
                else:
                    ocr_results = []
                
                all_results.append({
                    "filename": filename,
                    "width": img_data.get('width', 0),
                    "height": img_data.get('height', 0),
                    "bubble_count": len(boxes),
                    "boxes": [[int(coord) for coord in box] for box in boxes],
                    "ocr_results": ocr_results
                })
                
            except Exception as e:
                print(f"❌ 处理图片 '{filename}' 时发生严重错误: {e}") # <-- 修改日志
                traceback.print_exc() # 打印完整的错误堆栈
                all_results.append({
                    "filename": filename,
                    "error": str(e),
                    "bubble_count": 0,
                    "boxes": [],
                    "ocr_results": []
                })
        
        return jsonify({
            "success": True,
            "message": f"成功处理 {len(images)} 张图片",
            "results": all_results,
            "total_images": len(images),
            "total_bubbles": sum(r.get('bubble_count', 0) for r in all_results)
        })

    except Exception as e:
        print(f"❌ 批量处理请求失败: {e}") # <-- 修改日志
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"处理失败: {str(e)}"
        }), 500

# ... [ /load_model 和 /update_config 端点保持不变 ] ...

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """加载模型端点"""
    try:
        data = request.get_json()
        model_path = data.get('model_path', 'comic-speech-bubble-detector.onnx')
        success = load_model(model_path)
        if success:
            return jsonify({"success": True, "message": "模型加载成功"})
        else:
            return jsonify({"success": False, "message": "模型加载失败"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"加载模型出错: {str(e)}"}), 500

@app.route('/update_config', methods=['POST'])
def update_config():
    """更新OCR配置端点"""
    try:
        data = request.get_json()
        if 'ocr_config' in data:
            OCR_CONFIG.update(data['ocr_config'])
        return jsonify({
            "success": True,
            "message": "配置已更新",
            "current_config": OCR_CONFIG
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"更新配置出错: {str(e)}"}), 500


if __name__ == '__main__':
    print("="*50)
    print("Comic OCR Flask Server (带详细日志)")
    print("="*50)
    
    # 推荐的开发实践: 仅在工作子进程中加载模型，避免重复加载
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print("尝试加载默认模型...")
        load_model()
    
    print("启动Flask服务器...")
    print("服务器地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    print("="*50)
    
    # 在生产环境中，应使用 use_reloader=False
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)