# flask_server.py

import base64
import io
import json
import os
import traceback

import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- 全局变量 ---
model_session = None
model_loaded = False

# --- OCR 配置 ---
OCR_CONFIG = {
    "api_url": "http://127.0.0.1:1234/v1/chat/completions",
    "model_name": "qwen2.5vl-3b",
    "system_prompt": "You are a professional comic book text recognition engine. Your task is to accurately extract any and all text from the provided image bubble. Return only the transcribed text no space !"
}

def load_model(model_path='comic-speech-bubble-detector.onnx'):
    global model_session, model_loaded
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件未找到于 '{model_path}'")
        model_loaded = False
        return False
    try:
        providers = ort.get_available_providers()
        print(f"可用的ONNX Runtime执行提供者: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ 检测到CUDA！正在使用 NVIDIA GPU 加速")
            execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'DmlExecutionProvider' in providers:
            print("✅ 检测到DirectML！正在使用 Windows GPU 加速")
            execution_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            print("⚠️ 未找到兼容的GPU提供者，将使用 CPU。如需加速，请安装onnxruntime-gpu。")
            execution_providers = ['CPUExecutionProvider']
            
        model_session = ort.InferenceSession(model_path, providers=execution_providers)
        model_loaded = True
        print(f"✅ ONNX模型 '{model_path}' 加载成功，执行后端: {model_session.get_providers()}")
        return True
    except Exception as e:
        print(f"❌ ONNX模型加载失败: {e}")
        model_loaded = False
        return False

# <-- NEW FUNCTION: 用于绘制检测框和序号
def draw_boxes_on_image(image_cv, boxes, output_path):
    """在图片上绘制检测框并保存"""
    for i, box in enumerate(boxes):
        x, y, w, h = [int(v) for v in box]
        # 绘制绿色矩形框
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 准备绘制序号文本
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # 获取文本尺寸以便绘制背景
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制一个实心矩形作为文本背景
        cv2.rectangle(image_cv, (x, y - text_h - baseline), (x + text_w, y), (0, 150, 0), -1)
        
        # 绘制白色文本
        cv2.putText(image_cv, label, (x, y - baseline // 2), font, font_scale, (255, 255, 255), thickness)
        
    # 保存带标注的图片
    cv2.imwrite(output_path, image_cv)


# ... 其他函数(preprocess, postprocess, merge, sort)保持不变 ...
def preprocess_image(image, target_size=1024):
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
    predictions = np.squeeze(outputs).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions_over_conf = predictions[scores > conf_threshold]
    scores_over_conf = scores[scores > conf_threshold]
    if len(predictions_over_conf) == 0: return [], []
    boxes = predictions_over_conf[:, :4]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 0] = (boxes[:, 0] - left_pad) / scale
    boxes[:, 1] = (boxes[:, 1] - top_pad) / scale
    boxes[:, 2] /= scale
    boxes[:, 3] /= scale
    return boxes.tolist(), scores_over_conf.tolist()
def merge_boxes(boxes, y_thresh=10, x_overlap_ratio=0.95):
    if len(boxes) < 2: return boxes
    while True:
        merged_in_pass, merged_boxes, used = False, [], [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            current_box = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                box2 = boxes[j]
                x1_s, x1_e = current_box[0], current_box[0] + current_box[2]
                x2_s, x2_e = box2[0], box2[0] + box2[2]
                intersect_w = max(0, min(x1_e, x2_e) - max(x1_s, x2_s))
                min_w = min(current_box[2], box2[2])
                if min_w == 0: continue
                is_horiz_aligned = (intersect_w / min_w) > x_overlap_ratio
                y1_s, y1_e = current_box[1], current_box[1] + current_box[3]
                y2_s, y2_e = box2[1], box2[1] + box2[3]
                vert_gap = max(y1_s, y2_s) - min(y1_e, y2_e)
                is_vert_close = vert_gap < y_thresh
                if is_horiz_aligned and is_vert_close:
                    new_x = min(x1_s, x2_s)
                    new_y = min(y1_s, y2_s)
                    new_w = max(x1_e, x2_e) - new_x
                    new_h = max(y1_e, y2_e) - new_y
                    current_box = [new_x, new_y, new_w, new_h]
                    used[j], merged_in_pass = True, True
            merged_boxes.append(current_box)
            used[i] = True
        boxes = merged_boxes
        if not merged_in_pass: break
    return [[int(c) for c in b] for b in boxes]
def sort_boxes_manga_style(boxes):
    return sorted(boxes, key=lambda box: (box[1], -box[0]))


def detect_speech_bubbles(image_data, overlap_ratio=0.3):
    # ... 此函数内部逻辑不变，只是返回 image_cv 和 sorted_boxes ...
    if not model_loaded: raise Exception("模型未加载")
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_height, original_width, _ = image_cv.shape
    all_boxes, all_scores = [], []
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    if original_height > original_width * 1.5:
        window_size = original_width
        stride = int(window_size * (1 - overlap_ratio))
        for y_start in range(0, original_height, stride):
            y_end = min(y_start + window_size, original_height)
            if y_end - y_start < window_size * 0.2: continue
            tile = image_cv[y_start:y_end, :, :]
            input_tensor, scale, left_pad, top_pad = preprocess_image(tile)
            outputs = model_session.run([output_name], {input_name: input_tensor})[0]
            boxes, scores = postprocess_outputs(outputs, scale, left_pad, top_pad)
            for box in boxes: box[1] += y_start
            all_boxes.extend(boxes); all_scores.extend(scores)
    else:
        input_tensor, scale, left_pad, top_pad = preprocess_image(image_cv)
        outputs = model_session.run([output_name], {input_name: input_tensor})[0]
        all_boxes, all_scores = postprocess_outputs(outputs, scale, left_pad, top_pad)
    nms_boxes = []
    if len(all_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=0.5, nms_threshold=0.45)
        nms_boxes = [all_boxes[i] for i in indices]
    merged = merge_boxes(nms_boxes)
    sorted_boxes = sort_boxes_manga_style(merged)
    return image_cv, sorted_boxes
    
def perform_ocr(image, boxes, padding_pixels=2):
    # ... 此函数内部逻辑不变 ...
    headers, ocr_results = {"Content-Type": "application/json"}, []
    img_h, img_w = image.shape[:2]
    for i, current_box in enumerate(boxes):
        box_num = i + 1
        x, y, w, h = [int(v) for v in current_box]
        x1, y1 = max(0, x - padding_pixels), max(0, y - padding_pixels)
        x2, y2 = min(img_w, x + w + padding_pixels), min(img_h, y + h + padding_pixels)
        if not (x2 > x1 and y2 > y1):
            ocr_results.append({"box_number": box_num, "ocr_text": "[Invalid Box Size]"})
            continue
        cropped_bubble = image[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.png', cropped_bubble)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        payload = {"model": OCR_CONFIG["model_name"], "max_tokens": 512, "temperature": 0.1, "messages": [{"role": "system", "content": OCR_CONFIG["system_prompt"]}, {"role": "user", "content": [{"type": "text", "text": "Extract all text from this image."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]}
        try:
            response = requests.post(OCR_CONFIG["api_url"], headers=headers, data=json.dumps(payload), timeout=45)
            response.raise_for_status()
            ocr_text = response.json()['choices'][0]['message']['content'].strip()
            ocr_results.append({"box_number": box_num, "ocr_text": ocr_text})
        except requests.RequestException as e:
            print(f"❌ OCR请求失败 - 框 {box_num}: {e}")
            ocr_results.append({"box_number": box_num, "ocr_text": f"[OCR Error: {e}]"})
    return ocr_results

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model_loaded})

@app.route('/process_batch', methods=['POST'])
def process_batch():
    if not model_loaded:
        return jsonify({"success": False, "message": "模型未加载"}), 500
    
    data = request.get_json()
    if not data or 'images' not in data:
        return jsonify({"success": False, "message": "请求数据无效"}), 400

    config = data.get('config', {})
    is_debug_mode = config.get('debug', False) # <-- MODIFIED: 获取debug标志

    all_results = []
    for img_data in data['images']:
        filename = img_data.get('filename', 'unknown_image')
        try:
            image_cv, boxes = detect_speech_bubbles(img_data['data'], config.get('overlap_ratio', 0.2))
            
            # <-- MODIFIED: Debug模式逻辑
            if is_debug_mode:
                debug_dir = os.path.join('output', 'debug_images')
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, filename)
                draw_boxes_on_image(image_cv.copy(), boxes, debug_path) # 使用copy以防影响后续OCR

            ocr_results = []
            if boxes and config.get('ocr_enabled', True):
                ocr_results = perform_ocr(image_cv, boxes)
            
            all_results.append({
                "filename": filename, "bubble_count": len(boxes),
                "boxes": boxes, "ocr_results": ocr_results
            })
        except Exception as e:
            traceback.print_exc()
            all_results.append({"filename": filename, "error": str(e)})
    
    return jsonify({
        "success": True, "results": all_results,
        "total_bubbles": sum(r.get('bubble_count', 0) for r in all_results if 'error' not in r)
    })

if __name__ == '__main__':
    print("=" * 50); print("✨ Comic OCR Flask Server ✨"); print("=" * 50)
    load_model()
    print("\n🚀 服务器正在启动..."); print(f"🔗 地址: http://localhost:5000"); print(f"🩺 健康检查: http://localhost:5000/health"); print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)