import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO("best.pt")

def detect_teeth(image, conf_threshold, iou_threshold):
    if image is None:
        return None, "No image uploaded."

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False,
        augment=True,
    )
    result = results[0]
    boxes  = result.boxes
    names  = result.names

    annotated = img_bgr.copy()
    detection_info = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        label = f"{names[cls]}  {conf:.2f}"
        detection_info.append(f"{names[cls]}: {conf:.2%}")

        color = (0, int(200 * conf), int(100 * (1 - conf)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(annotated_rgb)

    n = len(boxes)
    stats = f"**{n} {'teeth' if n != 1 else 'tooth'} detected**\n\n"
    stats += "\n".join(sorted(detection_info)) if detection_info else "No detections."

    return out_img, stats


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    title="RayoScan AI Tooth Detection"
) as demo:

    gr.Markdown("""
    # RayoScan AI Dental Tooth Numbering Detection
    **YOLOv8l | Trained on ~500 dental X-ray images | mAP@0.5: 0.82+**

    Upload a dental X-ray to automatically detect and number individual teeth using the FDI numbering system.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Dental X-Ray")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                label="Confidence Threshold"
            )
            iou_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                label="IoU Threshold"
            )
            run_btn = gr.Button("Detect Teeth", variant="primary")

        with gr.Column(scale=1):
            output_img  = gr.Image(type="pil", label="Annotated Output")
            output_text = gr.Markdown(label="Detection Results")

    run_btn.click(
        fn=detect_teeth,
        inputs=[input_img, conf_slider, iou_slider],
        outputs=[output_img, output_text],
    )

    gr.Markdown("""
    ---
    **Model:** YOLOv8l fine-tuned on dental X-ray dataset  
    **Augmentation:** Mosaic, MixUp, HSV jitter, TTA at inference  
    **Numbering System:** FDI two-digit notation  
    **Built for:** RayoScan AI Diagnostics – AI Intern Assignment  
    """)

demo.launch()
