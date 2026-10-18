import os
import urllib.request
import torch
import torchvision.transforms as transforms
import timm
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ==========================================
# WEB PAGE SETTINGS & MODEL INITIALIZATION
# ==========================================
st.set_page_config(page_title="AWC Wildlife Engine Dashboard", layout="wide")
st.title("🐾 AWC Live Wildlife Web Engine")
st.markdown("### Field Agent Deployment Web Console")

# Routes the video connection smoothly over public internet firewalls
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:://google.com"]}]}
)

@st.cache_resource
def load_models():
    # --- AUTOMATED CLOUD DOWNLOADER ---
    CLASSIFIER_FILENAME = "awc-135-v1.pth"
    if not os.path.exists(CLASSIFIER_FILENAME):
        with st.spinner("Downloading Quoll Classifier Weights from Cloud Storage (~110MB)... This may take a moment."):
            # Direct download URL for awc-135-v1.pth 
            awc_url = "https://huggingface.co/Addax-Data-Science/AWC135-AWC-v1/resolve/main/awc-135-v1.pth"
            urllib.request.urlretrieve(awc_url, CLASSIFIER_FILENAME)
            st.success("Download Complete!")

    print("[Web AI] Mounting Vision Core Frameworks...")
    detector = YOLO('yolov8n.pt')
    
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
        
    classifier = timm.create_model('efficientnet_b5', pretrained=False, num_classes=len(labels))
    state_dict = torch.load(CLASSIFIER_FILENAME, map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    classifier.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(456),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return detector, classifier, labels, preprocess

detector, classifier, labels, preprocess = load_models()

# ==========================================
# VIDEO STREAMING ENGINE
# ==========================================
class WildlifeVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        results_list = detector(pil_img, verbose=False)
        
        if len(results_list) > 0 and results_list[0].boxes is not None:
            for box in results_list[0].boxes:
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                
                if conf > 0.45:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    is_human = (cls_id == 0)
                    is_yolo_animal = (14 <= cls_id <= 23)
                    
                    if is_human:
                        box_color = (45, 110, 235)  
                        label_text = f"Human ({conf*100:.1f}%)"
                    elif is_yolo_animal:
                        cropped_animal = pil_img.crop((x1, y1, x2, y2))
                        input_tensor = preprocess(cropped_animal).unsqueeze(0)
                        
                        with torch.no_grad():
                            outputs = classifier(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
                            prob_conf, class_idx = torch.max(probabilities, 0)
                            detected_species = labels[class_idx.item()]
                        
                        if prob_conf.item() > 0.88:
                            box_color = (105, 215, 75)  
                            label_text = f"{detected_species} ({prob_conf.item()*100:.1f}%)"
                        else:
                            continue
                    else:
                        continue

                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                    panel_y1 = max(y1 - text_h - 14, 0)
                    panel_y2 = y1 if y1 - text_h - 14 > 0 else text_h + 14
                    
                    cv2.rectangle(img, (x1, panel_y1), (x1 + text_w + 16, panel_y2), (25, 25, 25), cv2.FILLED)
                    cv2.rectangle(img, (x1, panel_y1), (x1 + 3, panel_y2), box_color, cv2.FILLED)
                    cv2.putText(img, label_text, (x1 + 10, panel_y2 - 6), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        return img

# ==========================================
# WEB USER INTERFACE STYLING
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Live Analysis Feed")
    webrtc_streamer(
        key="wildlife-tracker", 
        video_transformer_factory=WildlifeVideoTransformer,
        rtc_configuration=RTC_CONFIG
    )

with col2:
    st.markdown("#### System Metrics")
    st.success("🟢 CORE STATUS: ONLINE")
    st.metric(label="YOLO Base Engine", value="v8n Active")
    st.metric(label="Classifier Core Network", value="EfficientNet-B5")
    st.info("💡 Note: Camera access permissions must be granted to view your webcam stream inside the browser.")
