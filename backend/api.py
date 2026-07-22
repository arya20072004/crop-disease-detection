import io, json, pickle, base64, warnings, asyncio, uuid, os, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional
 
import cv2
import numpy as np
import pandas as pd
from PIL import Image
 
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
 
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.modules.leafseg import segment_leaf, SegmentationResult
from backend.modules.disease_localisation import localise_disease, LocalisationResult
from backend.modules.image_quality import check_image_quality, QualityResult
 
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
print(f"Loading models on {DEVICE}...")
 
# ── Load configs ──
with open(BASE_DIR / 'models/configs/class_names.json')   as f: CLASS_NAMES   = json.load(f)
with open(BASE_DIR / 'models/configs/lstm_config.json')   as f: lstm_cfg       = json.load(f)
with open(BASE_DIR / 'models/configs/fusion_config.json') as f: fusion_cfg     = json.load(f)
 
NUM_CLASSES      = len(CLASS_NAMES)
WEATHER_FEATURES = lstm_cfg['weather_features']
DISEASE_CLASSES  = lstm_cfg['disease_classes']
LOOKBACK         = lstm_cfg['lookback']
FORECAST         = lstm_cfg['forecast_steps']
N_DISEASES       = lstm_cfg['n_diseases']
N_FEATURES       = len(WEATHER_FEATURES)
CROP_TYPES       = fusion_cfg['crop_types']
GROWTH_STAGES    = fusion_cfg['growth_stages']
META_DIM         = fusion_cfg['meta_dim']
CNN_DIM          = 512
WEATHER_DIM      = 256
FUSION_INPUT_DIM = CNN_DIM + WEATHER_DIM + META_DIM
IMG_SIZE         = 380
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
 
val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
 
# ── CNN ──
class CropDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=38, dropout=0.4):
        super().__init__()
        backbone        = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        self.embedding  = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1792, 512), nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)
        self._feat_maps  = None
        self._feat_grads = None
        self.features[-1].register_forward_hook(
            lambda m, i, o: setattr(self, '_feat_maps', o))
        self.features[-1].register_full_backward_hook(
            lambda m, i, o: setattr(self, '_feat_grads', o[0]))
 
    def forward(self, x):
        x = self.features(x); x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(self.embedding(x))
 
    def get_embedding(self, x):
        with torch.no_grad():
            x = self.features(x); x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.embedding(x)
 
    def grad_cam(self, tensor, class_idx=None):
        """Concurrency-safe Grad-CAM. Does NOT touch requires_grad on any
        model parameter — uses torch.autograd.grad on a single retained
        feature tensor instead of .backward() through the whole graph."""
        self.eval()
        x = tensor.unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            # Forward through feature extractor, retain grad on feat map
            feat = self.features(x)
            feat.retain_grad()

            # Continue forward through avgpool → embedding → classifier
            pooled = self.avgpool(feat)
            flat   = torch.flatten(pooled, 1)
            logits = self.classifier(self.embedding(flat))

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            score = logits[0, class_idx]

            # Targeted gradient w.r.t. feature tensor only
            (grad,) = torch.autograd.grad(
                score, feat, retain_graph=False
            )

        # Grad-CAM: channel-wise mean of gradients as weights
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * feat.detach()).sum(dim=1).squeeze()
        cam     = torch.clamp(cam, min=0)
        cam     = cam / (cam.max() + 1e-8)
        conf    = logits.detach().softmax(dim=1)[0, class_idx].item()
        return cam.detach().cpu().numpy(), class_idx, conf
 
 
cnn_model = CropDiseaseClassifier(num_classes=NUM_CLASSES).to(DEVICE)
cnn_model.load_state_dict(
    torch.load(BASE_DIR / 'models/weights/best_model_v2.pth', map_location=DEVICE))
cnn_model.eval()
for p in cnn_model.parameters(): p.requires_grad = False
print("CNN loaded OK")
 
# ── LSTM ──
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * lstm_out).sum(dim=1), weights
 
class WeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 forecast_steps=7, n_diseases=26, dropout=0.3):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.n_diseases     = n_diseases
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_steps * n_diseases), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, w = self.attention(out)
        ctx    = self.dropout(ctx)
        pred   = self.fc(ctx).view(-1, self.forecast_steps, self.n_diseases)
        return pred, ctx, w
    def get_weather_context(self, x):
        with torch.no_grad():
            out, _ = self.lstm(x)
            ctx, _ = self.attention(out)
            return ctx
 
lstm_model = WeatherLSTM(
    input_dim=N_FEATURES, hidden_dim=lstm_cfg['hidden_dim'],
    num_layers=lstm_cfg['num_layers'], forecast_steps=FORECAST,
    n_diseases=N_DISEASES, dropout=lstm_cfg['dropout']
).to(DEVICE)
lstm_model.load_state_dict(
    torch.load(BASE_DIR / 'models/weights/best_lstm.pth', map_location=DEVICE))
lstm_model.eval()
for p in lstm_model.parameters(): p.requires_grad = False
print("LSTM loaded ✓")
 
# ── Fusion ──
class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=512, weather_dim=256, meta_dim=21,
                 n_diseases=26, attn_dim=256, n_heads=8, dropout=0.3):
        super().__init__()
        self.cnn_proj     = nn.Linear(cnn_dim, attn_dim)
        self.weather_proj = nn.Linear(weather_dim, attn_dim)
        self.cross_attn   = nn.MultiheadAttention(attn_dim, n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_norm    = nn.LayerNorm(attn_dim)
        self.self_attn    = nn.MultiheadAttention(attn_dim, n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.self_norm    = nn.LayerNorm(attn_dim)
        self.meta_proj    = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.fusion_head  = nn.Sequential(
            nn.Linear(attn_dim*2+64, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_diseases)
        )
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        cnn_f = x[:, :CNN_DIM]
        wx_f  = x[:, CNN_DIM:CNN_DIM+WEATHER_DIM]
        meta  = x[:, CNN_DIM+WEATHER_DIM:]
        cnn_q = self.cnn_proj(cnn_f).unsqueeze(1)
        wx_k  = self.weather_proj(wx_f).unsqueeze(1)
        co, _ = self.cross_attn(cnn_q, wx_k, wx_k)
        co    = self.attn_norm(co + cnn_q).squeeze(1)
        so, _ = self.self_attn(cnn_q, cnn_q, cnn_q)
        so    = self.self_norm(so + cnn_q).squeeze(1)
        mo    = self.meta_proj(meta)
        return self.fusion_head(self.dropout(
            torch.cat([co, so, mo], dim=1)))
 
    def predict_with_uncertainty(self, x, n_passes=50):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.train()
        with torch.no_grad():
            preds = torch.stack([
                torch.sigmoid(self.forward(x)) for _ in range(n_passes)])
        return preds.mean(dim=0), preds.std(dim=0)
 
 
fusion_model = CrossAttentionFusion(
    cnn_dim=CNN_DIM, weather_dim=WEATHER_DIM, meta_dim=META_DIM,
    n_diseases=N_DISEASES, attn_dim=256, n_heads=8, dropout=0.3
).to(DEVICE)
fusion_model.load_state_dict(
    torch.load(BASE_DIR / 'models/weights/best_fusion.pth', map_location=DEVICE))
fusion_model.eval()
print("Fusion model loaded ✓")
 
# ── Calibrators, XGBoost, scaler ──
with open(BASE_DIR / 'models/configs/calibrators.pkl', 'rb') as f:
    calibrators = pickle.load(f)
with open(BASE_DIR / 'models/configs/weather_scaler.pkl', 'rb') as f:
    weather_scaler = pickle.load(f)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(str(BASE_DIR / 'models/configs/yield_model.json'))
print("Calibrators, scaler, XGBoost loaded ✓")
 
# ── Weather data ──
weather_df = pd.read_csv(BASE_DIR / 'data/weather_features.csv',
                          index_col=0, parse_dates=True)
dcws_df    = pd.read_csv(BASE_DIR / 'data/dcws_scores.csv',
                          index_col=0, parse_dates=True)
print(f"Weather data loaded: {len(weather_df)} days ✓")
print("All models ready. Starting API...")

INTERVENTIONS = {
    'Tomato___Late_blight'       : {'fungicide': 'Mancozeb 75% WP @ 2.5 g/L',
                                    'frequency': 'Every 7 days',
                                    'timing'   : 'Apply at first sign or when risk > 60%',
                                    'cost_per_ha': 800, 'efficacy_pct': 70},
    'Potato___Late_blight'       : {'fungicide': 'Metalaxyl + Mancozeb @ 2.5 g/L',
                                    'frequency': 'Every 7–10 days',
                                    'timing'   : 'Preventive when risk > 50%',
                                    'cost_per_ha': 900, 'efficacy_pct': 75},
    'Tomato___Early_blight'      : {'fungicide': 'Chlorothalonil 75% WP @ 2 g/L',
                                    'frequency': 'Every 10 days',
                                    'timing'   : 'Begin when lower leaves show spots',
                                    'cost_per_ha': 600, 'efficacy_pct': 65},
    'Potato___Early_blight'      : {'fungicide': 'Mancozeb 75% WP @ 2 g/L',
                                    'frequency': 'Every 10–14 days',
                                    'timing'   : 'Start at tuber initiation stage',
                                    'cost_per_ha': 550, 'efficacy_pct': 60},
    'Apple___Apple_scab'         : {'fungicide': 'Captan 50% WP @ 2.5 g/L',
                                    'frequency': 'Every 7–10 days during wet weather',
                                    'timing'   : 'Critical: pink bud to petal fall',
                                    'cost_per_ha': 1200, 'efficacy_pct': 70},
    'Tomato___Bacterial_spot'    : {'fungicide': 'Copper oxychloride 50% WP @ 3 g/L',
                                    'frequency': 'Every 7 days',
                                    'timing'   : 'Apply before or at symptom onset',
                                    'cost_per_ha': 700, 'efficacy_pct': 55},
    'Squash___Powdery_mildew'    : {'fungicide': 'Sulphur 80% WP @ 3 g/L',
                                    'frequency': 'Every 10–14 days',
                                    'timing'   : 'At first sign of white powder',
                                    'cost_per_ha': 400, 'efficacy_pct': 75},
    'Corn_(maize)___Northern_Leaf_Blight': {
                                    'fungicide': 'Propiconazole 25% EC @ 1 mL/L',
                                    'frequency': 'Every 14 days',
                                    'timing'   : 'Apply at tasseling if risk is high',
                                    'cost_per_ha': 750, 'efficacy_pct': 65},
}
DEFAULT_INTERVENTION = {'fungicide': 'Consult local agronomist',
                        'frequency': 'Every 7–14 days',
                        'timing'   : 'At first confirmed symptom',
                        'cost_per_ha': 500, 'efficacy_pct': 50}
 
CROP_MAP = {
    'Apple': 'Apple', 'Blueberry': 'Blueberry',
    'Cherry_(including_sour)': 'Cherry', 'Corn_(maize)': 'Corn',
    'Grape': 'Grape', 'Orange': 'Orange', 'Peach': 'Peach',
    'Pepper,_bell': 'Pepper', 'Potato': 'Potato',
    'Raspberry': 'Raspberry', 'Soybean': 'Soybean',
    'Squash': 'Squash', 'Strawberry': 'Strawberry', 'Tomato': 'Tomato',
}
 
 
def encode_metadata(crop_type, growth_stage,
                    days_since_planting, days_to_harvest):
    vec    = np.zeros(META_DIM, dtype=np.float32)
    ct_idx = CROP_TYPES.index(crop_type) if crop_type in CROP_TYPES else 0
    gs_idx = GROWTH_STAGES.index(growth_stage) \
             if growth_stage in GROWTH_STAGES else 2
    vec[ct_idx]                          = 1.0
    vec[len(CROP_TYPES) + gs_idx]        = 1.0
    vec[len(CROP_TYPES)+len(GROWTH_STAGES)]   = min(days_since_planting/180, 1.0)
    vec[len(CROP_TYPES)+len(GROWTH_STAGES)+1] = min(days_to_harvest/90, 1.0)
    return vec
 
 
def numpy_to_b64(img_array: np.ndarray) -> str:
    """Encodes numpy image array to base64 PNG string."""
    _, buf = cv2.imencode('.png', img_array)
    return base64.b64encode(buf).decode('utf-8')
 
 
def preprocess_and_validate(img_array: np.ndarray) -> dict:
    """Pre-inference gate: reject bad images, then segment the leaf.
    Returns {'image_for_cnn': np.ndarray, 'diagnostics': dict}.
    Raises HTTPException(422) if the image fails quality checks."""
    quality = check_image_quality(img_array)
    if not quality.passed:
        raise HTTPException(
            status_code=422,
            detail={
                'error': 'image_quality_rejected',
                'reason': quality.retake_reason,
                'suggestions': quality.suggestions,
                'score': quality.score,
                'metrics': quality.metrics,
            }
        )

    seg = segment_leaf(img_array)

    diagnostics = {
        'quality_score': quality.score,
        'quality_warnings': [
            issue.message for issue in quality.issues if issue.level == 'soft'
        ],
        'segmentation_method': seg.method_used,
        'leaf_coverage': round(seg.leaf_coverage, 3),
        'segmentation_warning': seg.warning,
    }
    return {'image_for_cnn': seg.segmented_image, 'diagnostics': diagnostics}


def run_gradcam(image_array : np.ndarray,
                    class_idx   : int,
                    leaf_mask   : np.ndarray = None) -> dict:
        """
        Returns Grad-CAM overlay + disease localisation metrics.
        Now returns a dict instead of a bare base64 string.
        """
        resized   = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
        tensor    = val_transform(image=resized)['image']
        cam, _, _ = cnn_model.grad_cam(tensor, class_idx)   # cam is (h,w) float32

        # If no leaf mask provided (e.g. called from /explain without Part 1),
        # use a full-image mask as safe fallback.
        if leaf_mask is None:
            leaf_mask = np.full(resized.shape[:2], 255, dtype=np.uint8)
        else:
            leaf_mask = cv2.resize(leaf_mask, (IMG_SIZE, IMG_SIZE),
                                   interpolation=cv2.INTER_NEAREST)

        loc = localise_disease(
            img_rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
            cam_norm  = cam,
            leaf_mask = leaf_mask,
        )

        return {
            'gradcam'      : loc.annotated_b64,    # drop-in replacement
            'infected_pct' : round(loc.infected_pct, 2),
            'severity'     : loc.severity,
            'spot_count'   : len(loc.spots),
            'spots'        : [
                {
                    'id'           : i + 1,
                    'bbox'         : s.bbox,
                    'area_pct_leaf': round(s.area_pct_leaf, 2),
                    'centroid'     : s.centroid,
                }
                for i, s in enumerate(loc.spots)
            ],
            'warning'      : loc.warning,
        }

# ── Thread pool for blocking ML inference ──
# Keeps the asyncio event loop free; ML work runs in a dedicated OS thread.
_ML_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml_worker")
_executor = _ML_POOL

# ── In-memory job queue and tracking store ──
# job_store entry shape:
# {
#   'status': 'pending'|'processing'|'completed'|'failed',
#   'stage' : 'pending'|'cnn'|'lstm'|'fusion'|'done',
#   'result': { ... },
#   'error' : optional str
# }
job_queue = asyncio.Queue(maxsize=64)
job_store = {}

app = FastAPI(
    title       = "Crop Disease Early Warning System API",
    description = "Multimodal AI — CNN + Weather LSTM + Fusion",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

@app.middleware("http")
async def add_ngrok_header(request: Request, call_next):
    """Bypass the ngrok browser warning page for API calls from mobile apps."""
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response
 
 
# ── Request / Response models ──
class AnalyzeRequest(BaseModel):
    lat                 : float = 18.5204
    lon                 : float = 73.8567
    crop_type           : str   = "Tomato"
    growth_stage        : str   = "fruiting"
    days_since_planting : int   = 75
    days_to_harvest     : int   = 20
    area_ha             : float = 1.0
    market_price_per_kg : float = 25.0
    n_mc_passes         : int   = 30
 
class RecommendRequest(BaseModel):
    disease: str
    crop_type: str
    growth_stage: str
    risk_level: str
    area_ha: Optional[float] = 1.0
 
 
class ForecastRequest(BaseModel):
    lat      : float = 18.5204
    lon      : float = 73.8567
    top_n    : int   = 10
 

def _normalize_risk_level(value: str) -> str:
    normalized = (value or '').strip().lower()
    if normalized in {'low'}:
        return 'Low'
    if normalized in {'medium', 'moderate'}:
        return 'Medium'
    if normalized in {'high'}:
        return 'High'
    if normalized in {'critical', 'urgent'}:
        return 'Critical'
    return 'Medium'
 

def _normalize_disease_name(value: str) -> str:
    if not value:
        return ''
    key = value.replace('___', ' ').replace('_', ' ').strip()
    if key.lower().startswith('corn') and 'maize' not in key.lower():
        key = key.replace('corn', 'Corn')
    key = ' '.join(word.capitalize() for word in key.split())
    return key
 

def _adjust_dosage(base_dosage: str, area_ha: float) -> str:
    if area_ha is None or area_ha <= 0:
        return base_dosage
    extra_note = ''
    if 'g/l' in base_dosage.lower():
        try:
            amount = float(base_dosage.lower().split('g/l')[0].strip())
            approx_total = amount * 10.0 * area_ha
            extra_note = f" Approx. {approx_total:.1f} g per 10 L spray mix for {area_ha:.2f} ha."
        except Exception:
            extra_note = f" Apply at the same concentration over {area_ha:.2f} ha."
    elif 'ml/l' in base_dosage.lower():
        try:
            amount = float(base_dosage.lower().split('ml/l')[0].strip())
            approx_total = amount * 10.0 * area_ha
            extra_note = f" Approx. {approx_total:.1f} mL per 10 L spray mix for {area_ha:.2f} ha."
        except Exception:
            extra_note = f" Apply at the same concentration over {area_ha:.2f} ha."
    else:
        extra_note = f" Apply this recommendation across {area_ha:.2f} ha."
    return f"{base_dosage}{extra_note}"
 

RECOMMENDATION_KB = {
    'Tomato Late Blight': {
        'immediate_actions': [
            'Remove and destroy severely affected leaves immediately.',
            'Avoid overhead irrigation and keep foliage dry.',
            'Inspect neighbouring plants and isolate the affected area.',
        ],
        'fertilizer': {
            'name': 'Mancozeb 75% WP',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 7 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear gloves, spray in evening and avoid drift.',
        },
        'organic': {
            'name': 'Copper Oxychloride',
            'dosage': '3 g/L',
            'notes': 'Safe for organic farming and useful as a preventive barrier.',
        },
        'cultural_practices': [
            'Remove volunteer tomato plants and infected debris.',
            'Rotate with non-host crops for at least one season.',
            'Ensure adequate plant spacing and air flow.',
        ],
        'preventive_measures': [
            'Use certified disease-free seedling material.',
            'Maintain balanced irrigation and drainage.',
            'Continue monitoring daily during wet and humid weather.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 7 days before harvest.',
    },
    'Tomato Early Blight': {
        'immediate_actions': [
            'Prune infected lower leaves and dispose of them safely.',
            'Keep leaf surfaces dry and avoid afternoon wetting.',
            'Monitor the crop for new spots every 24 hours.',
        ],
        'fertilizer': {
            'name': 'Chlorothalonil 75% WP',
            'type': 'Fungicide',
            'dosage': '2 g/L water',
            'frequency': 'Every 10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Avoid working in wet foliage and wear protective clothing.',
        },
        'organic': {
            'name': 'Kaolin clay',
            'dosage': '40 g/L',
            'notes': 'Forms a protective film and is approved for organic systems.',
        },
        'cultural_practices': [
            'Mulch around plants to reduce soil splash.',
            'Remove diseased plant debris after harvest.',
            'Avoid excessive nitrogen fertilizer late in the season.',
        ],
        'preventive_measures': [
            'Plant resistant varieties where possible.',
            'Space plants to improve ventilation.',
            'Apply preventive sprays when humidity rises.',
        ],
        'expected_recovery_days': 12,
        'warning': 'Avoid applying in strong sunlight. PHI: 7 days.',
    },
    'Tomato Leaf Mold': {
        'immediate_actions': [
            'Remove and burn infected leaves immediately.',
            'Stop irrigation from above to reduce leaf wetness.',
            'Improve air circulation by pruning dense foliage.',
        ],
        'fertilizer': {
            'name': 'Mancozeb 75% WP',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 7 days',
            'application_method': 'Foliar spray',
            'precautions': 'Spray in the evening when dew is minimal.',
        },
        'organic': {
            'name': 'Sulphur 80% WP',
            'dosage': '3 g/L',
            'notes': 'Organic-safe and effective in high humidity.',
        },
        'cultural_practices': [
            'Avoid planting susceptible varieties in humid locations.',
            'Keep leaves dry by watering at the base.',
            'Prune lower branches to improve airflow.',
        ],
        'preventive_measures': [
            'Use drip irrigation and avoid overhead sprinklers.',
            'Inspect plants daily during humid periods.',
            'Destroy infected residues after harvest.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not spray during rain. PHI: 10 days before harvest.',
    },
    'Tomato Yellow Leaf Curl Virus': {
        'immediate_actions': [
            'Remove and destroy infected plants immediately.',
            'Control whitefly populations with yellow sticky traps.',
            'Avoid planting near old tomato fields or infected crops.',
        ],
        'fertilizer': {
            'name': 'Acetamiprid 20% SP',
            'type': 'Insecticide',
            'dosage': '0.4 g/L water',
            'frequency': 'Every 10 days',
            'application_method': 'Foliar spray targeting underside of leaves',
            'precautions': 'Wear protective gear and avoid drift to pollinators.',
        },
        'organic': {
            'name': 'Neem oil',
            'dosage': '5 mL/L',
            'notes': 'Use as a whitefly repellent in organic planting.',
        },
        'cultural_practices': [
            'Remove weed hosts and volunteer tomato plants.',
            'Maintain healthy crop nutrition to reduce stress.',
            'Use reflective mulch to deter insects.',
        ],
        'preventive_measures': [
            'Use virus-free transplants and resistant varieties.',
            'Install insect-proof netting if feasible.',
            'Monitor whitefly counts and act early.',
        ],
        'expected_recovery_days': 18,
        'warning': 'Do not apply during rain. PHI: 14 days before harvest.',
    },
    'Tomato Mosaic Virus': {
        'immediate_actions': [
            'Remove infected plants and sanitize tools between plots.',
            'Avoid working in the field when plants are wet.',
            'Use clean seed and avoid tobacco products in the field.',
        ],
        'fertilizer': {
            'name': 'Potassium phosphite 34% SL',
            'type': 'Plant strengthener',
            'dosage': '5 mL/L water',
            'frequency': 'Every 14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Apply in early morning or late evening.',
        },
        'organic': {
            'name': 'Seaweed extract',
            'dosage': '2 mL/L',
            'notes': 'Supports plant vigor and is safe for organic systems.',
        },
        'cultural_practices': [
            'Disinfect pruning tools and stakes after each use.',
            'Maintain good field hygiene and remove nearby weeds.',
            'Use resistant varieties if available.',
        ],
        'preventive_measures': [
            'Source certified virus-free seedlings.',
            'Avoid tobacco smoke and contaminated hands.',
            'Rotate crops and remove volunteer tomatoes.',
        ],
        'expected_recovery_days': 21,
        'warning': 'Viral infections cannot be cured; focus on prevention and plant health.',
    },
    'Potato Late Blight': {
        'immediate_actions': [
            'Remove affected leaves and stems immediately.',
            'Do not irrigate from above; keep canopy dry.',
            'Isolate the field if possible to prevent spread.',
        ],
        'fertilizer': {
            'name': 'Metalaxyl + Mancozeb',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 7–10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear gloves and avoid spraying in strong wind.',
        },
        'organic': {
            'name': 'Potassium bicarbonate',
            'dosage': '10 g/L',
            'notes': 'Provides an organic protective spray for tuber crops.',
        },
        'cultural_practices': [
            'Avoid planting tubers from infected fields.',
            'Crop rotate with non-hosts for at least two seasons.',
            'Remove cull piles and volunteer plants.',
        ],
        'preventive_measures': [
            'Use certified seed potatoes.',
            'Improve soil drainage and avoid excess nitrogen.',
            'Scout fields daily when weather is cool and wet.',
        ],
        'expected_recovery_days': 16,
        'warning': 'Do not apply during rain. PHI: 14 days before harvest.',
    },
    'Potato Early Blight': {
        'immediate_actions': [
            'Remove lower foliage showing spots and destroy it.',
            'Avoid wetting leaves when irrigating.',
            'Monitor nearby plants for spread.',
        ],
        'fertilizer': {
            'name': 'Mancozeb 75% WP',
            'type': 'Fungicide',
            'dosage': '2 g/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Do not enter treated fields until spray has dried.',
        },
        'organic': {
            'name': 'Neem oil',
            'dosage': '5 mL/L',
            'notes': 'Use as a low-impact spray for early-season disease suppression.',
        },
        'cultural_practices': [
            'Space rows to allow airflow around foliage.',
            'Remove plant debris at the end of season.',
            'Avoid high nitrogen levels late in the crop cycle.',
        ],
        'preventive_measures': [
            'Plant early to avoid high-risk weather.',
            'Use resistant potato varieties where available.',
            'Apply preventive protectant sprays under wet conditions.',
        ],
        'expected_recovery_days': 12,
        'warning': 'Avoid overhead irrigation after spray application.',
    },
    'Potato Black Scurf': {
        'immediate_actions': [
            'Remove infected tubers from the field and discard them.',
            'Avoid planting in cold, wet soils.',
            'Disinfect equipment and boots after handling affected plants.',
        ],
        'fertilizer': {
            'name': 'Thiophanate-methyl 70% WP',
            'type': 'Fungicide',
            'dosage': '2 g/L water',
            'frequency': 'At planting and repeat once if needed',
            'application_method': 'Tuber treatment or soil drench',
            'precautions': 'Avoid contact with skin; wear protective gloves.',
        },
        'organic': {
            'name': 'Trichoderma harzianum',
            'dosage': 'Based on supplier label',
            'notes': 'Biological seed treatment for soil-borne diseases.',
        },
        'cultural_practices': [
            'Use certified clean seed tubers.',
            'Avoid planting in fields with a history of scurf.',
            'Rotate with non-host crops for at least two years.',
        ],
        'preventive_measures': [
            'Keep seed tubers dry before planting.',
            'Store harvested tubers in well-ventilated conditions.',
            'Clean storage and handling areas frequently.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Treat seed tubers before planting. PHI: 21 days for harvested tubers.',
    },
    'Rice Blast': {
        'immediate_actions': [
            'Drain standing water and reduce canopy humidity.',
            'Remove severely infected tillers.',
            'Check adjacent fields for similar symptoms.',
        ],
        'fertilizer': {
            'name': 'Tricyclazole 75% WP',
            'type': 'Fungicide',
            'dosage': '2 g/L water',
            'frequency': 'Every 10 days when conditions are wet',
            'application_method': 'Foliar spray',
            'precautions': 'Spray when wind is calm and avoid drift.',
        },
        'organic': {
            'name': 'Bacillus subtilis',
            'dosage': 'Based on label',
            'notes': 'Biological option for rice blast prevention.',
        },
        'cultural_practices': [
            'Avoid dense nursery planting.',
            'Use recommended spacing and proper drainage.',
            'Remove infected stubble after harvest.',
        ],
        'preventive_measures': [
            'Plant resistant rice varieties.',
            'Avoid excessive nitrogen during wet weather.',
            'Manage irrigation to prevent prolonged leaf wetness.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 21 days before harvest.',
    },
    'Rice Brown Spot': {
        'immediate_actions': [
            'Remove severely infected leaves.',
            'Balance nitrogen and potassium application carefully.',
            'Monitor regularly and keep fields weed-free.',
        ],
        'fertilizer': {
            'name': 'Carbendazim 50% WP',
            'type': 'Fungicide',
            'dosage': '1.5 g/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Do not spray in direct sunlight.',
        },
        'organic': {
            'name': 'Neem cake soil application',
            'dosage': '5 kg/ha',
            'notes': 'Improves soil health and supports disease suppression.',
        },
        'cultural_practices': [
            'Avoid overfertilization with nitrogen.',
            'Maintain good water management and drainage.',
            'Remove infected debris after harvest.',
        ],
        'preventive_measures': [
            'Use balanced nutrition and avoid excess urea.',
            'Rotate rice with non-host crops.',
            'Keep the field free of straw and weeds.',
        ],
        'expected_recovery_days': 12,
        'warning': 'Avoid spray drift to neighbouring crops. PHI: 14 days.',
    },
    'Rice Bacterial Leaf Blight': {
        'immediate_actions': [
            'Remove severely infected plants and avoid flooding.',
            'Limit nitrogen application during the active phase.',
            'Check neighboring fields for early signs.',
        ],
        'fertilizer': {
            'name': 'Copper oxychloride 50% WP',
            'type': 'Bactericide',
            'dosage': '3 g/L water',
            'frequency': 'Every 7 days',
            'application_method': 'Foliar spray',
            'precautions': 'Spray in the evening and stay clear of water bodies.',
        },
        'organic': {
            'name': 'Potassium bicarbonate',
            'dosage': '10 g/L',
            'notes': 'Safe organic option for bacterial leaf blight.',
        },
        'cultural_practices': [
            'Use resistant varieties and healthy seedlings.',
            'Ensure good field drainage and reduce stagnant water.',
            'Avoid excessive nitrogen fertilizer.',
        ],
        'preventive_measures': [
            'Plant at recommended spacing.',
            'Monitor irrigation carefully.',
            'Use clean seed and avoid contaminated tools.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 21 days before harvest.',
    },
    'Rice Sheath Blight': {
        'immediate_actions': [
            'Reduce the height of standing water.',
            'Remove nearby weeds and volunteer grasses.',
            'Inspect the crop daily for spreading lesions.',
        ],
        'fertilizer': {
            'name': 'Azoxystrobin 18.2% SC',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray to lower canopy',
            'precautions': 'Apply in the evening and avoid drift.',
        },
        'organic': {
            'name': 'Trichoderma harzianum',
            'dosage': 'Based on supplier label',
            'notes': 'Biological soil treatment for sheath blight.',
        },
        'cultural_practices': [
            'Avoid high nitrogen rates and dense planting.',
            'Remove infected residue after harvest.',
            'Improve drainage and airflow in the canopy.',
        ],
        'preventive_measures': [
            'Rotate rice with non-host crops.',
            'Use well-decomposed organic matter.',
            'Monitor for disease during warm, humid weather.',
        ],
        'expected_recovery_days': 16,
        'warning': 'Do not apply during rain. PHI: 21 days before harvest.',
    },
    'Wheat Rust': {
        'immediate_actions': [
            'Scout fields immediately and remove hot spots.',
            'Prevent rust spread by avoiding overhead irrigation.',
            'Apply treatment before pustules are widespread.',
        ],
        'fertilizer': {
            'name': 'Propiconazole 25% EC',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear eye protection and avoid drift.',
        },
        'organic': {
            'name': 'Sulphur 80% WP',
            'dosage': '3 g/L',
            'notes': 'Useful for early preventive protection.',
        },
        'cultural_practices': [
            'Rotate with non-host crops.',
            'Remove volunteer wheat and grasses.',
            'Allow adequate row spacing for airflow.',
        ],
        'preventive_measures': [
            'Grow resistant cultivars.',
            'Avoid late nitrogen applications.',
            'Monitor weather forecasts and act quickly.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not spray during windy conditions. PHI: 21 days.',
    },
    'Wheat Powdery Mildew': {
        'immediate_actions': [
            'Remove infected plant parts and improve air circulation.',
            'Do not irrigate foliage from overhead.',
            'Treat early before more than 10% of leaves are infected.',
        ],
        'fertilizer': {
            'name': 'Sulphur 80% WP',
            'type': 'Fungicide',
            'dosage': '3 g/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Do not apply in hot midday sun.',
        },
        'organic': {
            'name': 'Potassium bicarbonate',
            'dosage': '10 g/L',
            'notes': 'Provides organic preventive coverage.',
        },
        'cultural_practices': [
            'Avoid dense planting and improve air flow.',
            'Remove alternate host plants if present.',
            'Maintain moderate irrigation levels.',
        ],
        'preventive_measures': [
            'Use resistant varieties when available.',
            'Avoid excessive nitrogen.',
            'Scout fields weekly in warm weather.',
        ],
        'expected_recovery_days': 10,
        'warning': 'Do not spray at midday. PHI: 14 days.',
    },
    'Wheat Loose Smut': {
        'immediate_actions': [
            'Remove and destroy smutted heads before seed shattering.',
            'Use clean seed and inspect seed lots carefully.',
            'Mark infected areas to avoid further spread.',
        ],
        'fertilizer': {
            'name': 'Seed treatment with Carboxin',
            'type': 'Fungicide',
            'dosage': 'As per label for seed treatment',
            'frequency': 'Single seed treatment before planting',
            'application_method': 'Seed treatment',
            'precautions': 'Follow label instructions and wear protection.',
        },
        'organic': {
            'name': 'Hot water seed treatment',
            'dosage': '50°C for 20 minutes',
            'notes': 'Effective organic seed treatment for loose smut.',
        },
        'cultural_practices': [
            'Use certified disease-free seed.',
            'Avoid saving seed from infected fields.',
            'Rotate to non-host crops.',
        ],
        'preventive_measures': [
            'Inspect seed batches before sowing.',
            'Manage volunteer wheat plants.',
            'Harvest clean seed separately.',
        ],
        'expected_recovery_days': 12,
        'warning': 'Treat seed before planting; no foliar cure exists.',
    },
    'Corn Gray Leaf Spot': {
        'immediate_actions': [
            'Remove heavily diseased leaves and improve air flow.',
            'Avoid planting in dense stands.',
            'Monitor fields for lesion spread.',
        ],
        'fertilizer': {
            'name': 'Azoxystrobin 18.2% SC',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Spray in the cooler part of the day.',
        },
        'organic': {
            'name': 'Bacillus subtilis',
            'dosage': 'Based on supplier label',
            'notes': 'Biological alternative for leaf spot control.',
        },
        'cultural_practices': [
            'Rotate with non-host crops.',
            'Use maize hybrids with improved tolerance.',
            'Avoid excess nitrogen under humid conditions.',
        ],
        'preventive_measures': [
            'Plant with adequate row spacing.',
            'Manage residue by tilling or removal.',
            'Monitor frequent rainfall events.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 21 days.',
    },
    'Corn Common Rust': {
        'immediate_actions': [
            'Inspect lower leaves and remove heavily infected plants.',
            'Avoid overhead irrigation if possible.',
            'Treat before spores spread to the upper canopy.',
        ],
        'fertilizer': {
            'name': 'Propiconazole 25% EC',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Avoid drift onto neighbouring orchards.',
        },
        'organic': {
            'name': 'Sulphur 80% WP',
            'dosage': '3 g/L',
            'notes': 'Useful as a preventive rust spray.',
        },
        'cultural_practices': [
            'Plant resistant hybrids when available.',
            'Do not over-irrigate the field.',
            'Remove volunteer maize plants.',
        ],
        'preventive_measures': [
            'Monitor spore traps if available.',
            'Space rows for better airflow.',
            'Limit nitrogen fertilizer when rust pressure is high.',
        ],
        'expected_recovery_days': 12,
        'warning': 'Do not spray in windy conditions. PHI: 21 days.',
    },
    'Corn Northern Corn Leaf Blight': {
        'immediate_actions': [
            'Remove and destroy heavily infected leaves.',
            'Improve ventilation by opening the canopy.',
            'Monitor for spread during humid weather.',
        ],
        'fertilizer': {
            'name': 'Propiconazole 25% EC',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear gloves and avoid inhalation of spray mist.',
        },
        'organic': {
            'name': 'Potassium bicarbonate',
            'dosage': '10 g/L',
            'notes': 'Provides low-risk coverage for leaf blight.',
        },
        'cultural_practices': [
            'Rotate maize with non-host crops.',
            'Remove volunteer grasses and maize.',
            'Avoid high-density planting.',
        ],
        'preventive_measures': [
            'Use resistant hybrids if available.',
            'Avoid irrigation at night.',
            'Scout fields regularly under humid conditions.',
        ],
        'expected_recovery_days': 16,
        'warning': 'Avoid spray drift. PHI: 21 days.',
    },
    'Cotton Boll Rot': {
        'immediate_actions': [
            'Remove and destroy infected squares and bolls.',
            'Avoid waterlogging and keep foliage dry.',
            'Monitor for spread after thunderstorms.',
        ],
        'fertilizer': {
            'name': 'Copper oxychloride 50% WP',
            'type': 'Fungicide',
            'dosage': '3 g/L water',
            'frequency': 'Every 10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear protective equipment while spraying.',
        },
        'organic': {
            'name': 'Neem oil',
            'dosage': '5 mL/L',
            'notes': 'Organic alternative for boll rot suppression.',
        },
        'cultural_practices': [
            'Ensure good drainage and avoid irrigation near bloom.',
            'Remove infected debris promptly.',
            'Keep the field weed-free.',
        ],
        'preventive_measures': [
            'Avoid high humidity around bolls.',
            'Use balanced nutrition and timely irrigation.',
            'Monitor fields after rain.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not spray during bloom if pollinators are active.',
    },
    'Cotton Leaf Curl': {
        'immediate_actions': [
            'Remove and destroy severely curled plants.',
            'Control whitefly vectors immediately.',
            'Prevent movement of infected plant material.',
        ],
        'fertilizer': {
            'name': 'Imidacloprid 17.8% SL',
            'type': 'Insecticide',
            'dosage': '0.2 mL/L water',
            'frequency': 'Every 7–10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Avoid spraying near flowering plants.',
        },
        'organic': {
            'name': 'Azadirachtin-based spray',
            'dosage': '2 mL/L',
            'notes': 'Useful as a low-impact whitefly management tool.',
        },
        'cultural_practices': [
            'Remove volunteer cotton and alternate host weeds.',
            'Use insect-proof nursery nets.',
            'Avoid planting near old cotton fields.',
        ],
        'preventive_measures': [
            'Use resistant varieties where available.',
            'Monitor whitefly populations regularly.',
            'Apply vector control early in the season.',
        ],
        'expected_recovery_days': 18,
        'warning': 'Do not apply during rainfall. PHI: 14 days.',
    },
    'Cotton Alternaria Blight': {
        'immediate_actions': [
            'Remove infected leaves and bolls as they appear.',
            'Avoid water stress and maintain good nutrition.',
            'Inspect adjacent fields and treat early.',
        ],
        'fertilizer': {
            'name': 'Mancozeb 75% WP',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Spray when air is calm and foliage is dry.',
        },
        'organic': {
            'name': 'Sulphur 80% WP',
            'dosage': '3 g/L',
            'notes': 'Organic-safe option for Alternaria management.',
        },
        'cultural_practices': [
            'Rotate with non-host crops.',
            'Maintain optimum plant nutrition.',
            'Remove crop residues after harvest.',
        ],
        'preventive_measures': [
            'Avoid excessive irrigation.',
            'Space plants to improve airflow.',
            'Scout regularly during humid weather.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Avoid spray drift. PHI: 14 days.',
    },
    'Soybean Frogeye Leaf Spot': {
        'immediate_actions': [
            'Scout leaves for lesions and remove badly affected plants.',
            'Improve air circulation by opening the canopy.',
            'Avoid working in the crop when foliage is wet.',
        ],
        'fertilizer': {
            'name': 'Chlorothalonil 75% WP',
            'type': 'Fungicide',
            'dosage': '2 g/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Apply in low wind conditions.',
        },
        'organic': {
            'name': 'Potassium bicarbonate',
            'dosage': '10 g/L',
            'notes': 'Supports disease suppression safely.',
        },
        'cultural_practices': [
            'Avoid overhead irrigation and keep foliage dry.',
            'Rotate with non-host crops.',
            'Plant at recommended row spacing.',
        ],
        'preventive_measures': [
            'Use certified seed.',
            'Monitor humidity and spray early.',
            'Avoid excess nitrogen fertilizer.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 21 days.',
    },
    'Soybean Sudden Death Syndrome': {
        'immediate_actions': [
            'Remove severely affected plants and destroy them.',
            'Avoid working in wet fields to reduce pathogen spread.',
            'Inspect root systems for discoloration.',
        ],
        'fertilizer': {
            'name': 'Prothioconazole 250 EC',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 10–14 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear protective gloves and avoid contaminating water sources.',
        },
        'organic': {
            'name': 'Biological seed treatment',
            'dosage': 'As per supplier label',
            'notes': 'Helps reduce soil-borne root pathogens.',
        },
        'cultural_practices': [
            'Ensure good drainage and avoid soil compaction.',
            'Rotate to non-host crops.',
            'Avoid planting in cool, wet soil.',
        ],
        'preventive_measures': [
            'Use fungicide-treated seed.',
            'Monitor soil moisture carefully.',
            'Avoid excessive nitrogen fertilizer.',
        ],
        'expected_recovery_days': 18,
        'warning': 'Do not apply during rain. PHI: 21 days.',
    },
    'Grape Black Rot': {
        'immediate_actions': [
            'Remove and destroy infected shoots and berries.',
            'Avoid overhead irrigation and keep trellis area clean.',
            'Inspect the canopy every 2–3 days.',
        ],
        'fertilizer': {
            'name': 'Mancozeb 75% WP',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 7–10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Wear protective clothing and avoid drift.',
        },
        'organic': {
            'name': 'Copper oxychloride',
            'dosage': '3 g/L',
            'notes': 'Approved for organic viticulture.',
        },
        'cultural_practices': [
            'Remove mummified berries from the vine.',
            'Improve air circulation within the canopy.',
            'Avoid planting vines too close together.',
        ],
        'preventive_measures': [
            'Use resistant grape varieties.',
            'Monitor humidity and disease pressure.',
            'Apply protectant sprays before wet weather.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 21 days.',
    },
    'Grape Powdery Mildew': {
        'immediate_actions': [
            'Remove infected leaves and thin canopy growth.',
            'Spray in the evening before dew appears.',
            'Inspect fruit clusters daily.',
        ],
        'fertilizer': {
            'name': 'Sulphur 80% WP',
            'type': 'Fungicide',
            'dosage': '3 g/L water',
            'frequency': 'Every 10 days',
            'application_method': 'Foliar spray',
            'precautions': 'Do not apply during strong winds.',
        },
        'organic': {
            'name': 'Potassium bicarbonate',
            'dosage': '10 g/L',
            'notes': 'Works well for powdery mildew control.',
        },
        'cultural_practices': [
            'Keep canopy open and remove excess shoots.',
            'Avoid excessive irrigation and humidity.',
            'Train vines for better airflow.',
        ],
        'preventive_measures': [
            'Apply protectant sprays early.',
            'Use tolerant varieties if available.',
            'Monitor weather and disease warnings.',
        ],
        'expected_recovery_days': 12,
        'warning': 'Do not spray in the hottest part of the day. PHI: 14 days.',
    },
    'Grape Downy Mildew': {
        'immediate_actions': [
            'Remove infected leaves and clusters.',
            'Avoid overhead irrigation and standing water.',
            'Treat early before the disease spreads.',
        ],
        'fertilizer': {
            'name': 'Copper oxychloride 50% WP',
            'type': 'Fungicide',
            'dosage': '3 g/L water',
            'frequency': 'Every 7 days',
            'application_method': 'Foliar spray',
            'precautions': 'Use in the cooler part of the day.',
        },
        'organic': {
            'name': 'Bordeaux mixture',
            'dosage': '100 g/10 L water',
            'notes': 'Traditional organic spray for downy mildew.',
        },
        'cultural_practices': [
            'Ensure good air movement through the canopy.',
            'Remove infected shoots promptly.',
            'Avoid over-fertilization with nitrogen.',
        ],
        'preventive_measures': [
            'Improve drainage and reduce humidity.',
            'Monitor during wet weather.',
            'Use resistant varieties if available.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during rain. PHI: 21 days.',
    },
    'Apple Scab': {
        'immediate_actions': [
            'Remove and destroy fallen leaves and infected fruit.',
            'Avoid overhead irrigation and wet foliage.',
            'Monitor trees every few days during wet weather.',
        ],
        'fertilizer': {
            'name': 'Captan 50% WP',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 7–10 days during wet weather',
            'application_method': 'Foliar spray',
            'precautions': 'Do not spray in strong wind.',
        },
        'organic': {
            'name': 'Sulphur 80% WP',
            'dosage': '3 g/L',
            'notes': 'Organic-safe repeat spray for apple scab.',
        },
        'cultural_practices': [
            'Rake and remove fallen leaves promptly.',
            'Prune for air circulation.',
            'Space trees appropriately in the orchard.',
        ],
        'preventive_measures': [
            'Plant resistant apple varieties.',
            'Keep the orchard floor clear of debris.',
            'Apply protective sprays before rain events.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not spray during rain. PHI: 7 days before harvest.',
    },
    'Apple Fire Blight': {
        'immediate_actions': [
            'Prune infected branches 20 cm below the lesion.',
            'Disinfect shears between cuts.',
            'Avoid working in the orchard when trees are wet.',
        ],
        'fertilizer': {
            'name': 'Streptomycin 25% WP',
            'type': 'Antibiotic spray',
            'dosage': '1 g/L water',
            'frequency': 'Repeat every 7 days during active blossom infection',
            'application_method': 'Blossom spray',
            'precautions': 'Follow local restrictions for antibiotic use.',
        },
        'organic': {
            'name': 'Copper oxychloride 50% WP',
            'dosage': '3 g/L water',
            'notes': 'Organic-approved option for blossom protection.',
        },
        'cultural_practices': [
            'Remove cankers in dry weather.',
            'Avoid excessive nitrogen late in season.',
            'Promote vigorous tree growth through good nutrition.',
        ],
        'preventive_measures': [
            'Use resistant rootstocks and scions.',
            'Avoid overhead irrigation during bloom.',
            'Monitor apple blossoms closely.',
        ],
        'expected_recovery_days': 18,
        'warning': 'Avoid pruning during wet weather. PHI: 14 days before harvest.',
    },
    'Apple Cedar Apple Rust': {
        'immediate_actions': [
            'Remove and destroy infected apple leaves and fruit.',
            'Keep cedar trees trimmed if they are nearby.',
            'Monitor for distinctive orange spores.',
        ],
        'fertilizer': {
            'name': 'Mancozeb 75% WP',
            'type': 'Fungicide',
            'dosage': '2.5 g/L water',
            'frequency': 'Every 7–10 days during wet periods',
            'application_method': 'Foliar spray',
            'precautions': 'Avoid spraying near pollinator habitat.',
        },
        'organic': {
            'name': 'Sulphur 80% WP',
            'dosage': '3 g/L',
            'notes': 'Can be used for organic rust suppression.',
        },
        'cultural_practices': [
            'Remove nearby cedar or alternate host plants.',
            'Improve orchard ventilation.',
            'Destroy infected leaves in fall.',
        ],
        'preventive_measures': [
            'Use rust-resistant apple cultivars.',
            'Avoid overhead irrigation.',
            'Apply preventive fungicides before wet weather.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not spray during rain. PHI: 14 days before harvest.',
    },
    'Pepper Bacterial Spot': {
        'immediate_actions': [
            'Remove infected leaves and fruit.',
            'Avoid working in the field when plants are wet.',
            'Apply protective sprays early in the infection cycle.',
        ],
        'fertilizer': {
            'name': 'Copper oxychloride 50% WP',
            'type': 'Bactericide',
            'dosage': '3 g/L water',
            'frequency': 'Every 7 days',
            'application_method': 'Foliar spray',
            'precautions': 'Avoid spraying during wind.',
        },
        'organic': {
            'name': 'Neem oil',
            'dosage': '5 mL/L',
            'notes': 'Organic-friendly bacterial suppression.',
        },
        'cultural_practices': [
            'Use drip irrigation and avoid overhead wetting.',
            'Remove diseased plants promptly.',
            'Maintain good field hygiene.',
        ],
        'preventive_measures': [
            'Use certified seed and transplants.',
            'Monitor humidity and leaf wetness.',
            'Rotate with non-host crops.',
        ],
        'expected_recovery_days': 14,
        'warning': 'Do not apply during irrigation. PHI: 10 days before harvest.',
    },
    'Pepper Phytophthora Blight': {
        'immediate_actions': [
            'Improve drainage immediately.',
            'Remove infected plants and affected soil.',
            'Avoid waterlogging in the bed.',
        ],
        'fertilizer': {
            'name': 'Mefenoxam 2% SL',
            'type': 'Fungicide',
            'dosage': '1 mL/L water',
            'frequency': 'Every 7–10 days',
            'application_method': 'Soil drench and foliar spray',
            'precautions': 'Use protective equipment and avoid water contamination.',
        },
        'organic': {
            'name': 'Potassium phosphite',
            'dosage': '5 mL/L',
            'notes': 'Helps strengthen plant defenses in organic systems.',
        },
        'cultural_practices': [
            'Ensure raised beds and well-draining soil.',
            'Avoid overhead irrigation.',
            'Remove infected plants and debris.',
        ],
        'preventive_measures': [
            'Use resistant varieties if available.',
            'Maintain soil pH between 6.0 and 6.5.',
            'Avoid continuous pepper planting in the same bed.',
        ],
        'expected_recovery_days': 16,
        'warning': 'Do not apply near open water bodies. PHI: 14 days before harvest.',
    },
}
 

def _recommendation_for(disease: str, crop_type: str, risk_level: str, area_ha: float):
    nice_disease = _normalize_disease_name(disease)
    record = RECOMMENDATION_KB.get(nice_disease)
    if record is None:
        # Try matching by partial disease name and crop.
        for key in RECOMMENDATION_KB.keys():
            if nice_disease.lower() in key.lower() or key.lower() in nice_disease.lower():
                record = RECOMMENDATION_KB[key]
                break
    if record is None:
        record = {
            'immediate_actions': [
                'Remove severely affected organs and improve crop hygiene.',
                'Monitor the field closely for 24 hours.',
                'Avoid overhead irrigation until symptoms subside.',
            ],
            'fertilizer': {
                'name': 'Recommended protective spray',
                'type': 'Fungicide / Pesticide',
                'dosage': 'Apply as per product label',
                'frequency': 'Every 7–10 days',
                'application_method': 'Foliar spray',
                'precautions': 'Wear protective clothing and gloves.',
            },
            'organic': {
                'name': 'Biological crop protector',
                'dosage': 'Apply as per label',
                'notes': 'Use organic preparations when available.',
            },
            'cultural_practices': [
                'Keep the crop well ventilated and weed-free.',
                'Avoid wetting the foliage during irrigation.',
                'Remove infected material from the field.',
            ],
            'preventive_measures': [
                'Use disease-free planting material.',
                'Maintain balanced nutrition and irrigation.',
                'Scout regularly for new symptoms.',
            ],
            'expected_recovery_days': 14,
            'warning': 'Follow label directions and avoid rain during application.',
        }
    severity = _normalize_risk_level(risk_level)
    fertilizer = record['fertilizer'].copy()
    fertilizer['dosage'] = _adjust_dosage(fertilizer['dosage'], area_ha)
    if severity == 'Low':
        fertilizer['notes'] = 'Use only as a preventive spray; follow cultural practices closely.'
        fertilizer = {
            'name': fertilizer['name'],
            'type': fertilizer['type'],
            'dosage': fertilizer['dosage'],
            'frequency': 'Use only if symptoms begin to appear',
            'application_method': fertilizer['application_method'],
            'precautions': fertilizer['precautions'],
        }
    elif severity == 'Critical':
        fertilizer['frequency'] = record['fertilizer']['frequency']
    else:
        fertilizer['frequency'] = record['fertilizer']['frequency']
    organic = record['organic'].copy()
    organic['dosage'] = _adjust_dosage(organic['dosage'], area_ha)
    actions = list(record['immediate_actions'])
    if severity == 'Low':
        actions.insert(0, 'Keep monitoring daily and remove any new affected tissue promptly.')
    elif severity == 'Medium':
        actions.insert(0, 'Apply a preventive spray and monitor every 48 hours.')
    elif severity == 'High':
        actions.insert(0, 'Treat immediately and inspect surrounding plants every day.')
    elif severity == 'Critical':
        actions.insert(0, 'Treat immediately with a certified chemical and follow strict sanitation.')
    schedule = ['Day 1', 'Day 7', 'Day 14']
    if severity == 'Low':
        schedule = ['Day 3', 'Day 10']
    elif severity == 'Critical':
        schedule = ['Day 1', 'Day 4', 'Day 8', 'Day 14']
    warning = record.get('warning')
    if severity == 'Critical' and record.get('warning'):
        warning = f"{record['warning']} Use maximum protective measures under critical risk." 
    return {
        'disease': nice_disease,
        'severity': severity,
        'immediate_actions': actions,
        'fertilizers': [fertilizer] if severity != 'Low' else [],
        'organic_alternatives': [organic],
        'cultural_practices': record['cultural_practices'],
        'preventive_measures': record['preventive_measures'],
        'expected_recovery_days': record['expected_recovery_days'],
        'follow_up_spray_schedule': schedule,
        'warning': warning or 'Follow label instructions and stay safe.',
    }
 

def _insert_supabase_recommendation(recommendation: dict):
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    if not supabase_url or not supabase_key:
        print('Supabase insert skipped; SUPABASE_URL or key is not configured.')
        return
    endpoint = supabase_url.rstrip('/') + '/rest/v1/recommendations'
    primary = recommendation['fertilizers'][0] if recommendation['fertilizers'] else recommendation['organic_alternatives'][0]
    notes = {
        'disease': recommendation['disease'],
        'severity': recommendation['severity'],
        'crop_type': recommendation.get('crop_type'),
        'growth_stage': recommendation.get('growth_stage'),
        'immediate_actions': recommendation['immediate_actions'],
        'fertilizers': recommendation['fertilizers'],
        'organic_alternatives': recommendation['organic_alternatives'],
        'cultural_practices': recommendation['cultural_practices'],
        'preventive_measures': recommendation['preventive_measures'],
        'warning': recommendation['warning'],
    }
    payload = {
        'disease': recommendation['disease'],
        'treatment': primary.get('name', ''),
        'dosage': primary.get('dosage', ''),
        'frequency': primary.get('frequency', ''),
        'notes': json.dumps(notes),
    }
    body = json.dumps([payload]).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Prefer': 'return=minimal',
    }
    try:
        req = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.status not in {200, 201, 204}:
                print(f'Supabase insert returned status {response.status}')
    except urllib.error.HTTPError as exc:
        print(f'Supabase insert failed: {exc.code} {exc.reason}')
    except Exception as exc:
        print(f'Supabase insert failed: {exc}')
 

# ── Routes ──
 
@app.post('/recommend')
async def recommend(request: RecommendRequest):
    disease = request.disease
    crop_type = request.crop_type
    growth_stage = request.growth_stage
    risk_level = _normalize_risk_level(request.risk_level)
    area_ha = request.area_ha or 1.0

    recommendation = _recommendation_for(disease, crop_type, risk_level, area_ha)
    recommendation['crop_type'] = crop_type
    recommendation['growth_stage'] = growth_stage
    recommendation['risk_level'] = risk_level

    try:
        _insert_supabase_recommendation(recommendation)
    except Exception as exc:
        print(f'Recommendation insert failed: {exc}')

    return recommendation
 
async def run_cnn(image_array: np.ndarray):
    """Stage 1 - fast CNN classification and embedding."""
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image'].unsqueeze(0).to(DEVICE)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        logits  = cnn_model(tensor)
        probs   = torch.softmax(logits, dim=1)[0]
        cnn_class = int(probs.argmax())
        cnn_conf  = float(probs[cnn_class])
        top5 = [
            {'class': CLASS_NAMES[i], 'prob': float(probs[i])}
            for i in probs.argsort(descending=True)[:5]
        ]
        embedding = cnn_model.get_embedding(tensor).squeeze(0).cpu().numpy()

    return {
        'cnn_class': cnn_class,
        'cnn_conf': cnn_conf,
        'cnn_top5': top5,
        'cnn_embedding': embedding,
    }


async def run_lstm():
    """Stage 2 - weather LSTM forecast context."""
    feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
    feat_norm = weather_scaler.transform(feat)
    wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        forecast_raw, wx_ctx, _ = lstm_model(wx_tensor)

    return {
        'lstm_forecast': forecast_raw[0].cpu().numpy(),
        'lstm_ctx': wx_ctx.cpu().numpy(),
    }


def run_fusion(cnn_embedding, lstm_ctx, meta_vector, n_mc_passes=30):
    """Stage 3 - fusion inference + calibration + economics."""
    fusion_input = torch.tensor(
        np.concatenate([cnn_embedding.reshape(1, -1), lstm_ctx.reshape(1, -1), meta_vector.reshape(1, -1)], axis=1),
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        mean_probs, std_probs = fusion_model.predict_with_uncertainty(
            fusion_input, n_passes=n_mc_passes
        )

    mean_probs = mean_probs[0].cpu().numpy()
    std_probs  = std_probs[0].cpu().numpy()

    cal_probs = np.array([
        calibrators[d].predict([mean_probs[i]])[0]
        for i, d in enumerate(DISEASE_CLASSES)
    ])

    top_idx     = int(cal_probs.argmax())
    top_disease = DISEASE_CLASSES[top_idx]
    top_risk    = float(cal_probs[top_idx])
    top_unc     = float(std_probs[top_idx])

    return {
        'fusion_raw': mean_probs.tolist(),
        'fusion_uncertainty': std_probs.tolist(),
        'fusion_calibrated': cal_probs.tolist(),
        'top_disease': top_disease,
        'top_risk': top_risk,
        'top_uncertainty': top_unc,
        'top_idx': top_idx,
    }


def assemble_forecast_and_yield(cnn_info, fusion_info, meta):
    """Helper: build final forecast/yield/intervention payload."""
    forecast_raw = fusion_info.get('forecast_raw')
    # Use LSTM forecast from job lsm stage if provided later
    # (In our working model, we can compute from run_lstm stage separately.)

    top_idx = fusion_info['top_idx']
    top_disease = fusion_info['top_disease']
    top_risk = fusion_info['top_risk']

    severity = cnn_info['cnn_conf']
    if CLASS_NAMES[cnn_info['cnn_class']] != top_disease:
        severity = 0.5

    xgb_feat = np.array([[
        top_risk,
        severity,
        GROWTH_STAGES.index(meta['growth_stage']),
        meta['days_to_harvest'],
        0, 0, top_idx
    ]])

    yield_loss = float(xgb_model.predict(xgb_feat)[0])

    rec = INTERVENTIONS.get(top_disease, DEFAULT_INTERVENTION)
    fin_loss = yield_loss * 20000 * meta['area_ha'] * meta['market_price_per_kg']
    treat_cost = rec['cost_per_ha'] * meta['area_ha']
    saved_val = fin_loss * (rec['efficacy_pct'] / 100)

    if top_risk > 0.6:
        urgency = 'TREAT TODAY'
    elif top_risk > 0.3:
        urgency = 'MONITOR — treat within 3 days if worsening'
    else:
        urgency = 'LOW RISK — continue monitoring'

    return {
        'yield': {
            'loss_pct': yield_loss,
            'loss_kg': yield_loss * 20000 * meta['area_ha'],
            'financial_loss': fin_loss,
            'treatment_cost': treat_cost,
            'saved_value': saved_val,
            'roi': (saved_val - treat_cost) / (treat_cost + 1e-8),
        },
        'intervention': {
            'urgency': urgency,
            'fungicide': rec['fungicide'],
            'frequency': rec['frequency'],
            'timing': rec['timing'],
        },
    }


async def process_job(job_id: str):
    """Background job worker executes pipeline stages and updates job_store."""
    job = job_store.get(job_id)
    if job is None:
        return

    try:
        job['status'] = 'processing'
        job['stage']  = 'cnn'

        cnn_info = await asyncio.get_event_loop().run_in_executor(
            _ML_POOL, run_cnn, job['input_image_np']
        )
        job['result']['cnn'] = {
            'detected': CLASS_NAMES[cnn_info['cnn_class']],
            'confidence': cnn_info['cnn_conf'],
            'top5': cnn_info['cnn_top5'],
        }
        job['result']['cnn_class'] = cnn_info['cnn_class']
        job['result']['cnn_conf'] = cnn_info['cnn_conf']

        job['stage'] = 'lstm'
        lstm_info = await run_lstm()
        job['result']['lstm'] = {
            'forecast': lstm_info['lstm_forecast'].tolist(),
            'context': lstm_info['lstm_ctx'].tolist(),
        }

        job['stage'] = 'fusion'
        fusion_info = await asyncio.get_event_loop().run_in_executor(
            _ML_POOL, run_fusion,
            cnn_info['cnn_embedding'], lstm_info['lstm_ctx'],
            encode_metadata(
                job['meta']['crop_type'], job['meta']['growth_stage'],
                job['meta']['days_since_planting'], job['meta']['days_to_harvest']
            ), job['meta'].get('n_mc_passes', 30)
        )

        job['result']['fusion'] = {
            'top_disease': fusion_info['top_disease'],
            'risk_score': fusion_info['top_risk'],
            'uncertainty': fusion_info['top_uncertainty'],
            'top5': [
                {
                    'disease': DISEASE_CLASSES[i],
                    'probability': float(fusion_info['fusion_calibrated'][i]),
                    'uncertainty': float(fusion_info['fusion_uncertainty'][i]),
                }
                for i in np.argsort(fusion_info['fusion_calibrated'])[::-1][:5]
            ],
        }

        job['stage'] = 'done'
        extras = assemble_forecast_and_yield(cnn_info, fusion_info, job['meta'])
        job['result'].update(extras)

        job['status'] = 'completed'
    except Exception as exc:
        job['status'] = 'failed'
        job['stage']  = 'failed'
        job['error']  = f"{type(exc).__name__}: {exc}"
        print(f"Job {job_id} failed:", exc)


async def job_worker(worker_id: int):
    """Continuously pull jobs from queue for async execution."""
    while True:
        job_id = await job_queue.get()
        try:
            await process_job(job_id)
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def _startup():
    """Warm up models and start worker tasks at app startup."""
    loop = asyncio.get_event_loop()
    def _warm():
        try:
            dummy_feat  = np.zeros((LOOKBACK, N_FEATURES), dtype=np.float32)
            dummy_norm  = weather_scaler.transform(dummy_feat)
            dummy_wx    = torch.tensor(dummy_norm).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                lstm_model(dummy_wx)
            print("Warmup pass complete ✓")
        except Exception as e:
            print(f"Warmup warning (non-fatal): {e}")
    await loop.run_in_executor(_ML_POOL, _warm)

    app.state.job_workers = [
        asyncio.create_task(job_worker(i)) for i in range(2)
    ]


@app.on_event("shutdown")
async def _shutdown():
    workers = getattr(app.state, 'job_workers', [])
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

def _run_analyze_sync(
    img_bytes, crop_type, growth_stage,
    days_since_planting, days_to_harvest,
    area_ha, market_price_per_kg,
    n_mc_passes, include_gradcam, lat, lon, quality
):
    """Synchronous full pipeline: CNN → LSTM → Fusion → Yield. 
    Runs in a ThreadPoolExecutor so the event loop stays free."""
    # Decode image
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np    = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Stage 0 — Leaf Segmentation (CV preprocessing)
    seg_result = segment_leaf(img_np)           # runs on original resolution
    if seg_result.warning:
        print(f"[Segmentation] {seg_result.warning}")

    # Stage 0b — Crop to leaf bounding box so CNN sees only leaf pixels
    x, y, w, h = seg_result.bbox
    _valid_crop    = w > 10 and h > 10
    leaf_img       = seg_result.segmented_image[y:y+h, x:x+w] if _valid_crop else seg_result.segmented_image
    leaf_mask_crop = seg_result.mask[y:y+h, x:x+w]             if _valid_crop else seg_result.mask

    # Stage 1 — CNN
    resized = cv2.resize(leaf_img, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image'].unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = cnn_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        cnn_class = int(probs.argmax())
        cnn_conf  = float(probs[cnn_class])
        top5 = [
            {'class': CLASS_NAMES[i], 'prob': float(probs[i])}
            for i in probs.argsort(descending=True)[:5]
        ]
        embedding = cnn_model.get_embedding(tensor).squeeze(0).cpu().numpy()

    # Stage 2 — LSTM
    feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
    feat_norm = weather_scaler.transform(feat)
    wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        forecast_raw, wx_ctx, _ = lstm_model(wx_tensor)
    lstm_forecast = forecast_raw[0].cpu().numpy()
    lstm_ctx      = wx_ctx.cpu().numpy()

    # Stage 3 — Fusion
    meta_vec     = encode_metadata(crop_type, growth_stage,
                                   days_since_planting, days_to_harvest)
    fusion_input = torch.tensor(
        np.concatenate([embedding.reshape(1, -1),
                        lstm_ctx.reshape(1, -1),
                        meta_vec.reshape(1, -1)], axis=1),
        dtype=torch.float32
    ).to(DEVICE)
    mean_probs, std_probs = fusion_model.predict_with_uncertainty(
        fusion_input, n_passes=n_mc_passes)
    mean_probs = mean_probs[0].cpu().numpy()
    std_probs  = std_probs[0].cpu().numpy()
    cal_probs  = np.array([
        calibrators[d].predict([mean_probs[i]])[0]
        for i, d in enumerate(DISEASE_CLASSES)
    ])
    top_idx     = int(cal_probs.argmax())
    top_disease = DISEASE_CLASSES[top_idx]
    top_risk    = float(cal_probs[top_idx])
    top_unc     = float(std_probs[top_idx])

    # Stage 4 — Yield & intervention
    meta = {
        'crop_type': crop_type, 'growth_stage': growth_stage,
        'days_since_planting': days_since_planting,
        'days_to_harvest': days_to_harvest,
        'area_ha': area_ha, 'market_price_per_kg': market_price_per_kg,
    }
    cnn_info    = {'cnn_class': cnn_class, 'cnn_conf': cnn_conf}
    fusion_info = {
        'top_idx': top_idx, 'top_disease': top_disease,
        'top_risk': top_risk, 'top_uncertainty': top_unc,
        'fusion_calibrated': cal_probs.tolist(),
        'fusion_uncertainty': std_probs.tolist(),
        'forecast_raw': lstm_forecast.tolist(),
    }
    extras = assemble_forecast_and_yield(cnn_info, fusion_info, meta)

    # Grad-CAM (opt-in)
    if include_gradcam:
        with torch.enable_grad():
            gradcam_result = run_gradcam(
                leaf_img,
                cnn_class,
                leaf_mask = leaf_mask_crop,
            )
    else:
        gradcam_result = None

    return {
        'cnn': {
            'detected': CLASS_NAMES[cnn_class],
            'confidence': cnn_conf,
            'top5': top5,
        },
        'fusion': {
            'top_disease': top_disease,
            'risk_score': top_risk,
            'uncertainty': top_unc,
            'top5': [
                {
                    'disease': DISEASE_CLASSES[i],
                    'probability': float(cal_probs[i]),
                    'uncertainty': float(std_probs[i]),
                }
                for i in np.argsort(cal_probs)[::-1][:5]
            ],
        },
        **extras,
        'gradcam': gradcam_result,
        'segmentation': {
            'method'        : seg_result.method_used,
            'leaf_coverage' : round(seg_result.leaf_coverage, 3),
            'warning'       : seg_result.warning,
            'bbox'          : seg_result.bbox,
            'bbox_crop'     : _valid_crop,
        },
        'quality': {
            'score'    : quality.score,
            'warnings' : [i.message for i in quality.issues if i.level == 'soft'],
            'metrics'  : quality.metrics,
        },
        'location': {'lat': lat, 'lon': lon},
        'timestamp': datetime.now().isoformat(),
    }

@app.post("/analyze")
def analyze(
    file                : UploadFile = File(...),
    lat                 : float = 18.5204,
    lon                 : float = 73.8567,
    crop_type           : str   = "Tomato",
    growth_stage        : str   = "fruiting",
    days_since_planting : int   = 75,
    days_to_harvest     : int   = 20,
    area_ha             : float = 1.0,
    market_price_per_kg : float = 25.0,
    # Reduced from 30 → 10.  Still gives reliable uncertainty bands,
    # cuts MC-dropout time by ~3×.  Flutter can pass 30 explicitly if needed.
    n_mc_passes         : int   = 10,
    # Set to False by default — Flutter requests Grad-CAM only when the
    # user taps "Show Heatmap", not on every analysis.
    include_gradcam     : bool  = False,
):
    # ── Validate content type ──
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Accepted: {', '.join(sorted(allowed_types))}"
        )

    # ── Read bytes (sync — not async) ──
    img_bytes = file.file.read()

    # ── File-size guard (15 MB) ──
    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum upload size is 15 MB."
        )

    # ── Decode to numpy RGB array ──
    try:
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(pil_img)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Please upload a valid JPEG, PNG, or WebP file."
        )

    # ── Pre-inference quality gate + segmentation (FIX 3) ──
    pv = preprocess_and_validate(img_array)
    segmented_image = pv['image_for_cnn']
    diagnostics     = pv['diagnostics']

    # ── CNN inference on the segmented image ──
    resized = cv2.resize(segmented_image, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image'].unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = cnn_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        cnn_class = int(probs.argmax())
        cnn_conf  = float(probs[cnn_class])
        top5 = [
            {'class': CLASS_NAMES[i], 'prob': float(probs[i])}
            for i in probs.argsort(descending=True)[:5]
        ]
        embedding = cnn_model.get_embedding(tensor).squeeze(0).cpu().numpy()

    # ── Confidence threshold check ──
    low_confidence = cnn_conf < 0.45

    # ── LSTM ──
    feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
    feat_norm = weather_scaler.transform(feat)
    wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        forecast_raw, wx_ctx, _ = lstm_model(wx_tensor)
    lstm_forecast = forecast_raw[0].cpu().numpy()
    lstm_ctx      = wx_ctx.cpu().numpy()

    # ── Fusion ──
    meta_vec     = encode_metadata(crop_type, growth_stage,
                                   days_since_planting, days_to_harvest)
    fusion_input = torch.tensor(
        np.concatenate([embedding.reshape(1, -1),
                        lstm_ctx.reshape(1, -1),
                        meta_vec.reshape(1, -1)], axis=1),
        dtype=torch.float32
    ).to(DEVICE)
    mean_probs, std_probs = fusion_model.predict_with_uncertainty(
        fusion_input, n_passes=n_mc_passes)
    mean_probs = mean_probs[0].cpu().numpy()
    std_probs  = std_probs[0].cpu().numpy()
    cal_probs  = np.array([
        calibrators[d].predict([mean_probs[i]])[0]
        for i, d in enumerate(DISEASE_CLASSES)
    ])
    top_idx     = int(cal_probs.argmax())
    top_disease = DISEASE_CLASSES[top_idx]
    top_risk    = float(cal_probs[top_idx])
    top_unc     = float(std_probs[top_idx])

    # ── Yield & intervention ──
    meta = {
        'crop_type': crop_type, 'growth_stage': growth_stage,
        'days_since_planting': days_since_planting,
        'days_to_harvest': days_to_harvest,
        'area_ha': area_ha, 'market_price_per_kg': market_price_per_kg,
    }
    cnn_info    = {'cnn_class': cnn_class, 'cnn_conf': cnn_conf}
    fusion_info = {
        'top_idx': top_idx, 'top_disease': top_disease,
        'top_risk': top_risk, 'top_uncertainty': top_unc,
        'fusion_calibrated': cal_probs.tolist(),
        'fusion_uncertainty': std_probs.tolist(),
        'forecast_raw': lstm_forecast.tolist(),
    }
    extras = assemble_forecast_and_yield(cnn_info, fusion_info, meta)

    # ── Grad-CAM on the SEGMENTED image (visual consistency) ──
    if include_gradcam:
        with torch.enable_grad():
            gradcam_result = run_gradcam(segmented_image, cnn_class)
    else:
        gradcam_result = None

    return {
        'cnn': {
            'detected': CLASS_NAMES[cnn_class],
            'confidence': cnn_conf,
            'low_confidence': low_confidence,
            'top5': top5,
        },
        'fusion': {
            'top_disease': top_disease,
            'risk_score': top_risk,
            'uncertainty': top_unc,
            'top5': [
                {
                    'disease': DISEASE_CLASSES[i],
                    'probability': float(cal_probs[i]),
                    'uncertainty': float(std_probs[i]),
                }
                for i in np.argsort(cal_probs)[::-1][:5]
            ],
        },
        **extras,
        'gradcam': gradcam_result,
        'image_diagnostics': diagnostics,
        'location': {'lat': lat, 'lon': lon},
        'timestamp': datetime.now().isoformat(),
    }


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job ID not found')
    response = {
        'job_id': job_id,
        'status': job['status'],
        'stage': job.get('stage', 'unknown'),
        'result': job.get('result', {}),
    }
    if job.get('error'):
        response['error'] = job['error']
    return response


@app.get("/explain/{job_id}")
async def explain(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job ID not found')
    if job['status'] != 'completed':
        raise HTTPException(status_code=400,
                            detail='Job not completed yet: explain available only after completion')

    cnn_class_name = job['result']['cnn']['detected']
    cnn_class_idx  = CLASS_NAMES.index(cnn_class_name)

    img_np = job.get('input_image_np')
    if img_np is None:
        raise HTTPException(status_code=410,
                            detail='Input image not available for explanation')

    gradcam_b64 = await asyncio.get_event_loop().run_in_executor(
        _ML_POOL, run_gradcam, img_np, cnn_class_idx
    )

    return {
        'job_id': job_id,
        'gradcam': gradcam_b64,
        'target_class': cnn_class_name,
    }


@app.get("/config")
def get_config():
    return {
        "crop_types"    : CROP_TYPES,
        "growth_stages" : GROWTH_STAGES,
        "disease_classes": DISEASE_CLASSES,
        "class_names"   : CLASS_NAMES,
    }
 
 
@app.get("/forecast")
async def get_forecast(lat: float = 18.5204, lon: float = 73.8567,
                       top_n: int = 10):
    """7-day disease risk forecast for a GPS location."""
    loop = asyncio.get_event_loop()

    def _run():
        feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
        feat_norm = weather_scaler.transform(feat)
        wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
        lstm_model.eval()
        with torch.no_grad():
            forecast_raw, _, _ = lstm_model(wx_tensor)
        forecast_np  = forecast_raw[0].cpu().numpy()
        peak_risk_np = forecast_np.max(axis=0)
        ranked_idx   = peak_risk_np.argsort()[::-1][:top_n]
        forecast_dates = [
            (datetime.now() + timedelta(days=i+1)).strftime('%a %d %b')
            for i in range(FORECAST)
        ]
        return {
            'location' : {'lat': lat, 'lon': lon},
            'dates'    : forecast_dates,
            'diseases' : [
                {
                    'disease'  : DISEASE_CLASSES[i],
                    'peak_risk': float(peak_risk_np[i]),
                    'daily'    : [float(forecast_np[d, i]) for d in range(FORECAST)],
                    'level'    : 'HIGH' if peak_risk_np[i] > 0.6
                                 else 'MODERATE' if peak_risk_np[i] > 0.3
                                 else 'LOW',
                }
                for i in ranked_idx
            ],
            'timestamp': datetime.now().isoformat(),
        }

    try:
        return await loop.run_in_executor(_ML_POOL, _run)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Forecast failed: {type(e).__name__}: {e}")
 
 
@app.get("/historical")
async def get_historical(disease: str, days_back: int = 365):
    """Historical DCWS risk scores for a disease."""
    if disease not in dcws_df.columns:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Disease '{disease}' not found. "
                f"Available (first 5): {list(dcws_df.columns[:5])}"
            )
        )
    loop   = asyncio.get_event_loop()
    series = await loop.run_in_executor(
        None, lambda: dcws_df[disease].tail(days_back))
    return {

        'disease': disease,
        'dates'  : series.index.strftime('%Y-%m-%d').tolist(),
        'scores' : series.round(4).tolist(),
        'mean'   : float(series.mean()),
        'max'    : float(series.max()),
    }

@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.get("/compare")
async def compare_crops(
    lat          : float = 18.5204,
    lon          : float = 73.8567,
    growth_stage : str   = "fruiting",
    days_to_harvest: int = 30,
):
    """Compares risk across all crop types for current weather."""
    loop = asyncio.get_event_loop()

    def _run():
        feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
        feat_norm = weather_scaler.transform(feat)
        wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
        lstm_model.eval()
        with torch.no_grad():
            _, wx_ctx, _ = lstm_model(wx_tensor)
        wx_ctx_np = wx_ctx.cpu().numpy()

        results = {}
        for crop in CROP_TYPES:
            meta = encode_metadata(crop, growth_stage, 60, days_to_harvest
                                   ).reshape(1, -1)
            cnn_zero     = np.zeros((1, CNN_DIM), dtype=np.float32)
            fusion_input = torch.tensor(
                np.concatenate([cnn_zero, wx_ctx_np, meta], axis=1),
                dtype=torch.float32
            ).to(DEVICE)
            fusion_model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(
                    fusion_model(fusion_input))[0].cpu().numpy()
            cal_probs = np.array([
                calibrators[d].predict([probs[i]])[0]
                for i, d in enumerate(DISEASE_CLASSES)
            ])
            results[crop] = {
                'max_risk'   : float(cal_probs.max()),
                'top_disease': DISEASE_CLASSES[int(cal_probs.argmax())],
                'risk_level' : 'HIGH' if cal_probs.max() > 0.6
                               else 'MODERATE' if cal_probs.max() > 0.3
                               else 'LOW',
            }
        return {'crops': results, 'timestamp': datetime.now().isoformat()}

    try:
        return await loop.run_in_executor(_ML_POOL, _run)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Comparison failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",       # bind to all interfaces so ngrok can reach it
        port=8000,
        reload=False,          # keep False in production
        workers=1,             # single worker keeps ML models in one process
        timeout_keep_alive=65, # longer than Flutter's 30 s connect timeout
        log_level="info",
    )