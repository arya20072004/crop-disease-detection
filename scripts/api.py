import io, json, pickle, base64, warnings
from pathlib import Path
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
 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
print(f"Loading models on {DEVICE}...")
 
# ── Load configs ──
with open(BASE_DIR / 'class_names.json')   as f: CLASS_NAMES   = json.load(f)
with open(BASE_DIR / 'lstm_config.json')   as f: lstm_cfg       = json.load(f)
with open(BASE_DIR / 'fusion_config.json') as f: fusion_cfg     = json.load(f)
 
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
        """Grad-CAM using manual hook-based gradient computation."""
        self.eval()

        # Temporarily re-enable grad on ALL feature parameters
        for p in self.parameters():
            p.requires_grad = True

        x = tensor.unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            # Fresh forward pass with grad tracking
            feat = self.features(x)
            pool = self.avgpool(feat)
            flat = torch.flatten(pool, 1)
            emb  = self.embedding(flat)
            logits = self.classifier(emb)

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            # Backward on target class score
            score = logits[0, class_idx]
            self.zero_grad()
            score.backward()

        # Grad-CAM: global average pool the gradients
        grads   = feat.grad if feat.grad is not None else self._feat_grads
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * feat).sum(dim=1).squeeze()
        cam     = torch.clamp(cam, min=0)
        cam     = cam / (cam.max() + 1e-8)
        conf    = logits.softmax(dim=1)[0, class_idx].item()

        # Re-freeze everything
        for p in self.parameters():
            p.requires_grad = False

        return cam.detach().cpu().numpy(), class_idx, conf
 
 
cnn_model = CropDiseaseClassifier(num_classes=NUM_CLASSES).to(DEVICE)
cnn_model.load_state_dict(
    torch.load(BASE_DIR / 'best_model_v2.pth', map_location=DEVICE))
cnn_model.eval()
for p in cnn_model.parameters(): p.requires_grad = False
print("CNN loaded ✓")
 
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
    torch.load(BASE_DIR / 'best_lstm.pth', map_location=DEVICE))
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
    torch.load(BASE_DIR / 'best_fusion.pth', map_location=DEVICE))
fusion_model.eval()
print("Fusion model loaded ✓")
 
# ── Calibrators, XGBoost, scaler ──
with open(BASE_DIR / 'calibrators.pkl', 'rb') as f:
    calibrators = pickle.load(f)
with open(BASE_DIR / 'weather_scaler.pkl', 'rb') as f:
    weather_scaler = pickle.load(f)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(str(BASE_DIR / 'yield_model.json'))
print("Calibrators, scaler, XGBoost loaded ✓")
 
# ── Weather data ──
weather_df = pd.read_csv(BASE_DIR / 'weather_features.csv',
                          index_col=0, parse_dates=True)
dcws_df    = pd.read_csv(BASE_DIR / 'dcws_scores.csv',
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
 
 
def run_gradcam(image_array: np.ndarray, class_idx: int) -> str:
    """Returns base64-encoded Grad-CAM overlay."""
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image']
    cam, _, _ = cnn_model.grad_cam(tensor, class_idx)
    cam_up    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap   = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    overlay   = (0.5 * resized + 0.5 * heatmap).astype(np.uint8)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return numpy_to_b64(overlay_rgb)

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
)
 
 
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
 
 
class ForecastRequest(BaseModel):
    lat      : float = 18.5204
    lon      : float = 73.8567
    top_n    : int   = 10
 
 
# ── Routes ──
 
@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "device"   : str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }
 
 
@app.get("/config")
def get_config():
    return {
        "crop_types"    : CROP_TYPES,
        "growth_stages" : GROWTH_STAGES,
        "disease_classes": DISEASE_CLASSES,
        "class_names"   : CLASS_NAMES,
    }
 
 
@app.post("/analyze")
async def analyze(
    file                : UploadFile = File(...),
    lat                 : float = 18.5204,
    lon                 : float = 73.8567,
    crop_type           : str   = "Tomato",
    growth_stage        : str   = "fruiting",
    days_since_planting : int   = 75,
    days_to_harvest     : int   = 20,
    area_ha             : float = 1.0,
    market_price_per_kg : float = 25.0,
    n_mc_passes         : int   = 30,
):
    # ── Read image ──
    img_bytes = await file.read()
    img_array = np.array(
        Image.open(io.BytesIO(img_bytes)).convert('RGB')
    )
    resized   = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    tensor    = val_transform(image=resized)['image']
 
    # ── CNN ──
    cnn_model.eval()
    with torch.no_grad():
        logits    = cnn_model(tensor.unsqueeze(0).to(DEVICE))
        cnn_probs = logits.softmax(dim=1)[0].cpu().numpy()
        cnn_class = int(cnn_probs.argmax())
        cnn_conf  = float(cnn_probs[cnn_class])
    cnn_emb = cnn_model.get_embedding(
        tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()
 
    # Grad-CAM
    gradcam_b64 = run_gradcam(img_array, cnn_class)
 
    # ── LSTM ──
    feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
    feat_norm = weather_scaler.transform(feat)
    wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
    lstm_model.eval()
    with torch.no_grad():
        forecast_raw, wx_ctx, _ = lstm_model(wx_tensor)
    wx_ctx_np    = wx_ctx.cpu().numpy()
    forecast_np  = forecast_raw[0].cpu().numpy()
    peak_risk_np = forecast_np.max(axis=0)
 
    # ── Metadata ──
    meta = encode_metadata(
        crop_type, growth_stage,
        days_since_planting, days_to_harvest
    ).reshape(1, -1)
 
    # ── Fusion ──
    fusion_input = torch.tensor(
        np.concatenate([cnn_emb, wx_ctx_np, meta], axis=1),
        dtype=torch.float32
    ).to(DEVICE)
    mean_probs, std_probs = fusion_model.predict_with_uncertainty(
        fusion_input, n_passes=n_mc_passes)
    mean_probs = mean_probs[0].cpu().numpy()
    std_probs  = std_probs[0].cpu().numpy()
 
    # ── Calibration ──
    cal_probs = np.array([
        calibrators[d].predict([mean_probs[i]])[0]
        for i, d in enumerate(DISEASE_CLASSES)
    ])
 
    # ── Yield impact ──
    top_idx     = int(cal_probs.argmax())
    top_disease = DISEASE_CLASSES[top_idx]
    top_risk    = float(cal_probs[top_idx])
    top_unc     = float(std_probs[top_idx])
    severity    = cnn_conf if CLASS_NAMES[cnn_class] == top_disease else 0.5
    xgb_feat    = np.array([[top_risk, severity,
                              GROWTH_STAGES.index(growth_stage),
                              days_to_harvest, 0, 0, top_idx]])
    yield_loss  = float(xgb_model.predict(xgb_feat)[0])
 
    # ── Intervention ──
    rec          = INTERVENTIONS.get(top_disease, DEFAULT_INTERVENTION)
    fin_loss     = yield_loss * 20000 * area_ha * market_price_per_kg
    treat_cost   = rec['cost_per_ha'] * area_ha
    saved_val    = fin_loss * (rec['efficacy_pct'] / 100)
    roi          = (saved_val - treat_cost) / (treat_cost + 1e-8)
 
    if top_risk > 0.6:   urgency = 'TREAT TODAY'
    elif top_risk > 0.3: urgency = 'MONITOR — treat within 3 days if worsening'
    else:                urgency = 'LOW RISK — continue monitoring'
 
    # ── 7-day forecast per disease ──
    forecast_dates = [
        (datetime.now() + timedelta(days=i+1)).strftime('%a %d %b')
        for i in range(FORECAST)
    ]
    top5_forecast_idx = peak_risk_np.argsort()[::-1][:5]
    forecast_top5 = [
        {
            'disease'   : DISEASE_CLASSES[i],
            'peak_risk' : float(peak_risk_np[i]),
            'daily'     : [float(forecast_np[d, i]) for d in range(FORECAST)],
            'dates'     : forecast_dates,
        }
        for i in top5_forecast_idx
    ]
 
    # ── Top 5 fusion results ──
    top5_idx  = cal_probs.argsort()[::-1][:5]
    top5_diseases = [
        {
            'disease'    : DISEASE_CLASSES[i],
            'probability': float(cal_probs[i]),
            'uncertainty': float(std_probs[i]),
        }
        for i in top5_idx
    ]
 
    return {
        'cnn': {
            'detected'  : CLASS_NAMES[cnn_class],
            'confidence': cnn_conf,
            'gradcam_b64': gradcam_b64,
            'top5'      : [
                {'class': CLASS_NAMES[i], 'prob': float(cnn_probs[i])}
                for i in cnn_probs.argsort()[::-1][:5]
            ],
        },
        'fusion': {
            'top_disease'  : top_disease,
            'risk_score'   : top_risk,
            'uncertainty'  : top_unc,
            'top5_diseases': top5_diseases,
        },
        'forecast': {
            'top5'  : forecast_top5,
            'dates' : forecast_dates,
        },
        'yield': {
            'loss_pct'       : yield_loss,
            'loss_kg'        : yield_loss * 20000 * area_ha,
            'financial_loss' : fin_loss,
            'treatment_cost' : treat_cost,
            'saved_value'    : saved_val,
            'roi'            : roi,
        },
        'intervention': {
            'urgency'  : urgency,
            'fungicide': rec['fungicide'],
            'frequency': rec['frequency'],
            'timing'   : rec['timing'],
        },
        'meta': {
            'crop_type'   : crop_type,
            'growth_stage': growth_stage,
            'location'    : {'lat': lat, 'lon': lon},
            'timestamp'   : datetime.now().isoformat(),
        }
    }
 
 
@app.get("/forecast")
def get_forecast(lat: float = 18.5204, lon: float = 73.8567,
                 top_n: int = 10):
    """7-day disease risk forecast for a GPS location."""
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
                'disease'   : DISEASE_CLASSES[i],
                'peak_risk' : float(peak_risk_np[i]),
                'daily'     : [float(forecast_np[d, i]) for d in range(FORECAST)],
                'level'     : 'HIGH' if peak_risk_np[i] > 0.6
                              else 'MODERATE' if peak_risk_np[i] > 0.3
                              else 'LOW',
            }
            for i in ranked_idx
        ],
        'timestamp': datetime.now().isoformat(),
    }
 
 
@app.get("/historical")
def get_historical(disease: str, days_back: int = 365):
    """Historical DCWS risk scores for a disease."""
    if disease not in dcws_df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Disease '{disease}' not found. "
                   f"Valid options: {list(dcws_df.columns[:5])}..."
        )
    series = dcws_df[disease].tail(days_back)
    return {
        'disease': disease,
        'dates'  : series.index.strftime('%Y-%m-%d').tolist(),
        'scores' : series.round(4).tolist(),
        'mean'   : float(series.mean()),
        'max'    : float(series.max()),
    }
 
 
@app.get("/compare")
def compare_crops(
    lat          : float = 18.5204,
    lon          : float = 73.8567,
    growth_stage : str   = "fruiting",
    days_to_harvest: int = 30,
):
    """Compares risk across all crop types for current weather."""
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
        # Use zero CNN embedding for weather-only comparison
        cnn_zero     = np.zeros((1, CNN_DIM), dtype=np.float32)
        fusion_input = torch.tensor(
            np.concatenate([cnn_zero, wx_ctx_np, meta], axis=1),
            dtype=torch.float32
        ).to(DEVICE)
 
        fusion_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(fusion_model(fusion_input))[0].cpu().numpy()
 
        cal_probs = np.array([
            calibrators[d].predict([probs[i]])[0]
            for i, d in enumerate(DISEASE_CLASSES)
        ])
        results[crop] = {
            'max_risk'    : float(cal_probs.max()),
            'top_disease' : DISEASE_CLASSES[int(cal_probs.argmax())],
            'risk_level'  : 'HIGH' if cal_probs.max() > 0.6
                            else 'MODERATE' if cal_probs.max() > 0.3
                            else 'LOW',
        }
 
    return {
        'crops'    : results,
        'timestamp': datetime.now().isoformat(),
    }