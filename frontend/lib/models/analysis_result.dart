// lib/models/analysis_result.dart

class AnalysisResult {
  final CnnResult cnn;
  final FusionResult fusion;
  final ForecastData forecast;
  final YieldData yield;
  final InterventionData intervention;
  final MetaData meta;
  final Map<String, dynamic>? gradcam;
  final SegmentationData? segmentation;
  final QualityData? quality;

  AnalysisResult({
    required this.cnn,
    required this.fusion,
    required this.forecast,
    required this.yield,
    required this.intervention,
    required this.meta,
    this.gradcam,
    this.segmentation,
    this.quality,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> j) => AnalysisResult(
      cnn: CnnResult.fromJson(j['cnn']),
      fusion: FusionResult.fromJson(j['fusion']),
      // Backend doesn't return forecast inside analyze — use empty default
      forecast: ForecastData(top5: [], dates: []),
      yield: YieldData.fromJson(j['yield']),
      intervention: InterventionData.fromJson(j['intervention']),
      // Build meta from top-level fields
      meta: MetaData(
        cropType: j['cnn']?['detected'] ?? '',
        growthStage: '',
        location: Map<String, double>.from(
          (j['location'] ?? {}).map((k, v) => MapEntry(k, (v as num).toDouble())),
        ),
        timestamp: j['timestamp'] ?? '',
      ),
      gradcam: j['gradcam'] != null ? Map<String, dynamic>.from(j['gradcam']) : null,
      segmentation: j['segmentation'] != null ? SegmentationData.fromJson(j['segmentation']) : null,
      quality: j['quality'] != null ? QualityData.fromJson(j['quality']) : null,
    );

  Map<String, dynamic> toJson() => {
        'cnn': cnn.toJson(),
        'fusion': fusion.toJson(),
        'forecast': forecast.toJson(),
        'yield': yield.toJson(),
        'intervention': intervention.toJson(),
        'meta': meta.toJson(),
        'gradcam': gradcam,
        'segmentation': segmentation?.toJson(),
        'quality': quality?.toJson(),
      };
}

class CnnResult {
  final String detected;
  final double confidence;
  final List<CnnTop5> top5;

  CnnResult({
    required this.detected,
    required this.confidence,
    required this.top5,
  });

  factory CnnResult.fromJson(Map<String, dynamic> j) => CnnResult(
        detected: j['detected'] ?? '',
        confidence: (j['confidence'] ?? 0).toDouble(),
        top5: (j['top5'] as List? ?? [])
            .map((e) => CnnTop5.fromJson(e))
            .toList(),
      );

  Map<String, dynamic> toJson() => {
        'detected': detected,
        'confidence': confidence,
        'top5': top5.map((e) => e.toJson()).toList(),
      };
}

class CnnTop5 {
  final String className;
  final double prob;
  CnnTop5({required this.className, required this.prob});
  factory CnnTop5.fromJson(Map<String, dynamic> j) =>
      CnnTop5(className: j['class'] ?? '', prob: (j['prob'] ?? 0).toDouble());
  Map<String, dynamic> toJson() => {'class': className, 'prob': prob};
}

class FusionResult {
  final String topDisease;
  final double riskScore;
  final double uncertainty;
  final List<DiseaseRisk> top5Diseases;

  FusionResult({
    required this.topDisease,
    required this.riskScore,
    required this.uncertainty,
    required this.top5Diseases,
  });

  factory FusionResult.fromJson(Map<String, dynamic> j) => FusionResult(
      topDisease: j['top_disease'] ?? '',
      riskScore: (j['risk_score'] ?? 0).toDouble(),
      uncertainty: (j['uncertainty'] ?? 0).toDouble(),
      top5Diseases: (j['top5'] as List? ?? [])  // was 'top5_diseases'
          .map((e) => DiseaseRisk.fromJson(e))
          .toList(),
    );

  Map<String, dynamic> toJson() => {
        'top_disease': topDisease,
        'risk_score': riskScore,
        'uncertainty': uncertainty,
        'top5_diseases': top5Diseases.map((e) => e.toJson()).toList(),
      };

  String get riskLevel {
    if (riskScore > 0.6) return 'HIGH';
    if (riskScore > 0.3) return 'MODERATE';
    return 'LOW';
  }
}

class DiseaseRisk {
  final String disease;
  final double probability;
  final double uncertainty;

  DiseaseRisk({
    required this.disease,
    required this.probability,
    required this.uncertainty,
  });

  factory DiseaseRisk.fromJson(Map<String, dynamic> j) => DiseaseRisk(
        disease: j['disease'] ?? '',
        probability: (j['probability'] ?? 0).toDouble(),
        uncertainty: (j['uncertainty'] ?? 0).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'disease': disease,
        'probability': probability,
        'uncertainty': uncertainty,
      };

  String get displayName => disease.replaceAll('___', ' — ');
}

class ForecastData {
  final List<DiseaseForecast> top5;
  final List<String> dates;

  ForecastData({required this.top5, required this.dates});

  factory ForecastData.fromJson(Map<String, dynamic> j) => ForecastData(
        top5: (j['top5'] as List? ?? [])
            .map((e) => DiseaseForecast.fromJson(e))
            .toList(),
        dates: List<String>.from(j['dates'] ?? []),
      );

  Map<String, dynamic> toJson() => {
        'top5': top5.map((e) => e.toJson()).toList(),
        'dates': dates,
      };
}

class DiseaseForecast {
  final String disease;
  final double peakRisk;
  final List<double> daily;
  final List<String> dates;

  DiseaseForecast({
    required this.disease,
    required this.peakRisk,
    required this.daily,
    required this.dates,
  });

  factory DiseaseForecast.fromJson(Map<String, dynamic> j) => DiseaseForecast(
        disease: j['disease'] ?? '',
        peakRisk: (j['peak_risk'] ?? 0).toDouble(),
        daily: (j['daily'] as List? ?? []).map((e) => (e as num).toDouble()).toList(),
        dates: List<String>.from(j['dates'] ?? []),
      );

  Map<String, dynamic> toJson() => {
        'disease': disease,
        'peak_risk': peakRisk,
        'daily': daily,
        'dates': dates,
      };

  String get displayName => disease.replaceAll('___', ' — ');
}

class YieldData {
  final double lossPct;
  final double lossKg;
  final double financialLoss;
  final double treatmentCost;
  final double savedValue;
  final double roi;

  YieldData({
    required this.lossPct,
    required this.lossKg,
    required this.financialLoss,
    required this.treatmentCost,
    required this.savedValue,
    required this.roi,
  });

  factory YieldData.fromJson(Map<String, dynamic> j) => YieldData(
        lossPct: (j['loss_pct'] ?? 0).toDouble(),
        lossKg: (j['loss_kg'] ?? 0).toDouble(),
        financialLoss: (j['financial_loss'] ?? 0).toDouble(),
        treatmentCost: (j['treatment_cost'] ?? 0).toDouble(),
        savedValue: (j['saved_value'] ?? 0).toDouble(),
        roi: (j['roi'] ?? 0).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'loss_pct': lossPct,
        'loss_kg': lossKg,
        'financial_loss': financialLoss,
        'treatment_cost': treatmentCost,
        'saved_value': savedValue,
        'roi': roi,
      };
}

class InterventionData {
  final String urgency;
  final String fungicide;
  final String frequency;
  final String timing;

  InterventionData({
    required this.urgency,
    required this.fungicide,
    required this.frequency,
    required this.timing,
  });

  factory InterventionData.fromJson(Map<String, dynamic> j) => InterventionData(
        urgency: j['urgency'] ?? '',
        fungicide: j['fungicide'] ?? '',
        frequency: j['frequency'] ?? '',
        timing: j['timing'] ?? '',
      );

  Map<String, dynamic> toJson() => {
        'urgency': urgency,
        'fungicide': fungicide,
        'frequency': frequency,
        'timing': timing,
      };

  bool get isCritical => urgency.contains('TODAY');
  bool get isModerate => urgency.contains('MONITOR');
}

class MetaData {
  final String cropType;
  final String growthStage;
  final Map<String, double> location;
  final String timestamp;

  MetaData({
    required this.cropType,
    required this.growthStage,
    required this.location,
    required this.timestamp,
  });

  factory MetaData.fromJson(Map<String, dynamic> j) => MetaData(
        cropType: j['crop_type'] ?? '',
        growthStage: j['growth_stage'] ?? '',
        location: Map<String, double>.from(
          (j['location'] ?? {}).map((k, v) => MapEntry(k, (v as num).toDouble())),
        ),
        timestamp: j['timestamp'] ?? '',
      );

  Map<String, dynamic> toJson() => {
        'crop_type': cropType,
        'growth_stage': growthStage,
        'location': location,
        'timestamp': timestamp,
      };
}

// ── Forecast result (standalone) ──
class ForecastResult {
  final Map<String, double> location;
  final List<String> dates;
  final List<DiseaseForecastItem> diseases;
  final String timestamp;

  ForecastResult({
    required this.location,
    required this.dates,
    required this.diseases,
    required this.timestamp,
  });

  factory ForecastResult.fromJson(Map<String, dynamic> j) => ForecastResult(
        location: Map<String, double>.from(
          (j['location'] ?? {}).map((k, v) => MapEntry(k, (v as num).toDouble())),
        ),
        dates: List<String>.from(j['dates'] ?? []),
        diseases: (j['diseases'] as List? ?? [])
            .map((e) => DiseaseForecastItem.fromJson(e))
            .toList(),
        timestamp: j['timestamp'] ?? '',
      );
}

class DiseaseForecastItem {
  final String disease;
  final double peakRisk;
  final List<double> daily;
  final String level;

  DiseaseForecastItem({
    required this.disease,
    required this.peakRisk,
    required this.daily,
    required this.level,
  });

  factory DiseaseForecastItem.fromJson(Map<String, dynamic> j) =>
      DiseaseForecastItem(
        disease: j['disease'] ?? '',
        peakRisk: (j['peak_risk'] ?? 0).toDouble(),
        daily: (j['daily'] as List? ?? [])
            .map((e) => (e as num).toDouble())
            .toList(),
        level: j['level'] ?? 'LOW',
      );

  String get displayName => disease.replaceAll('___', ' — ');
}

// ── Comparison result ──
class ComparisonResult {
  final Map<String, CropRiskInfo> crops;
  final String timestamp;

  ComparisonResult({required this.crops, required this.timestamp});

  factory ComparisonResult.fromJson(Map<String, dynamic> j) => ComparisonResult(
        crops: (j['crops'] as Map<String, dynamic>? ?? {}).map(
          (k, v) => MapEntry(k, CropRiskInfo.fromJson(v)),
        ),
        timestamp: j['timestamp'] ?? '',
      );
}

class CropRiskInfo {
  final double maxRisk;
  final String topDisease;
  final String riskLevel;

  CropRiskInfo({
    required this.maxRisk,
    required this.topDisease,
    required this.riskLevel,
  });

  factory CropRiskInfo.fromJson(Map<String, dynamic> j) => CropRiskInfo(
        maxRisk: (j['max_risk'] ?? 0).toDouble(),
        topDisease: j['top_disease'] ?? '',
        riskLevel: j['risk_level'] ?? 'LOW',
      );
}

// ── Historical result ──
class HistoricalResult {
  final String disease;
  final List<String> dates;
  final List<double> scores;
  final double mean;
  final double max;

  HistoricalResult({
    required this.disease,
    required this.dates,
    required this.scores,
    required this.mean,
    required this.max,
  });

  factory HistoricalResult.fromJson(Map<String, dynamic> j) => HistoricalResult(
        disease: j['disease'] ?? '',
        dates: List<String>.from(j['dates'] ?? []),
        scores: (j['scores'] as List? ?? [])
            .map((e) => (e as num).toDouble())
            .toList(),
        mean: (j['mean'] ?? 0).toDouble(),
        max: (j['max'] ?? 0).toDouble(),
      );
}

// ── Segmentation data ──
class SegmentationData {
  final String method;
  final double leafCoverage;
  final String? warning;
  final List<int>? bbox;

  SegmentationData({
    required this.method,
    required this.leafCoverage,
    this.warning,
    this.bbox,
  });

  factory SegmentationData.fromJson(Map<String, dynamic> j) => SegmentationData(
        method: j['method'] ?? '',
        leafCoverage: (j['leaf_coverage'] ?? 0).toDouble(),
        warning: j['warning'],
        bbox: j['bbox'] != null ? List<int>.from(j['bbox']) : null,
      );

  Map<String, dynamic> toJson() => {
        'method': method,
        'leaf_coverage': leafCoverage,
        'warning': warning,
        'bbox': bbox,
      };
}

// ── Quality data ──
class QualityData {
  final double score;
  final List<String> warnings;
  final Map<String, dynamic> metrics;

  QualityData({
    required this.score,
    required this.warnings,
    required this.metrics,
  });

  factory QualityData.fromJson(Map<String, dynamic> j) => QualityData(
        score: (j['score'] ?? 0).toDouble(),
        warnings: List<String>.from(j['warnings'] ?? []),
        metrics: Map<String, dynamic>.from(j['metrics'] ?? {}),
      );

  Map<String, dynamic> toJson() => {
        'score': score,
        'warnings': warnings,
        'metrics': metrics,
      };
}

// ── Exceptions ──
class ApiException implements Exception {
  final String message;
  ApiException(this.message);
}

class QualityRejectionException implements Exception {
  final String reason;
  final List<String> suggestions;
  final double score;
  final Map<String, dynamic> metrics;

  QualityRejectionException({
    required this.reason,
    required this.suggestions,
    required this.score,
    required this.metrics,
  });
}