// lib/services/api_service.dart
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/analysis_result.dart';

class ApiService {
  static const String _baseUrlKey = 'api_base_url';
  static const String _defaultUrl = 'http://10.0.2.2:8000'; // Android emulator
  // For physical device use your ngrok URL:
  // static const String _defaultUrl = 'https://xxxx.ngrok.io';

  late Dio _dio;
  String _baseUrl = _defaultUrl;
  late final Future<void> ready;

  ApiService() {
    _dio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 120),
      headers: {'Content-Type': 'application/json'},
      validateStatus: (status) => status != null && status < 500,
    ));
    ready = _loadBaseUrl();
  }

  String _normalizeBaseUrl(String url) {
    var normalized = url.trim();
    if (normalized.isEmpty) {
      throw ArgumentError('API base URL cannot be empty');
    }
    // Add scheme if missing
    if (!normalized.startsWith('http://') && !normalized.startsWith('https://')) {
      normalized = 'http://$normalized';
    }
    // Remove trailing slash
    if (normalized.endsWith('/')) {
      normalized = normalized.substring(0, normalized.length - 1);
    }
    return normalized;
  }

  Future<void> _loadBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = prefs.getString(_baseUrlKey) ?? _defaultUrl;
    _baseUrl = _normalizeBaseUrl(_baseUrl);
    _dio.options.baseUrl = _baseUrl;
  }

  Future<void> setBaseUrl(String url) async {
    final normalized = _normalizeBaseUrl(url);
    _baseUrl = normalized;
    _dio.options.baseUrl = normalized;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_baseUrlKey, normalized);
  }

  String get baseUrl => _baseUrl;

  // ── Health check ──
  Future<bool> checkHealth() async {
    try {
      final resp = await _dio.get('/health');
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Config ──
  Future<Map<String, dynamic>> getConfig() async {
    try {
      final resp = await _dio.get('/config');
      return Map<String, dynamic>.from(resp.data);
    } catch (e) {
      throw ApiException('Failed to load config: $e');
    }
  }

  // ── Full analysis ──
  Future<AnalysisResult> analyzeLeaf({
    required File imageFile,
    required double lat,
    required double lon,
    required String cropType,
    required String growthStage,
    required int daysSincePlanting,
    required int daysToHarvest,
    required double areaHa,
    required double marketPrice,
    int mcPasses = 30,
  }) async {
    try {
      final formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          imageFile.path,
          filename: 'leaf.jpg',
        ),
        'lat': lat,
        'lon': lon,
        'crop_type': cropType,
        'growth_stage': growthStage,
        'days_since_planting': daysSincePlanting,
        'days_to_harvest': daysToHarvest,
        'area_ha': areaHa,
        'market_price_per_kg': marketPrice,
        'n_mc_passes': mcPasses,
      });

      final resp = await _dio.post('/analyze', data: formData);

      // Handle 422 quality rejection (no longer a DioException thanks to validateStatus)
      if (resp.statusCode == 422) {
        // FastAPI wraps HTTPException detail in {"detail": {...}}
        final raw = resp.data;
        final data = (raw is Map && raw['detail'] is Map)
            ? Map<String, dynamic>.from(raw['detail'])
            : (raw is Map ? Map<String, dynamic>.from(raw) : null);

        if (data != null && data['error'] == 'image_quality_rejected') {
          throw QualityRejectionException(
            reason: data['reason'] ?? 'Unknown quality issue',
            suggestions: List<String>.from(data['suggestions'] ?? []),
            score: (data['score'] ?? 0).toDouble(),
            metrics: Map<String, dynamic>.from(data['metrics'] ?? {}),
          );
        }
        final reason = data?['reason'] ?? data?['detail'] ?? 'Unprocessable request (422)';
        throw ApiException('Analysis failed: $reason');
      }

      return AnalysisResult.fromJson(Map<String, dynamic>.from(resp.data));
    } on QualityRejectionException {
      rethrow;
    } on DioException catch (e) {
      throw ApiException('Analysis failed: ${e.message}');
    }
  }

  Future<Map<String, dynamic>> recommendTreatment({
    required String disease,
    required String cropType,
    required String growthStage,
    required String riskLevel,
    double areaHa = 1.0,
  }) async {
    try {
      final resp = await _dio.post('/recommend', data: {
        'disease': disease,
        'crop_type': cropType,
        'growth_stage': growthStage,
        'risk_level': riskLevel,
        'area_ha': areaHa,
      });
      return Map<String, dynamic>.from(resp.data);
    } on DioException catch (e) {
      throw ApiException('Recommendation failed: ${e.message}');
    }
  }

  // ── 7-day forecast ──
  Future<ForecastResult> getForecast({
    required double lat,
    required double lon,
    int topN = 10,
  }) async {
    try {
      final resp = await _dio.get('/forecast', queryParameters: {
        'lat': lat,
        'lon': lon,
        'top_n': topN,
      });
      return ForecastResult.fromJson(Map<String, dynamic>.from(resp.data));
    } on DioException catch (e) {
      throw ApiException('Forecast failed: ${e.message}');
    }
  }

  // ── Crop comparison ──
  Future<ComparisonResult> compareCrops({
    required double lat,
    required double lon,
    required String growthStage,
    required int daysToHarvest,
  }) async {
    try {
      final resp = await _dio.get('/compare', queryParameters: {
        'lat': lat,
        'lon': lon,
        'growth_stage': growthStage,
        'days_to_harvest': daysToHarvest,
      });
      return ComparisonResult.fromJson(Map<String, dynamic>.from(resp.data));
    } on DioException catch (e) {
      throw ApiException('Comparison failed: ${e.message}');
    }
  }

  // ── Historical data ──
  Future<HistoricalResult> getHistorical({
    required String disease,
    int daysBack = 365,
  }) async {
    try {
      final resp = await _dio.get('/historical', queryParameters: {
        'disease': disease,
        'days_back': daysBack,
      });
      return HistoricalResult.fromJson(Map<String, dynamic>.from(resp.data));
    } on DioException catch (e) {
      throw ApiException('Historical fetch failed: ${e.message}');
    }
  }
}

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}