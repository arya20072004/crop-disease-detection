// lib/config/api_config.dart
import 'package:shared_preferences/shared_preferences.dart';

/// Global configuration for API server URLs.
/// Allows users to configure the FastAPI backend endpoint from settings.
class ApiConfig {
  static const String _apiUrlKey = 'api_base_url';
  static const String _defaultUrl = 'http://10.0.2.2:8000';

  /// Retrieves the saved API base URL, or returns the default.
  static Future<String> getBaseUrl() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return prefs.getString(_apiUrlKey) ?? _defaultUrl;
    } catch (e) {
      return _defaultUrl;
    }
  }

  /// Saves a new API base URL to SharedPreferences.
  static Future<void> setBaseUrl(String url) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_apiUrlKey, url.trim());
    } catch (e) {
      throw Exception('Failed to save API URL: $e');
    }
  }

  /// Constructs a full endpoint URL by appending path to the saved base URL.
  /// Example: endpoint('/predict') → 'http://10.0.2.2:8000/predict'
  static Future<String> endpoint(String path) async {
    final base = await getBaseUrl();
    final cleanBase = base.endsWith('/') ? base.substring(0, base.length - 1) : base;
    final cleanPath = path.startsWith('/') ? path : '/$path';
    return '$cleanBase$cleanPath';
  }
}
