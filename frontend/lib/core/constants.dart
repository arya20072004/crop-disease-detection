// lib/core/constants.dart
class AppConstants {
  static const List<String> cropTypes = [
    'Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape',
    'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry',
    'Soybean', 'Squash', 'Strawberry', 'Tomato',
  ];

  static const List<String> growthStages = [
    'seedling', 'vegetative', 'flowering', 'fruiting', 'mature',
  ];

  static const Map<String, String> growthStageLabels = {
    'seedling'  : 'Seedling',
    'vegetative': 'Vegetative',
    'flowering' : 'Flowering',
    'fruiting'  : 'Fruiting',
    'mature'    : 'Mature',
  };

  static const Map<String, String> locationPresets = {
    'Pune, Maharashtra'    : '18.5204,73.8567',
    'Nashik, Maharashtra'  : '20.0059,73.7898',
    'Nagpur, Maharashtra'  : '21.1458,79.0882',
    'Delhi'                : '28.6139,77.2090',
    'Bangalore, Karnataka' : '12.9716,77.5946',
    'Hyderabad, Telangana' : '17.3850,78.4867',
    'Kolkata, West Bengal' : '22.5726,88.3639',
    'Ludhiana, Punjab'     : '30.9010,75.8573',
  };

  static const String cacheKeyLastResult  = 'last_analysis_result';
  static const String cacheKeyForecast    = 'last_forecast';
  static const String cacheKeyComparison  = 'last_comparison';
  static const String prefLang            = 'pref_language';
  static const String prefCropType        = 'pref_crop_type';
  static const String prefGrowthStage     = 'pref_growth_stage';
  static const String prefAreaHa          = 'pref_area_ha';
  static const String prefMarketPrice     = 'pref_market_price';
  static const String prefLat             = 'pref_lat';
  static const String prefLon             = 'pref_lon';
}