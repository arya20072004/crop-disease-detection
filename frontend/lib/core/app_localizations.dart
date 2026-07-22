// lib/core/app_localizations.dart
import 'package:flutter/material.dart';

class AppLocalizations {
  final Locale locale;
  AppLocalizations(this.locale);

  static AppLocalizations? of(BuildContext context) =>
      Localizations.of<AppLocalizations>(context, AppLocalizations);

  static const _strings = {
    'en': {
      // App
      'app_title'           : 'Crop Disease EWS',
      // Nav
      'scan_leaf'           : 'Scan',
      'next_7_days'         : 'Forecast',
      'compare'             : 'Compare',
      'past_risks'          : 'History',
      'settings'            : 'Settings',
      // Home
      'hero_title'          : 'Is your crop sick?',
      'hero_subtitle'       : 'Point your phone at a sick leaf to get help',
      'what_to_do'          : 'What To Do',
      'photo_tips_title'    : 'How to take a good photo',
      'tip_1'               : 'Get close to one leaf — fill the screen with it',
      'tip_2'               : 'Use natural daylight — avoid shade',
      'tip_3'               : 'Hold the phone steady — no blur',
      'tip_4'               : 'Allow location — needed for weather data',
      // Analyze
      'analyze'             : 'Analyze Leaf',
      'my_crop'             : 'My Crop',
      'check_crop'          : 'Check My Crop',
      'checking'            : 'Checking...',
      'offline_mode'        : 'Offline — showing cached results',
      // Results
      'disease_found'       : 'Disease Found',
      'could_also_be'       : 'Could also be:',
      'risk_level'          : 'Risk Level',
      'act_today'           : 'Your crop is in danger. Act today.',
      'watch_carefully'     : 'Watch your crop carefully this week.',
      'crop_healthy'        : 'Your crop looks healthy. Keep monitoring.',
      'money_impact'        : 'Money Impact',
      // Forecast
      'forecast_info'       : 'Based on weather in your area. Higher % = more danger.',
      'biggest_dangers'     : 'Biggest dangers this week',
      'all_diseases_watch'  : 'All diseases — watch list',
      // Compare
      'which_crop_safer'    : 'Which Crop Is Safer?',
      'compare_subtitle'    : 'Based on today\'s weather in your area',
      // History
      'select_disease'      : 'Select disease',
      'mean'                : 'Mean',
      'max'                 : 'Max',
      // Misc
      'language'            : 'Language',
      'api_url'             : 'API Server URL',
      'fetching'            : 'Fetching data...',
      'analyzing'           : 'Analyzing...',
      'no_connection'       : 'No internet connection',
    },
    'hi': {
      // App
      'app_title'           : 'फसल रोग प्रारंभिक चेतावनी',
      // Nav
      'scan_leaf'           : 'स्कैन',
      'next_7_days'         : 'पूर्वानुमान',
      'compare'             : 'तुलना',
      'past_risks'          : 'इतिहास',
      'settings'            : 'सेटिंग्स',
      // Home
      'hero_title'          : 'क्या आपकी फसल बीमार है?',
      'hero_subtitle'       : 'एक बीमार पत्ती पर फोन रखें और मदद पाएं',
      'what_to_do'          : 'क्या करें',
      'photo_tips_title'    : 'अच्छी फोटो कैसे लें',
      'tip_1'               : 'एक पत्ती के पास जाएं — स्क्रीन भरें',
      'tip_2'               : 'प्राकृतिक रोशनी में लें — छाया से बचें',
      'tip_3'               : 'फोन स्थिर रखें — धुंधला नहीं',
      'tip_4'               : 'लोकेशन दें — मौसम डेटा के लिए जरूरी है',
      // Analyze
      'analyze'             : 'पत्ती विश्लेषण',
      'my_crop'             : 'मेरी फसल',
      'check_crop'          : 'मेरी फसल जांचें',
      'checking'            : 'जांच हो रही है...',
      'offline_mode'        : 'ऑफलाइन — कैश परिणाम दिखाए जा रहे हैं',
      // Results
      'disease_found'       : 'रोग मिला',
      'could_also_be'       : 'यह भी हो सकता है:',
      'risk_level'          : 'जोखिम स्तर',
      'act_today'           : 'आपकी फसल खतरे में है। आज इलाज करें।',
      'watch_carefully'     : 'इस हफ्ते अपनी फसल पर ध्यान दें।',
      'crop_healthy'        : 'आपकी फसल स्वस्थ दिखती है। निगरानी रखें।',
      'money_impact'        : 'पैसों पर असर',
      // Forecast
      'forecast_info'       : 'आपके क्षेत्र के मौसम पर आधारित। ज्यादा % = ज्यादा खतरा।',
      'biggest_dangers'     : 'इस हफ्ते के सबसे बड़े खतरे',
      'all_diseases_watch'  : 'सभी रोग — ध्यान दें',
      // Compare
      'which_crop_safer'    : 'कौन सी फसल सुरक्षित है?',
      'compare_subtitle'    : 'आज के आपके क्षेत्र के मौसम पर आधारित',
      // History
      'select_disease'      : 'रोग चुनें',
      'mean'                : 'औसत',
      'max'                 : 'अधिकतम',
      // Misc
      'language'            : 'भाषा',
      'api_url'             : 'API सर्वर URL',
      'fetching'            : 'डेटा प्राप्त हो रहा है...',
      'analyzing'           : 'विश्लेषण हो रहा है...',
      'no_connection'       : 'इंटरनेट कनेक्शन नहीं',
    },
    'mr': {
      // App
      'app_title'           : 'पीक रोग पूर्व चेतावणी',
      // Nav
      'scan_leaf'           : 'स्कॅन',
      'next_7_days'         : 'अंदाज',
      'compare'             : 'तुलना',
      'past_risks'          : 'इतिहास',
      'settings'            : 'सेटिंग्ज',
      // Home
      'hero_title'          : 'तुमचे पीक आजारी आहे का?',
      'hero_subtitle'       : 'आजारी पानावर फोन धरा आणि मदत मिळवा',
      'what_to_do'          : 'काय करावे',
      'photo_tips_title'    : 'चांगला फोटो कसा घ्यावा',
      'tip_1'               : 'एका पानाजवळ जा — स्क्रीन भरा',
      'tip_2'               : 'नैसर्गिक प्रकाशात घ्या — सावली टाळा',
      'tip_3'               : 'फोन स्थिर ठेवा — अस्पष्ट नको',
      'tip_4'               : 'लोकेशन द्या — हवामान डेटासाठी आवश्यक',
      // Analyze
      'analyze'             : 'पान विश्लेषण',
      'my_crop'             : 'माझे पीक',
      'check_crop'          : 'माझे पीक तपासा',
      'checking'            : 'तपासत आहे...',
      'offline_mode'        : 'ऑफलाइन — कॅश केलेले परिणाम दाखवत आहे',
      // Results
      'disease_found'       : 'रोग सापडला',
      'could_also_be'       : 'हे पण असू शकते:',
      'risk_level'          : 'धोक्याची पातळी',
      'act_today'           : 'तुमचे पीक धोक्यात आहे. आजच उपचार करा.',
      'watch_carefully'     : 'या आठवड्यात पीकावर काळजीपूर्वक लक्ष ठेवा.',
      'crop_healthy'        : 'तुमचे पीक निरोगी दिसते. निरीक्षण ठेवा.',
      'money_impact'        : 'पैशांवर परिणाम',
      // Forecast
      'forecast_info'       : 'तुमच्या परिसरातील हवामानावर आधारित. जास्त % = जास्त धोका.',
      'biggest_dangers'     : 'या आठवड्यातील सर्वात मोठे धोके',
      'all_diseases_watch'  : 'सर्व रोग — लक्ष ठेवा',
      // Compare
      'which_crop_safer'    : 'कोणते पीक सुरक्षित आहे?',
      'compare_subtitle'    : 'आजच्या तुमच्या परिसरातील हवामानावर आधारित',
      // History
      'select_disease'      : 'रोग निवडा',
      'mean'                : 'सरासरी',
      'max'                 : 'जास्तीत जास्त',
      // Misc
      'language'            : 'भाषा',
      'api_url'             : 'API सर्व्हर URL',
      'fetching'            : 'डेटा मिळवत आहे...',
      'analyzing'           : 'विश्लेषण होत आहे...',
      'no_connection'       : 'इंटरनेट कनेक्शन नाही',
    },
  };

  String translate(String key) {
    final lang = locale.languageCode;
    return _strings[lang]?[key] ?? _strings['en']?[key] ?? key;
  }

  static const delegate = _AppLocalizationsDelegate();
  static const supportedLocales = [
    Locale('en'),
    Locale('hi'),
    Locale('mr'),
  ];
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  bool isSupported(Locale locale) =>
      ['en', 'hi', 'mr'].contains(locale.languageCode);

  @override
  Future<AppLocalizations> load(Locale locale) async =>
      AppLocalizations(locale);

  @override
  bool shouldReload(_) => false;
}