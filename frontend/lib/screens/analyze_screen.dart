// ══════════════════════════════════════════════════════════
// ANALYZE SCREEN
// ══════════════════════════════════════════════════════════

// lib/screens/analyze_screen.dart
import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:geolocator/geolocator.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';
import '../services/db_service.dart';
import '../models/models.dart';
import '../core/core.dart';
import '../widgets/result_widgets.dart';

class AnalyzeScreen extends StatefulWidget {
  const AnalyzeScreen({super.key});
  @override
  State<AnalyzeScreen> createState() => _AnalyzeScreenState();
}

class _AnalyzeScreenState extends State<AnalyzeScreen> {
  File? _image;
  AnalysisResult? _result;
  bool _loading = false;
  String? _error;
  String _cropType = 'Tomato';
  String _growthStage = 'fruiting';
  int _dsp = 75;
  int _dth = 20;
  double _areaHa = 1.0;
  double _marketPrice = 25.0;
  double? _lat, _lon;
  bool _isOffline = false;

  final _api = ApiService();
  final _db  = DBService();
  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final p = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _cropType    = p.getString(AppConstants.prefCropType) ?? 'Tomato';
      _growthStage = p.getString(AppConstants.prefGrowthStage) ?? 'fruiting';
      _areaHa      = p.getDouble(AppConstants.prefAreaHa) ?? 1.0;
      _marketPrice = p.getDouble(AppConstants.prefMarketPrice) ?? 25.0;
      _lat         = p.getDouble(AppConstants.prefLat);
      _lon         = p.getDouble(AppConstants.prefLon);
      final cached = p.getString(AppConstants.cacheKeyLastResult);
      if (cached != null) {
        try {
          _result    = AnalysisResult.fromJson(jsonDecode(cached));
          _isOffline = true;
        } catch (_) {}
      }
    });
    _getLocation();
  }

  Future<void> _getLocation() async {
    try {
      final perm = await Geolocator.requestPermission();
      if (perm == LocationPermission.denied) return;
      final pos = await Geolocator.getCurrentPosition();
      if (!mounted) return;
      setState(() { _lat = pos.latitude; _lon = pos.longitude; });
      final p = await SharedPreferences.getInstance();
      await p.setDouble(AppConstants.prefLat, pos.latitude);
      await p.setDouble(AppConstants.prefLon, pos.longitude);
    } catch (_) {}
  }

  Future<void> _pickImage(ImageSource source) async {
    final xf = await _picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1024,
    );
    if (xf != null) setState(() { _image = File(xf.path); _result = null; });
  }

  Future<void> _analyze() async {
    if (_image == null || _lat == null || _lon == null) return;
    setState(() { _loading = true; _error = null; _isOffline = false; });

    // Debug: confirm selected values before any DB/API call
    print('[AnalyzeScreen] Crop: $_cropType');
    print('[AnalyzeScreen] Stage: $_growthStage');
    print('[AnalyzeScreen] dsp=$_dsp  dth=$_dth  area=$_areaHa  price=$_marketPrice  lat=$_lat  lon=$_lon');

    // Persist crop settings to Supabase
    try {
      await _db.saveCrop({
        'crop_name'           : _cropType,
        'stage'               : _growthStage,
        'days_since_planting' : _dsp,
        'days_to_harvest'     : _dth,
        'area_ha'             : _areaHa,
        'market_price_per_kg' : _marketPrice,
        'lat'                 : _lat,
        'lon'                 : _lon,
      });
    } catch (e) {
      print('[AnalyzeScreen] DB ERROR saveCrop: $e');
    }

    try {
      final result = await _api.analyzeLeaf(
        imageFile: _image!,
        lat: _lat!, lon: _lon!,
        cropType: _cropType,
        growthStage: _growthStage,
        daysSincePlanting: _dsp,
        daysToHarvest: _dth,
        areaHa: _areaHa,
        marketPrice: _marketPrice,
      );
      final p = await SharedPreferences.getInstance();
      await p.setString(AppConstants.cacheKeyLastResult, jsonEncode(result.toJson()));

      // Persist prediction result to Supabase — capture the new row's ID
      String? predictionId;
      try {
        predictionId = await _db.savePrediction({
          'crop_name'   : _cropType,
          'stage'       : _growthStage,
          'lat'         : _lat,
          'lon'         : _lon,
          'disease'     : result.cnn?.detected,
          'confidence'  : result.cnn?.confidence,
          'risk'        : result.fusion?.riskScore,
          'risk_level'  : result.fusion?.riskScore != null
              ? (result.fusion!.riskScore > 0.6 ? 'HIGH'
                : result.fusion!.riskScore > 0.3 ? 'MODERATE' : 'LOW')
              : 'UNKNOWN',
          'top_disease' : result.fusion?.topDisease,
          'created_at'  : DateTime.now().toIso8601String(),
        });
      } catch (e) {
        print('[AnalyzeScreen] DB ERROR savePrediction: $e');
      }

      // Seed recommendations table for the detected disease (if not already present)
      final detectedDisease = result.fusion?.topDisease ?? result.cnn?.detected;
      if (detectedDisease != null) {
        print('[AnalyzeScreen] Disease string for recommendations: "$detectedDisease"');
        // Insert recommendation row if missing
        try {
          await _db.saveRecommendationIfMissing(detectedDisease);
        } catch (e) {
          print('[AnalyzeScreen] DB ERROR saveRecommendationIfMissing: $e');
        }
        // Also fetch recs for logging
        try {
          final recs = await _db.getRecommendations(detectedDisease);
          if (recs.isEmpty) {
            print('[AnalyzeScreen] No recommendations found in DB for "$detectedDisease"');
          } else {
            print('[AnalyzeScreen] Recommendations found: ${recs.length} rows');
            for (final r in recs) print('[AnalyzeScreen]   → $r');
          }
        } catch (e) {
          print('[AnalyzeScreen] DB ERROR getRecommendations: $e');
        }
      }


      if (!mounted) return;
      setState(() { _result = result; _loading = false; _isOffline = false; });
    } catch (e) {
      if (!mounted) return;
      if (e is QualityRejectionException) {
        _showQualityRejectionDialog(e);
      } else {
        setState(() { _error = e.toString(); _loading = false; });
      }
    }
  }

  void _showQualityRejectionDialog(QualityRejectionException e) {
    setState(() { _loading = false; });
    if (!mounted) return;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _QualityRejectionSheet(
        reason: e.reason,
        suggestions: e.suggestions,
        score: e.score,
        onRetake: () {
          Navigator.pop(context);
          _showImagePicker(context);
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(l?.translate('analyze') ?? 'Analyze Leaf')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Offline banner
            if (_isOffline)
              Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.amber.shade100,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.amber.shade400),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.wifi_off, color: Colors.amber, size: 18),
                    const SizedBox(width: 8),
                    Expanded(child: Text(
                      l?.translate('offline_mode') ?? 'Offline — cached results',
                      style: const TextStyle(fontSize: 12),
                    )),
                  ],
                ),
              ),

            // Image picker
            GestureDetector(
              onTap: () => _showImagePicker(context),
              child: Container(
                height: 220,
                decoration: BoxDecoration(
                  color: Theme.of(context).brightness == Brightness.dark
                      ? const Color(0xFF1A2A1A)  // Dark green tint in dark mode
                      : const Color(0xFFE8F5E9), // Light green tint in light mode
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: AppTheme.primary.withOpacity(0.5),
                    style: BorderStyle.solid,
                    width: 2,
                  ),
                ),
                child: _image != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.file(_image!, fit: BoxFit.cover),
                      )
                    : Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.add_a_photo_outlined,
                              size: 60, color: AppTheme.primary),
                          const SizedBox(height: 12),
                          Text('Tap to photograph a leaf',
                              style: TextStyle(
                                  fontSize: 16,
                                  color: AppTheme.primary,
                                  fontWeight: FontWeight.w600)),
                        ],
                      ),
              ),
            ),
            if (_image == null) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _pickImage(ImageSource.camera),
                      icon: const Text('📷', style: TextStyle(fontSize: 18)),
                      label: const Text('Camera'),
                      style: OutlinedButton.styleFrom(
                        minimumSize: const Size.fromHeight(52),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                        side: BorderSide(
                          color: AppTheme.primary,
                          width: 2,
                        ),
                        foregroundColor: AppTheme.primary,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _pickImage(ImageSource.gallery),
                      icon: const Text('🖼', style: TextStyle(fontSize: 18)),
                      label: const Text('Gallery'),
                      style: OutlinedButton.styleFrom(
                        minimumSize: const Size.fromHeight(52),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                        side: BorderSide(
                          color: AppTheme.primary,
                          width: 2,
                        ),
                        foregroundColor: AppTheme.primary,
                      ),
                    ),
                  ),
                ],
              ),
            ],

            const SizedBox(height: 16),

            // Crop settings card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(l?.translate('my_crop') ?? 'My Crop',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.bold)),
                    const SizedBox(height: 12),
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: AppConstants.cropTypes.map((crop) {
                          final selected = crop == _cropType;
                          final theme = Theme.of(context);
                          final bgColor = selected
                              ? AppTheme.primary.withOpacity(0.15)
                              : theme.brightness == Brightness.dark
                                  ? const Color(0xFF2A2A2A)
                                  : Colors.grey[200]!;
                          final textColor = selected
                              ? AppTheme.primary
                              : theme.colorScheme.onBackground;
                          return Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: ChoiceChip(
                              label: Text(crop),
                              selected: selected,
                              onSelected: (_) => setState(() => _cropType = crop),
                              selectedColor: AppTheme.primary.withOpacity(0.15),
                              backgroundColor: bgColor,
                              labelStyle: TextStyle(
                                color: textColor,
                                fontWeight: selected ? FontWeight.w700 : FontWeight.w600,
                              ),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                                side: BorderSide(
                                  color: selected ? AppTheme.primary : Colors.grey.shade400,
                                  width: 1.5,
                                ),
                              ),
                              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                            ),
                          );
                        }).toList(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: AppConstants.growthStages.map((stage) {
                          final selected = stage == _growthStage;
                          final theme = Theme.of(context);
                          final bgColor = selected
                              ? AppTheme.primary.withOpacity(0.15)
                              : theme.brightness == Brightness.dark
                                  ? const Color(0xFF2A2A2A)
                                  : Colors.grey[200]!;
                          final textColor = selected
                              ? AppTheme.primary
                              : theme.colorScheme.onBackground;
                          return Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: ChoiceChip(
                              label: Text(AppConstants.growthStageLabels[stage] ?? stage),
                              selected: selected,
                              onSelected: (_) => setState(() => _growthStage = stage),
                              selectedColor: AppTheme.primary.withOpacity(0.15),
                              backgroundColor: bgColor,
                              labelStyle: TextStyle(
                                color: textColor,
                                fontWeight: selected ? FontWeight.w700 : FontWeight.w600,
                              ),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                                side: BorderSide(
                                  color: selected ? AppTheme.primary : Colors.grey.shade400,
                                  width: 1.5,
                                ),
                              ),
                              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                            ),
                          );
                        }).toList(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(child: _NumericField(
                          label: 'Days planted',
                          value: _dsp.toDouble(),
                          onChanged: (v) => setState(() => _dsp = v.toInt()),
                        )),
                        const SizedBox(width: 12),
                        Expanded(child: _NumericField(
                          label: 'Days to harvest',
                          value: _dth.toDouble(),
                          onChanged: (v) => setState(() => _dth = v.toInt()),
                        )),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(child: _NumericField(
                          label: 'Area (ha)',
                          value: _areaHa,
                          onChanged: (v) => setState(() => _areaHa = v),
                          isDecimal: true,
                        )),
                        const SizedBox(width: 12),
                        Expanded(child: _NumericField(
                          label: 'Price (₹/kg)',
                          value: _marketPrice,
                          onChanged: (v) => setState(() => _marketPrice = v),
                          isDecimal: true,
                        )),
                      ],
                    ),
                    if (_lat != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Row(
                          children: [
                            const Icon(Icons.location_on,
                                size: 16, color: AppTheme.primary),
                            const SizedBox(width: 4),
                            Text(
                              '${_lat!.toStringAsFixed(4)}, ${_lon!.toStringAsFixed(4)}',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                  color: AppTheme.textSecondary),
                            ),
                            const Spacer(),
                            TextButton(
                              onPressed: _getLocation,
                              child: const Text('Refresh GPS'),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Analyze button
            ElevatedButton.icon(
              onPressed: (_image != null && _lat != null && !_loading)
                  ? _analyze
                  : null,
              icon: _loading
                  ? const SizedBox(
                      width: 20, height: 20,
                      child: CircularProgressIndicator(
                          color: Colors.white, strokeWidth: 2))
                  : const Icon(Icons.search),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(56),
              ),
              label: Text(_loading
                  ? (l?.translate('checking') ?? 'Checking...')
                  : (l?.translate('check_crop') ?? 'Check My Crop')),
            ),
            if (_image != null && _lat == null && !_loading)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.location_off, size: 14,
                        color: AppTheme.textSecondary),
                    const SizedBox(width: 4),
                    Text(
                      'Waiting for GPS location…',
                      style: Theme.of(context).textTheme.bodySmall
                          ?.copyWith(color: AppTheme.textSecondary),
                    ),
                  ],
                ),
              ),

            if (_error != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(_error!,
                        style: TextStyle(color: Colors.red.shade700, fontSize: 13)),
                    const SizedBox(height: 10),
                    TextButton(
                      onPressed: () => Navigator.pushNamed(context, '/settings'),
                      child: const Text('Need help?'),
                    ),
                  ],
                ),
              ),
            ],

            // Results
            if (_result != null) ...[
              const SizedBox(height: 24),
              AnalysisResultWidget(result: _result!),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: () {
                  final detectedDisease = _result!.fusion?.topDisease ?? _result!.cnn?.detected;
                  final riskLevel = _result!.fusion?.riskScore != null
                      ? (_result!.fusion!.riskScore > 0.85
                          ? 'Critical'
                          : _result!.fusion!.riskScore > 0.6
                              ? 'High'
                              : _result!.fusion!.riskScore > 0.3
                                  ? 'Medium'
                                  : 'Low')
                      : 'Medium';
                  final recommendationArgs = {
                    'disease': detectedDisease,
                    'crop_type': _cropType,
                    'growth_stage': _growthStage,
                    'risk_level': riskLevel,
                    'area_ha': _areaHa,
                    'confidence': _result!.cnn?.confidence,
                    'prediction_id': null, // since we removed saving
                  };
                  Navigator.pushNamed(context, '/recommendations', arguments: recommendationArgs);
                },
                icon: const Icon(Icons.medical_services),
                label: const Text('Get Treatment Recommendations'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  void _showImagePicker(BuildContext ctx) {
    showModalBottomSheet(
      context: ctx,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt),
              title: const Text('Take photo'),
              onTap: () {
                Navigator.pop(ctx);
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('Choose from gallery'),
              onTap: () {
                Navigator.pop(ctx);
                _pickImage(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }
}

class _NumericField extends StatefulWidget {
  final String label;
  final double value;
  final ValueChanged<double> onChanged;
  final bool isDecimal;

  const _NumericField({
    required this.label,
    required this.value,
    required this.onChanged,
    this.isDecimal = false,
  });

  @override
  State<_NumericField> createState() => _NumericFieldState();
}

class _NumericFieldState extends State<_NumericField> {
  late final TextEditingController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(
      text: widget.isDecimal
          ? widget.value.toString()
          : widget.value.toInt().toString(),
    );
  }

  @override
  void didUpdateWidget(_NumericField old) {
    super.didUpdateWidget(old);
    if (old.value != widget.value) {
      final newText = widget.isDecimal
          ? widget.value.toString()
          : widget.value.toInt().toString();
      if (_ctrl.text != newText) {
        _ctrl.text = newText;
        _ctrl.selection = TextSelection.collapsed(offset: newText.length);
      }
    }
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: _ctrl,
      decoration: InputDecoration(labelText: widget.label, isDense: true),
      keyboardType: TextInputType.numberWithOptions(decimal: widget.isDecimal),
      onChanged: (v) {
        final parsed = double.tryParse(v);
        if (parsed != null) widget.onChanged(parsed);
      },
    );
  }
}

class _QualityRejectionSheet extends StatelessWidget {
  final String reason;
  final List<String> suggestions;
  final double score;
  final VoidCallback onRetake;

  const _QualityRejectionSheet({
    required this.reason,
    required this.suggestions,
    required this.score,
    required this.onRetake,
  });

  Color get _scoreColor {
    if (score >= 80) return Colors.green;
    if (score >= 55) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: EdgeInsets.fromLTRB(
        24, 16, 24,
        24 + MediaQuery.of(context).viewInsets.bottom,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Drag handle
          Center(
            child: Container(
              width: 40, height: 4,
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Title row
          Row(
            children: [
              const Icon(Icons.warning_amber_rounded,
                  color: Colors.orange, size: 26),
              const SizedBox(width: 10),
              const Expanded(
                child: Text('Photo Quality Issue',
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.bold)),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: _scoreColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: _scoreColor.withOpacity(0.4)),
                ),
                child: Text(
                  '${score.toStringAsFixed(0)}/100',
                  style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: _scoreColor),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),

          // Reason
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.red.shade50,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.red.shade100),
            ),
            child: Text(reason,
                style: TextStyle(
                    fontSize: 14, color: Colors.red.shade800)),
          ),

          // Suggestions
          if (suggestions.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text('How to fix:',
                style: TextStyle(
                    fontWeight: FontWeight.w600, fontSize: 14)),
            const SizedBox(height: 8),
            ...suggestions.map((tip) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('→  ',
                      style: TextStyle(
                          color: Colors.green,
                          fontWeight: FontWeight.bold,
                          fontSize: 14)),
                  Expanded(
                    child: Text(tip,
                        style: const TextStyle(fontSize: 13)),
                  ),
                ],
              ),
            )),
          ],
          const SizedBox(height: 20),

          // Retake button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: onRetake,
              icon: const Icon(Icons.camera_alt_outlined),
              label: const Text('Retake Photo'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 0,
              ),
            ),
          ),
          const SizedBox(height: 4),

          // Dismiss option
          SizedBox(
            width: double.infinity,
            child: TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('Dismiss',
                  style: TextStyle(color: Colors.grey.shade600)),
            ),
          ),
        ],
      ),
    );
  }
}