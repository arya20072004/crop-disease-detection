// lib/screens/compare_screen.dart
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';
import '../models/analysis_result.dart';
import '../core/core.dart';

String _cleanName(String raw) =>
    raw.replaceAll('___', ' — ').replaceAll('_', ' ');

class CompareScreen extends StatefulWidget {
  const CompareScreen({super.key});
  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen> {
  ComparisonResult? _data;
  bool _loading = false;
  String? _error;
  double _lat = 18.5204, _lon = 73.8567;
  String _growthStage = 'fruiting';
  final int _daysToHarvest = 20;
  final _api = ApiService();

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final p = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _lat = p.getDouble(AppConstants.prefLat) ?? 18.5204;
      _lon = p.getDouble(AppConstants.prefLon) ?? 73.8567;
      _growthStage = p.getString(AppConstants.prefGrowthStage) ?? 'fruiting';
    });
    _fetch();
  }

  Future<void> _fetch() async {
    setState(() { _loading = true; _error = null; });
    try {
      final result = await _api.compareCrops(
        lat: _lat,
        lon: _lon,
        growthStage: _growthStage,
        daysToHarvest: _daysToHarvest,
      );
      if (!mounted) return;
      setState(() { _data = result; _loading = false; });
    } catch (e) {
      if (!mounted) return;
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(l?.translate('which_crop_safer') ?? 'Which Crop Is Safer?'),
        actions: [IconButton(icon: const Icon(Icons.refresh), onPressed: _fetch)],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.wifi_off, size: 48, color: AppTheme.textSecondary),
                        const SizedBox(height: 12),
                        Text(_error!, textAlign: TextAlign.center),
                        const SizedBox(height: 16),
                        ElevatedButton(onPressed: _fetch, child: const Text('Retry')),
                      ],
                    ),
                  ),
                )
              : _data == null
                  ? const Center(child: Text('Tap refresh to load comparison'))
                  : _buildContent(l),
    );
  }

  Widget _buildContent(AppLocalizations? l) {
    final crops = _data!.crops.entries.toList()
      ..sort((a, b) => b.value.maxRisk.compareTo(a.value.maxRisk));

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(
          l?.translate('compare_subtitle') ??
              'Based on today\'s weather in your area',
          style: TextStyle(fontSize: 13, color: Theme.of(context).textTheme.bodySmall?.color),
        ),
        const SizedBox(height: 12),
        ...crops.map((entry) {
          final crop = entry.key;
          final info = entry.value;
          final color = AppTheme.riskColor(info.maxRisk);
          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              color: Theme.of(context).cardColor,
              border: Border.all(color: Theme.of(context).dividerColor),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.04),
                  blurRadius: 6,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Row(
              children: [
                Container(
                  width: 6,
                  height: 92,
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: const BorderRadius.only(
                      topLeft: Radius.circular(16),
                      bottomLeft: Radius.circular(16),
                    ),
                  ),
                ),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(crop,
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                        const SizedBox(height: 6),
                        Text(
                          _cleanName(info.topDisease),
                          style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary),
                        ),
                      ],
                    ),
                  ),
                ),
                Container(
                  margin: const EdgeInsets.only(right: 12),
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '${(info.maxRisk * 100).toStringAsFixed(0)}%',
                        style: TextStyle(color: color, fontSize: 22, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text(info.riskLevel, style: TextStyle(color: color, fontSize: 12)),
                    ],
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}