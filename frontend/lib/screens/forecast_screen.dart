// lib/screens/forecast_screen.dart
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';
import '../services/db_service.dart';
import '../models/models.dart';
import '../core/core.dart';

String _cleanName(String raw) =>
    raw.replaceAll('___', ' — ').replaceAll('_', ' ');

class ForecastScreen extends StatefulWidget {
  const ForecastScreen({super.key});
  @override
  State<ForecastScreen> createState() => _ForecastScreenState();
}

class _ForecastScreenState extends State<ForecastScreen> {
  ForecastResult? _data;
  bool _loading = false;
  String? _error;
  double _lat = 18.5204, _lon = 73.8567;
  final _api = ApiService();
  final _db  = DBService();

  @override
  void initState() {
    super.initState();
    _loadLocation();
  }

  Future<void> _loadLocation() async {
    final p = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _lat = p.getDouble(AppConstants.prefLat) ?? 18.5204;
      _lon = p.getDouble(AppConstants.prefLon) ?? 73.8567;
    });
    _fetch();
  }

  Future<void> _fetch() async {
    setState(() { _loading = true; _error = null; });
    try {
      final result = await _api.getForecast(lat: _lat, lon: _lon);
      final p = await SharedPreferences.getInstance();
      await p.setString(AppConstants.cacheKeyForecast,
          jsonEncode({'diseases': result.diseases.map((d) => {
            'disease': d.disease,
            'peak_risk': d.peakRisk,
            'daily': d.daily,
            'level': d.level,
          }).toList()}));

      // Save forecast to Supabase — schema: risk_today, risk_day7, highest_risk
      try {
        await _db.saveForecast(
          diseases: result.diseases.take(10).map((d) => {
            'disease'  : d.disease,
            'peak_risk': d.peakRisk,
            'level'    : d.level,
            'daily'    : d.daily,   // needed to extract risk_today / risk_day7
          }).toList(),
        );
      } catch (e) {
        print('[ForecastScreen] DB ERROR saveForecast: $e');
      }

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
        title: Text(l?.translate('next_7_days') ?? 'Next 7 Days'),
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
                  ? const Center(child: Text('Tap refresh to load forecast'))
                  : _buildContent(l),
    );
  }

  Widget _buildContent(AppLocalizations? l) {
    final diseases = _data!.diseases;
    final topFive = diseases.take(5).toList();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Info banner
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFFFFF8E1),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFFFFCC02)),
            ),
            child: Text(
              l?.translate('forecast_info') ??
                  'Based on weather in your area. Higher % = more danger.',
              style: const TextStyle(fontSize: 13, color: Color(0xFF5D4037)),
            ),
          ),
          const SizedBox(height: 16),

          Text(
            l?.translate('biggest_dangers') ?? 'Biggest dangers this week',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),

          // Horizontal scroll cards — fixed height to avoid overflow
          SizedBox(
            height: 112,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: topFive.length,
              separatorBuilder: (_, __) => const SizedBox(width: 10),
              itemBuilder: (_, i) {
                final d = topFive[i];
                final color = AppTheme.riskColor(d.peakRisk);
                // Show only the disease part after the crop — cleaned
                final displayName = _cleanName(d.disease).split(' — ').last;
                return Container(
                  width: 130,
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: color.withOpacity(0.3)),
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        displayName,
                        style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.w600),
                        textAlign: TextAlign.center,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '${(d.peakRisk * 100).toStringAsFixed(0)}%',
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: color),
                      ),
                      Text(d.level, style: TextStyle(fontSize: 10, color: color)),
                    ],
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 20),

          Text(
            l?.translate('all_diseases_watch') ?? 'All diseases — watch list',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),

          Card(
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: AppTheme.primary.withOpacity(0.08),
                    borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
                  ),
                  child: const Row(
                    children: [
                      Expanded(flex: 4,
                          child: Text('Disease', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12))),
                      Expanded(child: Text('Highest', textAlign: TextAlign.center,
                          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12))),
                      Expanded(child: Text('Today', textAlign: TextAlign.center,
                          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12))),
                      Expanded(child: Text('Day 7', textAlign: TextAlign.center,
                          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12))),
                    ],
                  ),
                ),
                ...diseases.map((d) => Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    border: Border(top: BorderSide(color: Colors.grey.shade100)),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        flex: 4,
                        child: Text(
                          _cleanName(d.disease),
                          style: const TextStyle(fontSize: 11),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Expanded(
                        child: Text(
                          '${(d.peakRisk * 100).toStringAsFixed(0)}%',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              color: AppTheme.riskColor(d.peakRisk)),
                        ),
                      ),
                      Expanded(
                        child: Text(
                          d.daily.isNotEmpty
                              ? '${(d.daily.first * 100).toStringAsFixed(0)}%'
                              : '-',
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 11),
                        ),
                      ),
                      Expanded(
                        child: Text(
                          d.daily.length >= 7
                              ? '${(d.daily[6] * 100).toStringAsFixed(0)}%'
                              : '-',
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 11),
                        ),
                      ),
                    ],
                  ),
                )),
              ],
            ),
          ),
        ],
      ),
    );
  }
}