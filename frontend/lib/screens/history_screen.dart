// lib/screens/history_screen.dart
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/db_service.dart';
import '../models/analysis_result.dart';
import '../core/core.dart';

String _cleanName(String raw) =>
    raw.replaceAll('___', ' — ').replaceAll('_', ' ');

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  static const _diseases = [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Apple___Apple_scab',
    'Tomato___Bacterial_spot',
  ];

  HistoricalResult? _data;
  bool _loading = false;
  String? _error;
  String _disease = 'Tomato___Early_blight';
  final _api = ApiService();
  final _db  = DBService();

  List<Map<String, dynamic>> _myScans = [];
  bool _scansLoading = false;

  @override
  void initState() {
    super.initState();
    _fetch();
    _loadMyScans();
  }

  Future<void> _loadMyScans() async {
    setState(() => _scansLoading = true);
    try {
      final scans = await _db.getPredictions();
      print('[HistoryScreen] Loaded ${scans.length} scans from Supabase');
      if (!mounted) return;
      setState(() { _myScans = scans; _scansLoading = false; });
    } catch (e) {
      print('[HistoryScreen] DB ERROR getPredictions: $e');
      if (mounted) setState(() => _scansLoading = false);
    }
  }

  Future<void> _fetch() async {
    setState(() { _loading = true; _error = null; });
    try {
      final result = await _api.getHistorical(disease: _disease);
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
        title: Text(l?.translate('past_risks') ?? 'Past Risks'),
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
              : _buildContent(l),
    );
  }

  Widget _buildContent(AppLocalizations? l) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── My Recent Scans (from Supabase) ──
        if (_scansLoading)
          const Padding(
            padding: EdgeInsets.all(16),
            child: Center(child: LinearProgressIndicator()),
          )
        else if (_myScans.isNotEmpty) ...[
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
            child: Text(
              'My Recent Scans',
              style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
          ),
          SizedBox(
            height: 100,
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: _myScans.length,
              separatorBuilder: (_, __) => const SizedBox(width: 10),
              itemBuilder: (_, i) {
                final s = _myScans[i];
                final predictionId = s['id']?.toString() ?? '';
                final disease = s['disease'] as String? ?? s['top_disease'] as String? ?? '—';
                final risk = (s['risk'] as num?)?.toDouble() ?? (s['risk_score'] as num?)?.toDouble() ?? 0.0;
                final riskLevel = s['risk_level'] as String? ?? (risk > 0.6 ? 'HIGH' : risk > 0.3 ? 'MODERATE' : 'LOW');
                final color = AppTheme.riskColor(risk);
                return GestureDetector(
                  onTap: () async {
                    // Fix 2: Record that user viewed this prediction
                    if (predictionId.isNotEmpty) {
                      await _db.savePredictionHistory(predictionId);
                    }
                    if (!mounted) return;
                    // Show a quick detail sheet
                    showModalBottomSheet(
                      context: context,
                      backgroundColor: Colors.white,
                      shape: const RoundedRectangleBorder(
                        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
                      ),
                      builder: (_) => Padding(
                        padding: const EdgeInsets.all(24),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(_cleanName(disease),
                                style: TextStyle(
                                    fontSize: 18, fontWeight: FontWeight.bold, color: color)),
                            const SizedBox(height: 8),
                            Text('Risk Level: $riskLevel',
                                style: TextStyle(color: color, fontWeight: FontWeight.w600)),
                            const SizedBox(height: 4),
                            Text('Risk Score: ${(risk * 100).toStringAsFixed(1)}%'),
                            if (s['crop_name'] != null) ...[
                              const SizedBox(height: 4),
                              Text('Crop: ${s['crop_name']}'),
                            ],
                            if (s['created_at'] != null) ...[
                              const SizedBox(height: 4),
                              Text('Scanned: ${s['created_at'].toString().substring(0, 10)}',
                                  style: const TextStyle(color: Colors.grey, fontSize: 12)),
                            ],
                          ],
                        ),
                      ),
                    );
                  },
                  child: Container(
                    width: 150,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: color.withOpacity(0.3)),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _cleanName(disease),
                          style: TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.w600),
                          maxLines: 2, overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          riskLevel,
                          style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: color),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 8),
        ],

        // Disease picker
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
          child: DropdownButtonFormField<String>(
            value: _disease,
            decoration: InputDecoration(
              labelText: l?.translate('select_disease') ?? 'Select disease',
              isDense: true,
            ),
            isExpanded: true,
            items: _diseases.map((disease) {
              return DropdownMenuItem(
                value: disease,
                child: Text(
                  _cleanName(disease),
                  overflow: TextOverflow.ellipsis,
                ),
              );
            }).toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() => _disease = value);
              _fetch();
            },
          ),
        ),

        if (_data != null) ...[
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
            child: Text(
              _cleanName(_data!.disease),
              style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Row(
              children: [
                _StatChip(
                  l?.translate('mean') ?? 'Mean',
                  '${(_data!.mean * 100).toStringAsFixed(1)}%',
                  AppTheme.riskColor(_data!.mean),
                ),
                const SizedBox(width: 12),
                _StatChip(
                  l?.translate('max') ?? 'Max',
                  '${(_data!.max * 100).toStringAsFixed(1)}%',
                  AppTheme.riskColor(_data!.max),
                ),
              ],
            ),
          ),
          Expanded(
            child: Card(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              child: ListView.builder(
                itemCount: _data!.dates.length,
                itemBuilder: (_, i) {
                  final score = _data!.scores[i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    child: Row(
                      children: [
                        Container(
                          width: 16,
                          height: 16,
                          decoration: BoxDecoration(
                            color: AppTheme.riskColor(score),
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(_data!.dates[i], style: const TextStyle(fontSize: 14)),
                        ),
                        Text(
                          '${(score * 100).toStringAsFixed(1)}%',
                          style: TextStyle(
                            color: AppTheme.riskColor(score),
                            fontWeight: FontWeight.bold,
                            fontSize: 18,
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ),
        ] else
          const Expanded(child: Center(child: Text('No history data'))),
      ],
    );
  }
}

class _StatChip extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _StatChip(this.label, this.value, this.color);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Text(label, style: TextStyle(fontSize: 11, color: color)),
          Text(value,
              style: TextStyle(fontWeight: FontWeight.bold, color: color, fontSize: 20)),
        ],
      ),
    );
  }
}