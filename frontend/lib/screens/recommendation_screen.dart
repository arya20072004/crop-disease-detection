import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shimmer/shimmer.dart';
import '../services/api_service.dart';
import '../services/db_service.dart';
import '../core/core.dart';

class RecommendationScreen extends StatefulWidget {
  const RecommendationScreen({super.key});

  @override
  State<RecommendationScreen> createState() => _RecommendationScreenState();
}

class _RecommendationScreenState extends State<RecommendationScreen> {
  final ApiService _api = ApiService();
  final DBService _db = DBService();
  bool _loading = true;
  String? _error;
  Recommendation? _recommendation;
  String? _cropType;
  String? _growthStage;
  double _confidence = 0.0;
  double _areaHa = 1.0;
  String? _predictionId;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _fetchRecommendation();
    });
  }

  Future<void> _fetchRecommendation() async {
    final route = ModalRoute.of(context);
    final args = route?.settings.arguments;
    if (args == null || args is! Map<String, dynamic>) {
      setState(() {
        _error = 'Recommendation arguments were not provided.';
        _loading = false;
      });
      return;
    }

    final disease = args['disease']?.toString() ?? '';
    final cropType = args['crop_type']?.toString() ?? '';
    final growthStage = args['growth_stage']?.toString() ?? '';
    final riskLevel = args['risk_level']?.toString() ?? 'Medium';
    final areaHa = (args['area_ha'] is num) ? (args['area_ha'] as num).toDouble() : double.tryParse(args['area_ha']?.toString() ?? '') ?? 1.0;
    final confidence = (args['confidence'] is num) ? (args['confidence'] as num).toDouble() : double.tryParse(args['confidence']?.toString() ?? '') ?? 0.0;
    final predictionId = args['prediction_id']?.toString();

    setState(() {
      _cropType = cropType;
      _growthStage = growthStage;
      _areaHa = areaHa;
      _confidence = confidence;
      _predictionId = predictionId;
      _loading = true;
      _error = null;
    });

    try {
      final data = await _api.recommendTreatment(
        disease: disease,
        cropType: cropType,
        growthStage: growthStage,
        riskLevel: riskLevel,
        areaHa: areaHa,
      );
      setState(() {
        _recommendation = Recommendation.fromJson(data);
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Color _severityColor(String severity) {
    switch (severity.toLowerCase()) {
      case 'critical':
        return Colors.red.shade700;
      case 'high':
        return Colors.red;
      case 'medium':
        return Colors.orange.shade700;
      default:
        return Colors.green;
    }
  }

  Future<void> _onSaveReport() async {
    if (_predictionId != null) {
      await _db.savePredictionHistory(_predictionId!);
    }
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Saved report and queued PDF export (placeholder)'),
      ),
    );
  }

  Future<void> _onShare() async {
    if (_recommendation == null) return;
    final summary = StringBuffer();
    summary.writeln('Disease: ${_recommendation!.disease}');
    summary.writeln('Crop: ${_cropType ?? ''}');
    summary.writeln('Stage: ${_growthStage ?? ''}');
    summary.writeln('Risk: ${_recommendation!.severity}');
    summary.writeln('Confidence: ${(_confidence * 100).toStringAsFixed(1)}%');
    summary.writeln('\nImmediate actions:');
    for (final action in _recommendation!.immediateActions) {
      summary.writeln('• $action');
    }
    if (_recommendation!.fertilizers.isNotEmpty) {
      summary.writeln('\nRecommended treatments:');
      for (final item in _recommendation!.fertilizers) {
        summary.writeln('- ${item.name} (${item.type}) ${item.dosage}');
      }
    }
    summary.writeln('\nWarning: ${_recommendation!.warning}');
    await Share.share(summary.toString());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Treatment Plan'),
      ),
      body: _loading
          ? _buildLoading()
          : _error != null
              ? _buildError()
              : _recommendation == null
                  ? const Center(child: Text('No recommendation available.'))
                  : SingleChildScrollView(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          _buildSummaryCard(),
                          const SizedBox(height: 16),
                          _buildImmediateActionsCard(),
                          const SizedBox(height: 16),
                          _buildTreatmentsCard(),
                          const SizedBox(height: 16),
                          _buildOrganicCard(),
                          const SizedBox(height: 16),
                          _buildScheduleCard(),
                          const SizedBox(height: 16),
                          _buildListCard(
                            title: '🌾 Field Management',
                            items: _recommendation!.culturalPractices,
                          ),
                          const SizedBox(height: 16),
                          _buildListCard(
                            title: '🛡️ Prevention Tips',
                            items: _recommendation!.preventiveMeasures,
                          ),
                          const SizedBox(height: 16),
                          _buildWarningBanner(),
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              Expanded(
                                child: ElevatedButton.icon(
                                  onPressed: _onSaveReport,
                                  icon: const Icon(Icons.save),
                                  label: const Text('Save Report'),
                                  style: ElevatedButton.styleFrom(
                                    minimumSize: const Size.fromHeight(50),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: OutlinedButton.icon(
                                  onPressed: _onShare,
                                  icon: const Icon(Icons.share),
                                  label: const Text('Share'),
                                  style: OutlinedButton.styleFrom(
                                    minimumSize: const Size.fromHeight(50),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
    );
  }

  Widget _buildLoading() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Shimmer.fromColors(
        baseColor: Colors.grey.shade700,
        highlightColor: Colors.grey.shade500,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: List.generate(
            5,
            (index) => Container(
              margin: const EdgeInsets.only(bottom: 16),
              height: 120,
              decoration: BoxDecoration(
                color: Colors.grey.shade800,
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildError() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, size: 56, color: Colors.red),
            const SizedBox(height: 12),
            Text(
              _error ?? 'Unknown error',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _fetchRecommendation,
              child: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryCard() {
    final color = _severityColor(_recommendation!.severity);
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _recommendation!.disease,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Chip(
                  label: Text(_recommendation!.severity.toUpperCase()),
                  backgroundColor: color.withOpacity(0.15),
                  labelStyle: TextStyle(color: color, fontWeight: FontWeight.bold),
                ),
                Chip(
                  label: Text('${(_confidence * 100).toStringAsFixed(1)}% confidence'),
                ),
                if (_cropType != null) Chip(label: Text(_cropType!)),
                if (_growthStage != null) Chip(label: Text(_growthStage!)),
                Chip(label: Text('Area: ${_areaHa.toStringAsFixed(1)} ha')),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImmediateActionsCard() {
    return Card(
      color: Theme.of(context).brightness == Brightness.dark
          ? Colors.red.shade900.withOpacity(0.15)
          : Colors.red.shade50,
      elevation: 0,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('⚠️ Do This Now', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
            const SizedBox(height: 10),
            ..._recommendation!.immediateActions.asMap().entries.map((entry) {
              final index = entry.key + 1;
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text('$index. ${entry.value}', style: const TextStyle(fontSize: 14)),
              );
            }),
          ],
        ),
      ),
    );
  }

  Widget _buildTreatmentsCard() {
    return Card(
      child: ExpansionTile(
        initiallyExpanded: true,
        title: const Text('💊 Recommended Treatments', style: TextStyle(fontWeight: FontWeight.bold)),
        children: _recommendation!.fertilizers.isEmpty
            ? [
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    'No chemical sprays are recommended at this risk level. Focus on cultural and preventive measures.',
                    style: TextStyle(color: Theme.of(context).textTheme.bodyMedium?.color),
                  ),
                ),
              ]
            : _recommendation!.fertilizers.map((item) {
                return ExpansionTile(
                  title: Text(item.name, style: const TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: Text(item.type),
                  children: [
                    _buildDetailRow('Dosage', item.dosage),
                    _buildDetailRow('Frequency', item.frequency),
                    _buildDetailRow('Method', item.applicationMethod),
                    _buildDetailRow('Precautions', item.precautions),
                  ],
                );
              }).toList(),
      ),
    );
  }

  Widget _buildOrganicCard() {
    return Card(
      color: Theme.of(context).brightness == Brightness.dark
          ? Colors.green.shade900.withOpacity(0.12)
          : Colors.green.shade50,
      child: ExpansionTile(
        initiallyExpanded: true,
        title: const Text('🌿 Organic Alternatives', style: TextStyle(fontWeight: FontWeight.bold)),
        children: _recommendation!.organicAlternatives.map((item) {
          return ListTile(
            title: Text(item.name, style: const TextStyle(fontWeight: FontWeight.w600)),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 4),
                Text('Dosage: ${item.dosage}'),
                const SizedBox(height: 2),
                Text(item.notes),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildScheduleCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('📅 Spray Schedule', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  ..._recommendation!.followUpSchedule.map((day) => Container(
                        margin: const EdgeInsets.only(right: 10),
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                        decoration: BoxDecoration(
                          color: AppTheme.primary.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(day, style: TextStyle(color: AppTheme.primary, fontWeight: FontWeight.bold)),
                      )),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade200,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text('Recovery in ${_recommendation!.expectedRecoveryDays} days'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildListCard({required String title, required List<String> items}) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            ...items.map((item) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('• ', style: TextStyle(fontSize: 16)),
                      Expanded(child: Text(item, style: const TextStyle(fontSize: 14))),
                    ],
                  ),
                )),
          ],
        ),
      ),
    );
  }

  Widget _buildWarningBanner() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.red.shade200),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded, color: Colors.red),
          const SizedBox(width: 10),
          Expanded(child: Text(_recommendation!.warning, style: const TextStyle(fontSize: 14))),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text('$label:', style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
          const SizedBox(width: 10),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }
}

class Recommendation {
  final String disease;
  final String severity;
  final List<String> immediateActions;
  final List<TreatmentItem> fertilizers;
  final List<OrganicAlternative> organicAlternatives;
  final List<String> culturalPractices;
  final List<String> preventiveMeasures;
  final int expectedRecoveryDays;
  final List<String> followUpSchedule;
  final String warning;

  Recommendation({
    required this.disease,
    required this.severity,
    required this.immediateActions,
    required this.fertilizers,
    required this.organicAlternatives,
    required this.culturalPractices,
    required this.preventiveMeasures,
    required this.expectedRecoveryDays,
    required this.followUpSchedule,
    required this.warning,
  });

  factory Recommendation.fromJson(Map<String, dynamic> json) {
    return Recommendation(
      disease: json['disease']?.toString() ?? '',
      severity: json['severity']?.toString() ?? 'Medium',
      immediateActions: List<String>.from(json['immediate_actions'] ?? []),
      fertilizers: (json['fertilizers'] as List? ?? [])
          .map((e) => TreatmentItem.fromJson(Map<String, dynamic>.from(e)))
          .toList(),
      organicAlternatives: (json['organic_alternatives'] as List? ?? [])
          .map((e) => OrganicAlternative.fromJson(Map<String, dynamic>.from(e)))
          .toList(),
      culturalPractices: List<String>.from(json['cultural_practices'] ?? []),
      preventiveMeasures: List<String>.from(json['preventive_measures'] ?? []),
      expectedRecoveryDays: (json['expected_recovery_days'] ?? 0).toInt(),
      followUpSchedule: List<String>.from(json['follow_up_spray_schedule'] ?? []),
      warning: json['warning']?.toString() ?? '',
    );
  }
}

class TreatmentItem {
  final String name;
  final String type;
  final String dosage;
  final String frequency;
  final String applicationMethod;
  final String precautions;

  TreatmentItem({
    required this.name,
    required this.type,
    required this.dosage,
    required this.frequency,
    required this.applicationMethod,
    required this.precautions,
  });

  factory TreatmentItem.fromJson(Map<String, dynamic> json) => TreatmentItem(
        name: json['name']?.toString() ?? '',
        type: json['type']?.toString() ?? '',
        dosage: json['dosage']?.toString() ?? '',
        frequency: json['frequency']?.toString() ?? '',
        applicationMethod: json['application_method']?.toString() ?? '',
        precautions: json['precautions']?.toString() ?? '',
      );
}

class OrganicAlternative {
  final String name;
  final String dosage;
  final String notes;

  OrganicAlternative({
    required this.name,
    required this.dosage,
    required this.notes,
  });

  factory OrganicAlternative.fromJson(Map<String, dynamic> json) => OrganicAlternative(
        name: json['name']?.toString() ?? '',
        dosage: json['dosage']?.toString() ?? '',
        notes: json['notes']?.toString() ?? '',
      );
}
