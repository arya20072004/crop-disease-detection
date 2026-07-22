// lib/widgets/result_widgets.dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:percent_indicator/percent_indicator.dart';
import '../models/models.dart';
import '../core/core.dart';

// Helper — cleans raw disease/class names for display
String _cleanName(String raw) =>
    raw.replaceAll('___', ' — ').replaceAll('_', ' ');

// ══════════════════════════════════════════
// MAIN WRAPPER
// ══════════════════════════════════════════

class AnalysisResultWidget extends StatelessWidget {
  final AnalysisResult result;
  const AnalysisResultWidget({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Quality warnings banner
        if (result.quality != null && result.quality!.warnings.isNotEmpty)
          QualityWarningsBanner(warnings: result.quality!.warnings),
        CnnCard(cnn: result.cnn, gradcam: result.gradcam),
        const SizedBox(height: 12),
        FusionRiskCard(fusion: result.fusion),
        const SizedBox(height: 12),
        if (result.gradcam != null &&
            ((result.gradcam!['infected_pct'] as num?)?.toDouble() ?? 0) > 0) ...[
          DiseaseLocalizationCard(gradcam: result.gradcam!),
          const SizedBox(height: 12),
        ],
        if (result.forecast.top5.isNotEmpty) ...[
          ForecastMiniCard(forecast: result.forecast),
          const SizedBox(height: 12),
        ],
        YieldImpactCard(yieldData: result.yield),
        const SizedBox(height: 12),
        InterventionCard(intervention: result.intervention),
        const SizedBox(height: 12),
        if (result.segmentation != null) ...[
          SegmentationInfoCard(segmentation: result.segmentation!),
          const SizedBox(height: 12),
        ],
        if (result.quality != null) ...[
          QualityScoreCard(quality: result.quality!),
        ],
      ],
    );
  }
}

// ══════════════════════════════════════════
// CNN DIAGNOSIS CARD
// ══════════════════════════════════════════

class CnnCard extends StatefulWidget {
  final CnnResult cnn;
  final Map<String, dynamic>? gradcam;
  const CnnCard({super.key, required this.cnn, this.gradcam});

  @override
  State<CnnCard> createState() => _CnnCardState();
}

class _CnnCardState extends State<CnnCard> {
  Uint8List? _gradcamBytes;

  @override
  void initState() {
    super.initState();
    _decodeGradcam(widget.gradcam?['gradcam']);
  }

  @override
  void didUpdateWidget(CnnCard old) {
    super.didUpdateWidget(old);
    if (old.gradcam?['gradcam'] != widget.gradcam?['gradcam']) {
      _decodeGradcam(widget.gradcam?['gradcam']);
    }
  }

  void _decodeGradcam(String? b64) {
    if (b64 == null || b64.isEmpty) { _gradcamBytes = null; return; }
    try { _gradcamBytes = base64Decode(b64); } catch (_) { _gradcamBytes = null; }
  }

  @override
  Widget build(BuildContext context) {
    final cnn = widget.cnn;
    final isLowConf = cnn.confidence < 0.45;
    final confidenceColor = cnn.confidence < 0.5 ? AppTheme.riskMod : AppTheme.riskLow;
    final l = AppLocalizations.of(context);

    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // ── Banner — disease name only, no duplication below ──
          Container(
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
            decoration: BoxDecoration(
              color: confidenceColor.withOpacity(0.15),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  l?.translate('disease_found') ?? 'Disease Found',
                  style: TextStyle(
                    color: confidenceColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  _cleanName(cnn.detected),
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 20,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${(cnn.confidence * 100).toStringAsFixed(0)}% sure',
                  style: TextStyle(
                    color: AppTheme.textSecondary,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),

          // ── Body — confidence indicator + alternatives only ──
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_gradcamBytes != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.memory(
                      _gradcamBytes!,
                      height: 200,
                      width: double.infinity,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        height: 200,
                        color: Colors.grey.shade100,
                        child: const Center(
                          child: Icon(Icons.broken_image, size: 48, color: Colors.grey),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Red areas show where the problem is strongest.',
                    style: TextStyle(color: AppTheme.textSecondary, fontSize: 12),
                  ),
                  const SizedBox(height: 12),
                ],

                // Confidence ring — no repeated disease name
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            isLowConf
                                ? 'Low confidence — retake in better light'
                                : 'AI confidence level',
                            style: TextStyle(
                              fontSize: 13,
                              color: isLowConf ? AppTheme.riskMod : AppTheme.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                    CircularPercentIndicator(
                      radius: 44,
                      lineWidth: 6,
                      percent: cnn.confidence.clamp(0.0, 1.0),
                      center: Text(
                        '${(cnn.confidence * 100).toStringAsFixed(0)}%\nsure',
                        textAlign: TextAlign.center,
                        style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold),
                      ),
                      progressColor: isLowConf ? AppTheme.riskMod : AppTheme.riskLow,
                      backgroundColor: Colors.grey.shade200,
                    ),
                  ],
                ),

                if (isLowConf) ...[
                  const SizedBox(height: 12),
                  const _WarningBanner('Photo unclear — try again in better light'),
                ],

                if (cnn.top5.isNotEmpty) ...[
                  const SizedBox(height: 14),
                  Text(
                    l?.translate('could_also_be') ?? 'Could also be:',
                    style: TextStyle(color: AppTheme.textSecondary, fontSize: 13),
                  ),
                  const SizedBox(height: 8),
                  ...cnn.top5.skip(1).take(3).map((t) => Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                _cleanName(t.className),
                                style: const TextStyle(fontSize: 13),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            Text(
                              '${(t.prob * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                fontSize: 13,
                                color: AppTheme.riskColor(t.prob),
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      )),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════
// FUSION RISK CARD
// ══════════════════════════════════════════

class FusionRiskCard extends StatelessWidget {
  final FusionResult fusion;
  const FusionRiskCard({super.key, required this.fusion});

  @override
  Widget build(BuildContext context) {
    final color = AppTheme.riskColor(fusion.riskScore);
    final l = AppLocalizations.of(context);

    String plainRiskMessage;
    if (fusion.riskScore > 0.6) {
      plainRiskMessage = l?.translate('act_today') ?? 'Your crop is in danger. Act today.';
    } else if (fusion.riskScore > 0.3) {
      plainRiskMessage = l?.translate('watch_carefully') ?? 'Watch your crop carefully this week.';
    } else {
      plainRiskMessage = l?.translate('crop_healthy') ?? 'Your crop looks healthy. Keep monitoring.';
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardHeader(
              icon: Icons.shield_outlined,
              title: l?.translate('risk_level') ?? 'Risk Level',
            ),
            const SizedBox(height: 12),

            // Top disease banner
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: color.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: color.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _cleanName(fusion.topDisease),
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 17,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    '${(fusion.riskScore * 100).toStringAsFixed(0)}%  [${AppTheme.riskLabelFriendly(fusion.riskScore)}]',
                    style: TextStyle(
                      color: color,
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    plainRiskMessage,
                    style: TextStyle(fontSize: 14, color: color, fontWeight: FontWeight.w500),
                  ),
                ],
              ),
            ),

            if (fusion.top5Diseases.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text(
                'Other diseases to watch:',
                style: TextStyle(color: AppTheme.textSecondary, fontSize: 13),
              ),
              const SizedBox(height: 10),
              ...fusion.top5Diseases.map((d) => Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                _cleanName(d.disease),
                                style: const TextStyle(fontSize: 12),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            Text(
                              '${(d.probability * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                color: AppTheme.riskColor(d.probability),
                                fontWeight: FontWeight.w600,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        LinearProgressIndicator(
                          value: d.probability.clamp(0.0, 1.0),
                          backgroundColor: Colors.grey.shade200,
                          valueColor: AlwaysStoppedAnimation(
                              AppTheme.riskColor(d.probability)),
                          minHeight: 6,
                          borderRadius: BorderRadius.circular(3),
                        ),
                      ],
                    ),
                  )),
            ],
          ],
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════
// 7-DAY FORECAST MINI CARD
// ══════════════════════════════════════════

class ForecastMiniCard extends StatelessWidget {
  final ForecastData forecast;
  const ForecastMiniCard({super.key, required this.forecast});

  @override
  Widget build(BuildContext context) {
    if (forecast.top5.isEmpty) return const SizedBox.shrink();

    final top = forecast.top5.first;
    final spots = top.daily
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value.clamp(0.0, 1.0)))
        .toList();
    final riskColor = AppTheme.riskColor(top.peakRisk);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _CardHeader(icon: Icons.wb_sunny_outlined, title: 'Coming Days Risk'),
            const SizedBox(height: 4),
            Text(
              _cleanName(top.disease),
              style: TextStyle(color: AppTheme.textSecondary, fontSize: 12),
            ),
            const SizedBox(height: 4),
            Text(
              'Risk may rise over the next 7 days',
              style: TextStyle(color: AppTheme.textSecondary, fontSize: 12),
            ),
            const SizedBox(height: 14),
            SizedBox(
              height: 130,
              child: LineChart(
                LineChartData(
                  gridData: FlGridData(
                    show: true,
                    horizontalInterval: 0.25,
                    getDrawingHorizontalLine: (_) =>
                        FlLine(color: Colors.grey.shade200, strokeWidth: 1),
                    drawVerticalLine: false,
                  ),
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        interval: 1,
                        getTitlesWidget: (v, _) {
                          final i = v.toInt();
                          final dates = top.dates.isNotEmpty ? top.dates : forecast.dates;
                          if (i < 0 || i >= dates.length) {
                            return Text('D${i + 1}', style: const TextStyle(fontSize: 9));
                          }
                          return Text(
                            dates[i].length >= 3 ? dates[i].substring(0, 3) : dates[i],
                            style: const TextStyle(fontSize: 9),
                          );
                        },
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        interval: 0.5,
                        reservedSize: 32,
                        getTitlesWidget: (v, _) => Text(
                          '${(v * 100).toInt()}%',
                          style: const TextStyle(fontSize: 9),
                        ),
                      ),
                    ),
                    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  ),
                  borderData: FlBorderData(show: false),
                  minX: 0, maxX: 6, minY: 0, maxY: 1,
                  lineBarsData: [
                    LineChartBarData(
                      spots: spots,
                      isCurved: true,
                      color: riskColor,
                      barWidth: 2.5,
                      dotData: const FlDotData(show: true),
                      belowBarData: BarAreaData(
                        show: true,
                        color: riskColor.withOpacity(0.12),
                      ),
                    ),
                  ],
                  extraLinesData: ExtraLinesData(
                    horizontalLines: [
                      HorizontalLine(
                        y: 0.6,
                        color: Colors.red.withOpacity(0.4),
                        strokeWidth: 1,
                        dashArray: [4, 4],
                        label: HorizontalLineLabel(
                          show: true,
                          labelResolver: (_) => 'High',
                          style: const TextStyle(fontSize: 9, color: Colors.red),
                        ),
                      ),
                      HorizontalLine(
                        y: 0.3,
                        color: Colors.orange.withOpacity(0.4),
                        strokeWidth: 1,
                        dashArray: [4, 4],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════
// YIELD IMPACT CARD
// ══════════════════════════════════════════

class YieldImpactCard extends StatelessWidget {
  final YieldData yieldData;
  const YieldImpactCard({super.key, required this.yieldData});

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardHeader(
              icon: Icons.currency_rupee,
              title: l?.translate('money_impact') ?? 'Money Impact',
              iconColor: AppTheme.riskHigh,
            ),
            const Divider(height: 20),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Text.rich(
                TextSpan(children: [
                  const TextSpan(text: 'If untreated, you may lose '),
                  TextSpan(
                    text: '₹${yieldData.financialLoss.toStringAsFixed(0)}',
                    style: const TextStyle(fontWeight: FontWeight.bold, color: AppTheme.riskHigh),
                  ),
                ]),
                style: const TextStyle(fontSize: 14),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Text.rich(
                TextSpan(children: [
                  const TextSpan(text: 'Treatment costs about '),
                  TextSpan(
                    text: '₹${yieldData.treatmentCost.toStringAsFixed(0)}',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                ]),
                style: const TextStyle(fontSize: 14),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Text.rich(
                TextSpan(children: [
                  const TextSpan(text: 'By treating, you save '),
                  TextSpan(
                    text: '₹${yieldData.savedValue.toStringAsFixed(0)}',
                    style: const TextStyle(fontWeight: FontWeight.bold, color: AppTheme.riskLow),
                  ),
                ]),
                style: const TextStyle(fontSize: 14),
              ),
            ),
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppTheme.riskLow.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                'Every ₹1 you spend protects ₹${yieldData.roi.toStringAsFixed(1)} in crops',
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppTheme.riskLow,
                  fontSize: 14,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════
// INTERVENTION CARD
// ══════════════════════════════════════════

class InterventionCard extends StatelessWidget {
  final InterventionData intervention;
  const InterventionCard({super.key, required this.intervention});

  @override
  Widget build(BuildContext context) {
    final color = intervention.isCritical
        ? AppTheme.riskHigh
        : intervention.isModerate
            ? AppTheme.riskMod
            : AppTheme.riskLow;
    final stepColor = AppTheme.stepColor(intervention.urgency);
    final l = AppLocalizations.of(context);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardHeader(
              icon: Icons.task_alt,
              title: l?.translate('what_to_do') ?? 'What To Do',
              iconColor: color,
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
              decoration: BoxDecoration(
                color: color.withOpacity(0.08),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: color.withOpacity(0.4), width: 1.5),
              ),
              child: Text(
                intervention.urgency,
                style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 15),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 16),
            _InterventionStep(
              step: 1,
              color: stepColor,
              title: 'Spray ${intervention.fungicide}',
              subtitle: 'Mix well and spray on all leaves',
            ),
            const Divider(height: 24),
            _InterventionStep(
              step: 2,
              color: stepColor,
              title: 'Repeat: ${intervention.frequency}',
              subtitle: 'Continue until leaves look healthy',
            ),
            const Divider(height: 24),
            _InterventionStep(
              step: 3,
              color: stepColor,
              title: 'Best time: ${intervention.timing}',
              subtitle: 'Remove badly infected leaves and burn them',
            ),
          ],
        ),
      ),
    );
  }
}

class _InterventionStep extends StatelessWidget {
  final int step;
  final Color color;
  final String title;
  final String subtitle;

  const _InterventionStep({
    required this.step,
    required this.color,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 28,
          height: 28,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          alignment: Alignment.center,
          child: Text(
            '$step',
            style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text(subtitle, style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary)),
            ],
          ),
        ),
      ],
    );
  }
}

// ══════════════════════════════════════════
// QUALITY WARNINGS BANNER
// ══════════════════════════════════════════

class QualityWarningsBanner extends StatelessWidget {
  final List<String> warnings;
  const QualityWarningsBanner({super.key, required this.warnings});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(8),
        border: const Border(left: BorderSide(color: Colors.amber, width: 4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.warning_amber, size: 16, color: Colors.amber),
              const SizedBox(width: 6),
              Text('Image Quality Warnings', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 8),
          ...warnings.map((w) => Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Text('• $w', style: const TextStyle(fontSize: 12)),
          )),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════
// DISEASE LOCALIZATION CARD
// ══════════════════════════════════════════

class DiseaseLocalizationCard extends StatefulWidget {
  final Map<String, dynamic> gradcam;
  const DiseaseLocalizationCard({super.key, required this.gradcam});

  @override
  State<DiseaseLocalizationCard> createState() => _DiseaseLocalizationCardState();
}

class _DiseaseLocalizationCardState extends State<DiseaseLocalizationCard> {
  Uint8List? _imageBytes;

  @override
  void initState() {
    super.initState();
    _decodeImage(widget.gradcam['gradcam']);
  }

  @override
  void didUpdateWidget(DiseaseLocalizationCard old) {
    super.didUpdateWidget(old);
    if (old.gradcam['gradcam'] != widget.gradcam['gradcam']) {
      _decodeImage(widget.gradcam['gradcam']);
    }
  }

  void _decodeImage(String? b64) {
    if (b64 == null || b64.isEmpty) { _imageBytes = null; return; }
    try { _imageBytes = base64Decode(b64); } catch (_) { _imageBytes = null; }
  }

  Color _severityColor(String severity) {
    switch (severity.toLowerCase()) {
      case 'low':      return Colors.green;
      case 'moderate': return Colors.orange;
      case 'high':     return Colors.deepOrange;
      case 'severe':   return Colors.red;
      default:         return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    final gradcam        = widget.gradcam;
    final infectedPct    = (gradcam['infected_pct'] as num?)?.toDouble() ?? 0.0;
    final severity       = gradcam['severity'] as String? ?? 'Unknown';
    final spotCount      = (gradcam['spot_count'] as num?)?.toInt() ?? 0;
    final List spots     = gradcam['spots'] ?? [];
    final String? gradcamWarning = gradcam['warning'] as String?;
    final sevColor       = _severityColor(severity);

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(children: [
              Icon(Icons.biotech_outlined, color: Colors.green),
              const SizedBox(width: 8),
              Text('Disease Localisation',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            ]),
            const Divider(height: 20),

            // Annotated image
            if (_imageBytes != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.memory(
                  _imageBytes!,
                  width: double.infinity,
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => Container(
                    height: 150,
                    color: Colors.grey.shade100,
                    child: const Center(
                      child: Icon(Icons.broken_image, size: 48, color: Colors.grey),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 14),
            ],

            // Infected area + severity badge in one row
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Infected Leaf Area',
                          style: TextStyle(fontSize: 12, color: Colors.grey)),
                      const SizedBox(height: 4),
                      Text('${infectedPct.toStringAsFixed(1)}%',
                          style: TextStyle(
                              fontSize: 28,
                              fontWeight: FontWeight.bold,
                              color: sevColor)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: sevColor.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: sevColor.withOpacity(0.4)),
                  ),
                  child: Text(severity,
                      style: TextStyle(
                          fontWeight: FontWeight.w700,
                          color: sevColor,
                          fontSize: 14)),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Spot count
            Row(children: [
              Icon(Icons.location_on_outlined, size: 16, color: Colors.grey),
              const SizedBox(width: 4),
              Text('$spotCount disease region(s) detected',
                  style: TextStyle(fontSize: 13, color: Colors.grey.shade700)),
            ]),

            // Grad-CAM warning if present
            if (gradcamWarning != null) ...[
              const SizedBox(height: 10),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.amber.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.amber.shade200),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, size: 16, color: Colors.amber.shade700),
                    const SizedBox(width: 6),
                    Expanded(child: Text(gradcamWarning,
                        style: TextStyle(fontSize: 12,
                            color: Colors.amber.shade800))),
                  ],
                ),
              ),
            ],

            // Individual spots list — only show if > 1 spot
            if (spots.length > 1) ...[
              const SizedBox(height: 12),
              Text('Affected regions:',
                  style: const TextStyle(
                      fontWeight: FontWeight.w600, fontSize: 13)),
              const SizedBox(height: 6),
              ...spots.take(5).map((spot) => Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(children: [
                  Container(
                    width: 20, height: 20,
                    decoration: BoxDecoration(
                      color: Colors.red.shade400,
                      shape: BoxShape.circle,
                    ),
                    child: Center(
                      child: Text('${spot['id']}',
                          style: const TextStyle(color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.bold)),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text('Region ${spot['id']}: '
                      '${(spot['area_pct_leaf'] as num).toStringAsFixed(1)}% of leaf',
                      style: const TextStyle(fontSize: 12)),
                ]),
              )),
              if (spots.length > 5)
                Text('+ ${spots.length - 5} more regions',
                    style: TextStyle(fontSize: 11, color: Colors.grey)),
            ],
          ],
        ),
      ),
    );
  }
}


// ══════════════════════════════════════════
// SEGMENTATION INFO CARD
// ══════════════════════════════════════════

class SegmentationInfoCard extends StatelessWidget {
  final SegmentationData segmentation;
  const SegmentationInfoCard({super.key, required this.segmentation});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardHeader(
              icon: Icons.cut,
              title: 'Leaf Segmentation',
              iconColor: AppTheme.primary,
            ),
            const Divider(height: 20),
            Text('Method: ${segmentation.method}', style: const TextStyle(fontSize: 14)),
            Text('Leaf Coverage: ${(segmentation.leafCoverage * 100).toStringAsFixed(1)}%', style: const TextStyle(fontSize: 14)),
            if (segmentation.warning != null) ...[
              const SizedBox(height: 8),
              _WarningBanner(segmentation.warning!),
            ],
          ],
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════
// QUALITY SCORE CARD
// ══════════════════════════════════════════

class QualityScoreCard extends StatelessWidget {
  final QualityData quality;
  const QualityScoreCard({super.key, required this.quality});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardHeader(
              icon: Icons.photo_camera,
              title: 'Image Quality',
              iconColor: AppTheme.primary,
            ),
            const Divider(height: 20),
            Text('Score: ${quality.score.toStringAsFixed(1)}/100', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('Metrics:', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            ...quality.metrics.entries.map((e) => Padding(
              padding: const EdgeInsets.only(bottom: 2),
              child: Text('${e.key}: ${e.value}', style: const TextStyle(fontSize: 12)),
            )),
          ],
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════
// SHARED SMALL WIDGETS
// ══════════════════════════════════════════

class _CardHeader extends StatelessWidget {
  final IconData icon;
  final String title;
  final Color? iconColor;

  const _CardHeader({required this.icon, required this.title, this.iconColor});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, color: iconColor ?? AppTheme.primary, size: 22),
        const SizedBox(width: 8),
        Text(
          title,
          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
        ),
      ],
    );
  }
}

class _WarningBanner extends StatelessWidget {
  final String message;
  const _WarningBanner(this.message);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(8),
        border: const Border(left: BorderSide(color: Colors.amber, width: 4)),
      ),
      child: Row(
        children: [
          const Icon(Icons.warning_amber, size: 16, color: Colors.amber),
          const SizedBox(width: 6),
          Expanded(child: Text(message, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }
}