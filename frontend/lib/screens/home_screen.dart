// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import '../core/core.dart';
import '../services/db_service.dart';

class HomeScreen extends StatefulWidget {
  final void Function(int)? onTabSwitch;
  const HomeScreen({super.key, this.onTabSwitch});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _db = DBService();
  bool _loading = true;
  Map<String, dynamic>? _latestPrediction;
  Map<String, dynamic>? _latestForecast;
  int _unreadAlerts = 0;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _loading = true);
    try {
      final predictions = await _db.getPredictions();
      final forecasts = await _db.getLatestForecasts();
      final unreadAlerts = await _db.getUnreadAlertsCount();

      if (!mounted) return;
      setState(() {
        if (predictions.isNotEmpty) {
          // Check if it's within 48 hours
          final createdAt = DateTime.parse(predictions.first['created_at']);
          if (DateTime.now().difference(createdAt).inHours <= 48) {
            _latestPrediction = predictions.first;
          }
        }
        if (forecasts.isNotEmpty) {
          _latestForecast = forecasts.first;
        }
        _unreadAlerts = unreadAlerts;
        _loading = false;
      });
    } catch (e) {
      print('[HomeScreen] _loadData error: $e');
      if (mounted) setState(() => _loading = false);
    }
  }

  String _cleanName(String raw) =>
      raw.replaceAll('___', ' — ').replaceAll('_', ' ');

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l?.translate('app_title') ?? 'Fasal Saarthi'),
        actions: [
          Stack(
            alignment: Alignment.center,
            children: [
              IconButton(
                icon: const Icon(Icons.notifications_outlined),
                onPressed: () {
                  // Navigate to notifications screen
                },
              ),
              if (_unreadAlerts > 0)
                Positioned(
                  right: 11,
                  top: 11,
                  child: Container(
                    padding: const EdgeInsets.all(2),
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    constraints: const BoxConstraints(
                      minWidth: 14,
                      minHeight: 14,
                    ),
                    child: Text(
                      '$_unreadAlerts',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                )
            ],
          ),
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _loadData,
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Live Risk Card or Fallback
                    _latestPrediction != null
                        ? _buildLiveRiskCard(context, l)
                        : _buildFallbackCard(context, l),
                    const SizedBox(height: 24),

                    // 7-Day Risk Strip
                    if (_latestForecast != null)
                      _buildForecastStrip(context, l),
                    if (_latestForecast != null) const SizedBox(height: 24),

                    Text(
                      l?.translate('what_to_do') ?? 'Quick Actions',
                      style: theme.textTheme.titleMedium
                          ?.copyWith(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 12),

                    GridView.count(
                      crossAxisCount: 2,
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      crossAxisSpacing: 12,
                      mainAxisSpacing: 12,
                      childAspectRatio: 1.1,
                      children: [
                        _QuickActionCard(
                          icon: Icons.camera_alt,
                          label: l?.translate('scan_leaf') ?? 'Scan a Leaf',
                          color: AppTheme.primary,
                          onTap: () => widget.onTabSwitch?.call(1),
                        ),
                        _QuickActionCard(
                          icon: Icons.wb_sunny,
                          label: l?.translate('next_7_days') ?? 'Next 7 Days',
                          color: const Color(0xFF1976D2),
                          onTap: () => widget.onTabSwitch?.call(2),
                        ),
                        _QuickActionCard(
                          icon: Icons.assignment,
                          label: 'Treatment Log',
                          color: const Color(0xFF7B1FA2),
                          onTap: () => widget.onTabSwitch?.call(3),
                        ),
                        _QuickActionCard(
                          icon: Icons.timeline,
                          label: l?.translate('past_risks') ?? 'Past Results',
                          color: const Color(0xFFE64A19),
                          onTap: () => widget.onTabSwitch?.call(3),
                        ),
                      ],
                    ),

                    const SizedBox(height: 24),

                    Text(
                      l?.translate('photo_tips_title') ?? 'How to take a good photo',
                      style: theme.textTheme.titleMedium
                          ?.copyWith(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 12),

                    ...[
                      ('📸', l?.translate('tip_1') ?? 'Get close to one leaf — fill the screen with it'),
                      ('☀️', l?.translate('tip_2') ?? 'Use natural daylight — avoid shade'),
                      ('🤲', l?.translate('tip_3') ?? 'Hold the phone steady — no blur'),
                      ('📍', l?.translate('tip_4') ?? 'Allow location — needed for weather data'),
                    ].map((tip) => Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(tip.$1, style: const TextStyle(fontSize: 20)),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(tip.$2,
                                    style: theme.textTheme.bodyMedium),
                              ),
                            ],
                          ),
                        )),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildLiveRiskCard(BuildContext context, AppLocalizations? l) {
    final risk = (_latestPrediction!['risk'] as num?)?.toDouble() ??
        (_latestPrediction!['risk_score'] as num?)?.toDouble() ??
        0.0;
    final disease = _latestPrediction!['disease'] as String? ??
        _latestPrediction!['top_disease'] as String? ??
        'Unknown';
    final cropName = _latestPrediction!['crop_name'] as String? ?? 'Your Crop';
    final color = AppTheme.riskColor(risk);
    final riskLevel = risk > 0.6 ? 'HIGH RISK' : risk > 0.3 ? 'MODERATE RISK' : 'LOW RISK';

    return GestureDetector(
      onTap: () => widget.onTabSwitch?.call(3), // Navigate to History
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.5), width: 2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  cropName,
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    risk > 0.6 ? 'TREAT TODAY' : 'MONITOR',
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold),
                  ),
                )
              ],
            ),
            const SizedBox(height: 12),
            Text(
              _cleanName(disease),
              style: TextStyle(
                color: Theme.of(context).colorScheme.onBackground,
                fontWeight: FontWeight.bold,
                fontSize: 22,
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Text(
                  '${(risk * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.w900,
                    fontSize: 32,
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  riskLevel,
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget _buildFallbackCard(BuildContext context, AppLocalizations? l) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.primary.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primary.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  l?.translate('hero_title') ?? 'Is your crop sick?',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onBackground,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  l?.translate('hero_subtitle') ??
                      'Scan a leaf to see your crop\'s risk',
                  style: TextStyle(
                      color: Theme.of(context).textTheme.bodySmall?.color,
                      fontSize: 14),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Icon(Icons.camera_alt, color: AppTheme.primary, size: 40),
        ],
      ),
    );
  }

  Widget _buildForecastStrip(BuildContext context, AppLocalizations? l) {
    final dailyRisk = _latestForecast!['daily_risk'] as List<dynamic>? ?? [];
    if (dailyRisk.isEmpty) return const SizedBox.shrink();

    final disease = _cleanName(_latestForecast!['disease'] as String? ?? '');

    return GestureDetector(
      onTap: () => widget.onTabSwitch?.call(2), // Navigate to Forecast
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '7-Day Risk Forecast',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold, color: Colors.grey[700]),
              ),
              const Icon(Icons.arrow_forward_ios, size: 12, color: Colors.grey)
            ],
          ),
          const SizedBox(height: 4),
          Text(
            disease,
            style: const TextStyle(fontSize: 12, color: Colors.grey),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: List.generate(
              dailyRisk.length > 7 ? 7 : dailyRisk.length,
              (index) {
                final risk = (dailyRisk[index] as num).toDouble();
                return Container(
                  width: MediaQuery.of(context).size.width / 8 - 4,
                  height: 36,
                  decoration: BoxDecoration(
                    color: AppTheme.riskColor(risk).withOpacity(0.8),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Center(
                    child: Text(
                      '${(risk * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                );
              },
            ),
          )
        ],
      ),
    );
  }
}

class _QuickActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _QuickActionCard({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: color, size: 36),
            const SizedBox(height: 8),
            Text(
              label,
              textAlign: TextAlign.center,
              style: TextStyle(
                  color: color, fontWeight: FontWeight.w700, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}