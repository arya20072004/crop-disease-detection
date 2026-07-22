// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import '../core/core.dart';

class HomeScreen extends StatelessWidget {
  final void Function(int)? onTabSwitch;
  const HomeScreen({super.key, this.onTabSwitch});

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(l?.translate('app_title') ?? 'Crop Disease EWS'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Hero card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AppTheme.primaryDark,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          l?.translate('hero_title') ?? 'Is your crop sick?',
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 24,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          l?.translate('hero_subtitle') ??
                              'Point your phone at a sick leaf to get help',
                          style: const TextStyle(color: Colors.white70, fontSize: 15),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  const Icon(Icons.eco, color: Colors.white, size: 48),
                ],
              ),
            ),
            const SizedBox(height: 24),

            Text(
              l?.translate('what_to_do') ?? 'What do you want to do?',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
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
                  onTap: () => onTabSwitch?.call(1),
                ),
                _QuickActionCard(
                  icon: Icons.wb_sunny,
                  label: l?.translate('next_7_days') ?? 'Next 7 Days',
                  color: const Color(0xFF1976D2),
                  onTap: () => onTabSwitch?.call(2),
                ),
                _QuickActionCard(
                  icon: Icons.compare_arrows,
                  label: l?.translate('compare') ?? 'Compare Crops',
                  color: const Color(0xFF7B1FA2),
                  onTap: () => onTabSwitch?.call(3),
                ),
                _QuickActionCard(
                  icon: Icons.timeline,
                  label: l?.translate('past_risks') ?? 'Past Results',
                  color: const Color(0xFFE64A19),
                  onTap: () => onTabSwitch?.call(4),
                ),
              ],
            ),

            const SizedBox(height: 24),

            Text(
              l?.translate('photo_tips_title') ?? 'How to take a good photo',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
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
                    child: Text(tip.$2, style: Theme.of(context).textTheme.bodyMedium),
                  ),
                ],
              ),
            )),
          ],
        ),
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
              style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}