// lib/screens/notifications_settings_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/app_theme.dart';

class NotificationsSettingsScreen extends StatefulWidget {
  const NotificationsSettingsScreen({super.key});

  @override
  State<NotificationsSettingsScreen> createState() =>
      _NotificationsSettingsScreenState();
}

class _NotificationsSettingsScreenState
    extends State<NotificationsSettingsScreen> {
  bool _diseaseAlerts = true;
  bool _weatherAlerts = true;
  bool _weeklyReports = false;
  bool _marketUpdates = false;

  @override
  Widget build(BuildContext context) {
    final th = Theme.of(context);
    final cs = th.colorScheme;

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: cs.onBackground),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('Notification Settings',
            style: GoogleFonts.nunito(
                fontWeight: FontWeight.w800, fontSize: 20)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _tile(context, 'Disease Alerts',
              'Get alerted when a disease is detected',
              _diseaseAlerts, (v) => setState(() => _diseaseAlerts = v)),
          _tile(context, 'Weather Alerts',
              'Receive weather-related crop warnings',
              _weatherAlerts, (v) => setState(() => _weatherAlerts = v)),
          _tile(context, 'Weekly Reports',
              'Summary of your farm activity each week',
              _weeklyReports, (v) => setState(() => _weeklyReports = v)),
          _tile(context, 'Market Updates',
              'Crop market price notifications',
              _marketUpdates, (v) => setState(() => _marketUpdates = v)),
        ],
      ),
    );
  }

  Widget _tile(BuildContext context, String title, String subtitle, bool value,
      ValueChanged<bool> onChanged) {
    final th = Theme.of(context);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: th.cardColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: SwitchListTile(
        activeColor: AppTheme.primary,
        title: Text(title,
            style: GoogleFonts.nunito(
                color: th.colorScheme.onBackground,
                fontWeight: FontWeight.w600,
                fontSize: 14)),
        subtitle: Text(subtitle,
            style: GoogleFonts.nunito(
                color: th.textTheme.bodySmall?.color, fontSize: 12)),
        value: value,
        onChanged: onChanged,
      ),
    );
  }
}
