// lib/screens/app_preferences_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/app_theme.dart';

class AppPreferencesScreen extends StatefulWidget {
  const AppPreferencesScreen({super.key});

  @override
  State<AppPreferencesScreen> createState() => _AppPreferencesScreenState();
}

class _AppPreferencesScreenState extends State<AppPreferencesScreen> {
  String _units       = 'Metric';
  String _tempUnit    = '°C';
  bool   _autoSync    = true;
  bool   _offlineMode = false;

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
        title: Text('App Preferences',
            style: GoogleFonts.nunito(
                fontWeight: FontWeight.w800, fontSize: 20)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _sectionLabel(context, 'Measurement'),
          _dropdownTile(context, 'Units', 'Area & distance units', _units,
              ['Metric', 'Imperial'],
              (v) => setState(() => _units = v!)),
          _dropdownTile(context, 'Temperature', 'Display temperature in',
              _tempUnit, ['°C', '°F'],
              (v) => setState(() => _tempUnit = v!)),
          const SizedBox(height: 16),
          _sectionLabel(context, 'Connectivity'),
          _switchTile(context, 'Auto Sync',
              'Sync data automatically in background',
              _autoSync, (v) => setState(() => _autoSync = v)),
          _switchTile(context, 'Offline Mode',
              'Work without internet connection',
              _offlineMode, (v) => setState(() => _offlineMode = v)),
        ],
      ),
    );
  }

  Widget _sectionLabel(BuildContext context, String text) => Padding(
        padding: const EdgeInsets.only(bottom: 8, left: 4),
        child: Text(text,
            style: GoogleFonts.nunito(
                color: Theme.of(context).textTheme.bodySmall?.color,
                fontSize: 12,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.8)),
      );

  Widget _dropdownTile(BuildContext context, String title, String subtitle,
      String value, List<String> options, ValueChanged<String?> onChanged) {
    final th = Theme.of(context);
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: th.cardColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: GoogleFonts.nunito(
                        color: th.colorScheme.onBackground,
                        fontWeight: FontWeight.w600,
                        fontSize: 14)),
                Text(subtitle,
                    style: GoogleFonts.nunito(
                        color: th.textTheme.bodySmall?.color, fontSize: 12)),
              ],
            ),
          ),
          DropdownButton<String>(
            value: value,
            dropdownColor: th.cardColor,
            underline: const SizedBox(),
            style: GoogleFonts.nunito(
                color: th.colorScheme.onBackground, fontSize: 13),
            items: options
                .map((o) => DropdownMenuItem(value: o, child: Text(o)))
                .toList(),
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  Widget _switchTile(BuildContext context, String title, String subtitle,
      bool value, ValueChanged<bool> onChanged) {
    final th = Theme.of(context);
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
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
