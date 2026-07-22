// ══════════════════════════════════════════
// SETTINGS SCREEN
// ══════════════════════════════════════════

// lib/screens/settings_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../core/core.dart';
import '../providers.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});
  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  final _api  = ApiService();
  final _auth = AuthService();
  late TextEditingController _urlCtrl;
  bool _apiOk = false;

  @override
  void initState() {
    super.initState();
    _urlCtrl = TextEditingController();
    _initSettings();
  }

  Future<void> _initSettings() async {
    await _api.ready;
    if (!mounted) return;
    _urlCtrl.text = _api.baseUrl;
    _checkApi();
  }

  Future<void> _checkApi() async {
    final ok = await _api.checkHealth();
    if (!mounted) return;
    setState(() => _apiOk = ok);
  }

  Future<void> _saveUrl() async {
    await _api.setBaseUrl(_urlCtrl.text.trim());
    await _checkApi();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_apiOk ? 'Connected ✓' : 'Cannot reach server')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final locale = ref.watch(localeProvider);
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Language
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Language',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.bold)),
                  const SizedBox(height: 12),
                  ...{
                    'en': 'English',
                    'hi': 'हिन्दी',
                    'mr': 'मराठी',
                  }.entries.map((e) => RadioListTile<String>(
                    dense: true,
                    title: Text(e.value),
                    value: e.key,
                    groupValue: locale.languageCode,
                    onChanged: (v) => ref.read(localeProvider.notifier)
                        .state = Locale(v!),
                  )),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // API URL
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text('API Server',
                          style: Theme.of(context).textTheme.titleSmall
                              ?.copyWith(fontWeight: FontWeight.bold)),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: _apiOk
                              ? AppTheme.riskLow.withOpacity(0.1)
                              : AppTheme.riskHigh.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _apiOk ? 'Connected' : 'Offline',
                          style: TextStyle(
                            color: _apiOk
                                ? AppTheme.riskLow
                                : AppTheme.riskHigh,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _urlCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Server URL',
                      hintText: 'https://xxxx.ngrok.io',
                      isDense: true,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      ElevatedButton(
                          onPressed: _saveUrl, child: const Text('Save & Test')),
                      const SizedBox(width: 8),
                      TextButton(
                        onPressed: () {
                          _urlCtrl.text = 'http://10.0.2.2:8000';
                        },
                        child: const Text('Reset to default'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'For physical device: use ngrok URL\n'
                    'For emulator: use http://10.0.2.2:8000',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.textSecondary),
                  ),
                ],
              ),
            ),
          ),
          // Account
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Account',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  if (_auth.getUser() != null)
                    Text(
                      _auth.getUser()!.email ?? '',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.textSecondary),
                    ),
                  const SizedBox(height: 12),
                  ElevatedButton.icon(
                    onPressed: () async {
                      await _auth.logout();
                      // AuthGate in main.dart handles navigation automatically
                    },
                    icon: const Icon(Icons.logout),
                    label: const Text('Log Out'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.riskHigh,
                      minimumSize: const Size.fromHeight(48),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}