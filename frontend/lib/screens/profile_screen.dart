// ══════════════════════════════════════════
// PROFILE SCREEN
// ══════════════════════════════════════════

// lib/screens/profile_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart' as provider;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../config/api_config.dart';
import '../core/app_theme.dart';
import '../providers.dart';
import '../providers/theme_provider.dart';
import 'edit_profile_screen.dart';
import 'notifications_settings_screen.dart';
import 'privacy_screen.dart';
import 'app_preferences_screen.dart';
import 'help_screen.dart';

final _sb = Supabase.instance.client;

class ProfileScreen extends ConsumerStatefulWidget {
  const ProfileScreen({super.key});

  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  bool   _isLoading = true;

  // Profile data
  String _userName       = '';
  String _email      = '';
  String _phone      = 'Not provided';
  String _joinDate   = '—';
  String _role       = 'Farmer';
  int    _fieldsManaged = 0;

  // Stats
  int    _imagesAnalyzed   = 0;
  int    _issuesResolved   = 0;
  int    _reportsGenerated = 0;
  String _avgConfidence    = '0%';

  // Settings
  String _language      = 'English';
  bool   _notifications = true;

  // API Server configuration
  late final TextEditingController _apiUrlController;
  String _apiHealthStatus = 'Checking...';
  bool _apiConnected = false;

  @override
  void initState() {
    super.initState();
    _apiUrlController = TextEditingController();
    _loadPrefs();
    _loadApiUrl();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadProfileData();
    });
  }

  @override
  void dispose() {
    _apiUrlController.dispose();
    super.dispose();
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _language      = prefs.getString('language')    ?? 'English';
      _notifications = prefs.getBool('notifications') ?? true;
    });
  }

  Future<void> _saveLanguage(String lang) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('language', lang);
    const localeMap = {'English': 'en', 'हिंदी': 'hi', 'मराठी': 'mr'};
    ref.read(localeProvider.notifier).state = Locale(localeMap[lang] ?? 'en');
    setState(() => _language = lang);
  }

  Future<void> _loadApiUrl() async {
    try {
      final url = await ApiConfig.getBaseUrl();
      if (mounted) {
        setState(() => _apiUrlController.text = url);
      }
    } catch (e) {
      debugPrint('[ProfileScreen] Error loading API URL: $e');
    }
  }

  Future<void> _saveApiUrl(String url) async {
    try {
      await ApiConfig.setBaseUrl(url.trim());
      if (mounted) {
        setState(() => _apiHealthStatus = 'Saved!');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('API URL saved successfully!'),
            backgroundColor: AppTheme.riskLow,
            duration: Duration(seconds: 2),
          ),
        );
        // Reset status after 2 seconds
        await Future.delayed(const Duration(seconds: 2));
        if (mounted) {
          setState(() => _apiHealthStatus = 'Checking...');
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: AppTheme.riskHigh,
          ),
        );
      }
    }
  }

  Future<void> _saveNotifications(bool v) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('notifications', v);
    setState(() => _notifications = v);
  }

  Future<void> _loadProfileData() async {
    final user = _sb.auth.currentUser;
    if (user == null) {
      if (mounted) Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    try {
      // Run all queries in parallel
      final results = await Future.wait([
        // Query 1: profile (only columns that exist: name, created_at)
        _sb
          .from('profiles')
          .select('name, created_at')
          .eq('id', user.id)
          .single()
          .catchError((e) {
            debugPrint('[ProfileScreen] profile query error: $e');
            return {'name': 'User', 'created_at': null};
          }),

        // Query 2: crops count
        _sb
          .from('crops')
          .select('id')
          .eq('user_id', user.id)
          .catchError((e) {
            debugPrint('[ProfileScreen] crops query error: $e');
            return [];
          }),

        // Query 3: predictions
        _sb
          .from('predictions')
          .select('confidence')
          .eq('user_id', user.id)
          .catchError((e) {
            debugPrint('[ProfileScreen] predictions query error: $e');
            return [];
          }),

        // Query 4: prediction history
        _sb
          .from('prediction_history')
          .select('id')
          .eq('user_id', user.id)
          .catchError((e) {
            debugPrint('[ProfileScreen] history query error: $e');
            return [];
          }),
      ]);

      final profileData = results[0] as Map<String, dynamic>;
      final cropsData = results[1] as List;
      final predictionsData = results[2] as List;
      final historyData = results[3] as List;

      // Safe confidence average
      double avgConf = 0.0;
      if (predictionsData.isNotEmpty) {
        final total = predictionsData
          .map((p) => ((p['confidence'] ?? 0.0) as num).toDouble())
          .reduce((a, b) => a + b);
        avgConf = total / predictionsData.length;
      }

      if (mounted) {
        setState(() {
          _userName = profileData['name'] ?? 'User';
          _joinDate = profileData['created_at'] != null
            ? DateFormat('dd/MM/yyyy')
              .format(DateTime.parse(profileData['created_at']))
            : 'Not available';
          _phone = 'Not provided';  // phone column doesn't exist yet
          _email = user.email ?? '';
          _fieldsManaged = cropsData.length;
          _imagesAnalyzed = predictionsData.length;
          _reportsGenerated = historyData.length;
          _avgConfidence = '${avgConf.toStringAsFixed(0)}%';
          _role = 'Farmer';
          _issuesResolved = 0;
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint('[ProfileScreen] _loadProfileData fatal error: $e');
      if (mounted) {
        setState(() {
          _userName = 'User';
          _email = _sb.auth.currentUser?.email ?? '';
          _phone = 'Not provided';
          _joinDate = 'Not available';
          _fieldsManaged = 0;
          _imagesAnalyzed = 0;
          _reportsGenerated = 0;
          _avgConfidence = '0%';
          _role = 'Farmer';
          _issuesResolved = 0;
          _isLoading = false;
        });
      }
    }
  }

  String get _initials {
    final t = _userName.trim();
    if (t.isEmpty) {
      final lp = _email.split('@').first;
      return lp.isEmpty ? '?' : lp[0].toUpperCase();
    }
    final parts = t.split(RegExp(r'\s+'));
    if (parts.length == 1) return parts[0][0].toUpperCase();
    return '${parts[0][0]}${parts.last[0]}'.toUpperCase();
  }

  Future<void> _logout() async {
    final cs = Theme.of(context).colorScheme;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Logout',
            style: GoogleFonts.nunito(fontWeight: FontWeight.w800)),
        content: Text('Are you sure you want to logout?',
            style: GoogleFonts.nunito()),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.pop(ctx, true),
            child: Text('Logout',
                style: GoogleFonts.nunito(color: Colors.white)),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      await _sb.auth.signOut();
    }
  }

  // ── Build ──────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final th = Theme.of(context);
    final cs = th.colorScheme;

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: cs.onBackground),
          onPressed: () => Navigator.maybePop(context),
        ),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Profile',
                style: GoogleFonts.nunito(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: cs.onBackground)),
            Text('Manage your account settings',
                style: GoogleFonts.nunito(
                    fontSize: 12, color: th.textTheme.bodySmall?.color)),
          ],
        ),
        toolbarHeight: 70,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _darkCard(context, child: _buildProfileCard(context)),
            const SizedBox(height: 16),
            _darkCard(context, child: _buildSettingsCard(context)),
            const SizedBox(height: 16),
            _darkCard(context, child: _buildAccountDetailsCard(context)),
            const SizedBox(height: 16),
            _darkCard(context, child: _buildUsageStatsCard(context)),
            const SizedBox(height: 16),
            _darkCard(context, padding: EdgeInsets.zero,
                child: _buildBottomCard(context)),
            const SizedBox(height: 24),
            Center(
              child: Text('Fasal Saarthi v1.0.0',
                  style: GoogleFonts.nunito(
                      fontSize: 12,
                      color: th.textTheme.bodySmall?.color)),
            ),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );
  }

  // ── Sections ───────────────────────────────────────────────────────────────

  Widget _buildProfileCard(BuildContext context) {
    final th = Theme.of(context);
    final cs = th.colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CircleAvatar(
              radius: 32,
              backgroundColor: Colors.grey[700],
              child: _isLoading
                  ? const SizedBox(
                      width: 22, height: 22,
                      child: CircularProgressIndicator(
                          color: Colors.white, strokeWidth: 2))
                  : Text(_initials,
                      style: GoogleFonts.nunito(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: Colors.white)),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _isLoading
                      ? _shimmerLine(120, 18, context)
                      : Text(
                          _userName.isEmpty ? _email.split('@').first : _userName,
                          style: GoogleFonts.nunito(
                              fontSize: 18,
                              fontWeight: FontWeight.w800,
                              color: cs.onBackground)),
                  const SizedBox(height: 4),
                  _isLoading
                      ? _shimmerLine(80, 13, context)
                      : Text(_role,
                          style: GoogleFonts.nunito(
                              fontSize: 12,
                              color: th.textTheme.bodySmall?.color)),
                ],
              ),
            ),
            IconButton(
              icon: Icon(Icons.edit_outlined,
                  color: th.textTheme.bodyMedium?.color),
              onPressed: () => Navigator.push(context,
                  MaterialPageRoute(
                      builder: (_) => const EditProfileScreen())),
              tooltip: 'Edit Profile',
            ),
          ],
        ),
        const SizedBox(height: 16),
        Divider(color: th.dividerColor, height: 1),
        const SizedBox(height: 16),
        if (_isLoading)
          Center(
              child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: CircularProgressIndicator(color: cs.primary),
          ))
        else
          GridView.count(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
            childAspectRatio: 2.4,
            children: [
              _infoTile(context, 'Email',
                  _email.isEmpty ? '—' : _email, Icons.email_outlined),
              _infoTile(context, 'Phone', _phone, Icons.phone_outlined),
              _infoTile(context, 'Join Date', _joinDate,
                  Icons.calendar_today_outlined),
              _infoTile(context, 'Fields Managed', '$_fieldsManaged Active',
                  Icons.grass_outlined),
            ],
          ),
      ],
    );
  }

  Widget _buildSettingsCard(BuildContext context) {
    final th          = Theme.of(context);
    final themeProvider = context.read<ThemeProvider>();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('Settings', context),
        // ── API Server URL setting ────────────────────────────────────────
        _settingsRow(
          context: context,
          icon: Icons.dns_rounded,
          label: 'API Server URL',
          subtitle: 'Configure FastAPI backend',
          trailing: SizedBox(
            width: 120,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: _apiConnected
                    ? AppTheme.riskLow.withOpacity(0.1)
                    : AppTheme.riskHigh.withOpacity(0.1),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                _apiHealthStatus,
                style: TextStyle(
                  color: _apiConnected ? AppTheme.riskLow : AppTheme.riskHigh,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 0, vertical: 8),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _apiUrlController,
                  style: TextStyle(
                    color: th.colorScheme.onBackground,
                    fontSize: 12,
                  ),
                  decoration: InputDecoration(
                    hintText: 'http://192.168.x.x:8000',
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 72,
                child: ElevatedButton(
                  onPressed: () => _saveApiUrl(_apiUrlController.text),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: const Text('Save', style: TextStyle(fontSize: 13)),
                ),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Text(
            'Enter your FastAPI server IP (not localhost on physical device)',
            style: TextStyle(
              color: Colors.grey,
              fontSize: 11,
            ),
          ),
        ),
        Divider(color: th.dividerColor, height: 1),
        // ── Language setting ──────────────────────────────────────────────
        _settingsRow(
          context: context,
          icon: Icons.language,
          label: 'Language',
          subtitle: 'Choose your preferred language',
          trailing: DropdownButton<String>(
            value: _language,
            dropdownColor: th.cardColor,
            underline: const SizedBox(),
            style: GoogleFonts.nunito(
                fontSize: 13,
                color: th.colorScheme.onBackground),
            items: const [
              DropdownMenuItem(value: 'English', child: Text('English')),
              DropdownMenuItem(value: 'हिंदी',   child: Text('हिंदी')),
              DropdownMenuItem(value: 'मराठी',   child: Text('मराठी')),
            ],
            onChanged: (v) { if (v != null) _saveLanguage(v); },
          ),
        ),
        Divider(color: th.dividerColor, height: 1),
        _settingsRow(
          context: context,
          icon: Icons.notifications_outlined,
          label: 'Notifications',
          subtitle: 'Receive alerts and updates',
          trailing: Switch(
              value: _notifications,
              onChanged: _saveNotifications,
              activeColor: AppTheme.primary),
        ),
        Divider(color: th.dividerColor, height: 1),
        // Dark mode toggle — wired to ThemeProvider so it affects entire app
        provider.Consumer<ThemeProvider>(
          builder: (_, tp, __) => _settingsRow(
            context: context,
            icon: Icons.dark_mode_outlined,
            label: 'Dark Mode',
            subtitle: 'Switch to dark theme',
            trailing: Switch(
                value: tp.isDarkMode,
                onChanged: tp.toggleTheme,
                activeColor: AppTheme.primary),
          ),
        ),
      ],
    );
  }

  Widget _buildAccountDetailsCard(BuildContext context) {
    final th = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('Account Details', context),
        _accountTile(context, Icons.person_outline, 'Edit Profile', () {
          Navigator.push(context,
              MaterialPageRoute(builder: (_) => const EditProfileScreen()));
        }),
        Divider(color: th.dividerColor, height: 1),
        _accountTile(context, Icons.notifications_outlined, 'Notifications', () {
          Navigator.push(context,
              MaterialPageRoute(
                  builder: (_) => const NotificationsSettingsScreen()));
        }),
        Divider(color: th.dividerColor, height: 1),
        _accountTile(context, Icons.shield_outlined, 'Privacy & Security', () {
          Navigator.push(context,
              MaterialPageRoute(builder: (_) => const PrivacyScreen()));
        }),
        Divider(color: th.dividerColor, height: 1),
        _accountTile(context, Icons.settings_outlined, 'App Preferences', () {
          Navigator.push(context,
              MaterialPageRoute(
                  builder: (_) => const AppPreferencesScreen()));
        }),
      ],
    );
  }

  Widget _buildUsageStatsCard(BuildContext context) {
    final th = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle('Usage Statistics', context),
        if (_isLoading)
          Center(
              child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 32),
            child: CircularProgressIndicator(
                color: th.colorScheme.primary),
          ))
        else
          GridView.count(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            mainAxisSpacing: 12,
            crossAxisSpacing: 12,
            childAspectRatio: 1.6,
            children: [
              _statTile(context, 'Images Analyzed',   '$_imagesAnalyzed',   Colors.blue),
              _statTile(context, 'Issues Resolved',   '$_issuesResolved',   Colors.green),
              _statTile(context, 'Reports Generated', '$_reportsGenerated', Colors.orange),
              _statTile(context, 'Avg Confidence',    _avgConfidence,       Colors.purple),
            ],
          ),
      ],
    );
  }

  Widget _buildBottomCard(BuildContext context) {
    final th = Theme.of(context);
    return Column(
      children: [
        ListTile(
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          leading: _iconBox(context, Icons.help_outline,
              th.textTheme.bodyMedium?.color ?? Colors.grey),
          title: Text('Help Center',
              style: GoogleFonts.nunito(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: th.colorScheme.onBackground)),
          onTap: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const HelpScreen())),
        ),
        Divider(color: th.dividerColor, height: 1, indent: 16, endIndent: 16),
        ListTile(
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          leading: _iconBox(context, Icons.logout, Colors.red,
              bgColor: Colors.red.withOpacity(0.12)),
          title: Text('Logout',
              style: GoogleFonts.nunito(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.red)),
          onTap: _logout,
        ),
      ],
    );
  }

  // ── Widget helpers ─────────────────────────────────────────────────────────

  Widget _darkCard(BuildContext context,
      {required Widget child, EdgeInsets? padding}) {
    final th = Theme.of(context);
    return Container(
      decoration: BoxDecoration(
        color: th.cardColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.15),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      padding: padding ?? const EdgeInsets.all(16),
      child: child,
    );
  }

  Widget _sectionTitle(String text, BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 12),
        child: Text(text,
            style: GoogleFonts.nunito(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Theme.of(context).colorScheme.onBackground)),
      );

  Widget _infoTile(BuildContext context, String label, String value,
      IconData icon) {
    final th = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 13, color: th.textTheme.bodySmall?.color),
            const SizedBox(width: 4),
            Flexible(
              child: Text(label,
                  style: GoogleFonts.nunito(
                      fontSize: 12,
                      color: th.textTheme.bodySmall?.color),
                  overflow: TextOverflow.ellipsis),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(value,
            style: GoogleFonts.nunito(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: th.colorScheme.onBackground),
            overflow: TextOverflow.ellipsis,
            maxLines: 2),
      ],
    );
  }

  Widget _settingsRow({
    required BuildContext context,
    required IconData icon,
    required String label,
    required String subtitle,
    required Widget trailing,
  }) {
    final th = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          _iconBox(context, icon,
              th.textTheme.bodyMedium?.color ?? Colors.grey),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: GoogleFonts.nunito(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: th.colorScheme.onBackground)),
                Text(subtitle,
                    style: GoogleFonts.nunito(
                        fontSize: 12,
                        color: th.textTheme.bodySmall?.color)),
              ],
            ),
          ),
          trailing,
        ],
      ),
    );
  }

  Widget _accountTile(BuildContext context, IconData icon, String label,
      VoidCallback onTap) {
    final th = Theme.of(context);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 4),
        child: Row(
          children: [
            _iconBox(context, icon,
                th.textTheme.bodyMedium?.color ?? Colors.grey),
            const SizedBox(width: 12),
            Expanded(
              child: Text(label,
                  style: GoogleFonts.nunito(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: th.colorScheme.onBackground)),
            ),
            Icon(Icons.chevron_right,
                color: th.textTheme.bodySmall?.color, size: 20),
          ],
        ),
      ),
    );
  }

  Widget _statTile(BuildContext context, String label, String value,
      Color color) {
    final th = Theme.of(context);
    // Use a slightly offset card color for stats inner tiles
    final tileBg = th.brightness == Brightness.dark
        ? const Color(0xFF252525)
        : Colors.grey[100]!;
    return Container(
      decoration: BoxDecoration(
        color: tileBg,
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(value,
              style: TextStyle(
                  fontSize: 36, fontWeight: FontWeight.bold, color: color)),
          const SizedBox(height: 4),
          Text(label,
              style: TextStyle(
                  color: th.textTheme.bodySmall?.color, fontSize: 12),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _iconBox(BuildContext context, IconData icon, Color iconColor,
      {Color? bgColor}) {
    final th = Theme.of(context);
    final bg = bgColor ??
        (th.brightness == Brightness.dark
            ? const Color(0xFF2A2A2A)
            : Colors.grey[200]!);
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Icon(icon, size: 20, color: iconColor),
    );
  }

  Widget _shimmerLine(double width, double height, BuildContext context) {
    final th = Theme.of(context);
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: th.brightness == Brightness.dark
            ? Colors.grey[800]
            : Colors.grey[300],
        borderRadius: BorderRadius.circular(6),
      ),
    );
  }
}