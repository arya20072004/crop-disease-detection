// lib/screens/privacy_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/app_theme.dart';

class PrivacyScreen extends StatelessWidget {
  const PrivacyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: cs.onBackground),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('Privacy & Security',
            style: GoogleFonts.nunito(
                fontWeight: FontWeight.w800, fontSize: 20)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _section(context, 'Data Privacy',
              'Your farm data is encrypted and stored securely. We never share '
                  'your personal information with third parties without your consent.',
              Icons.lock_outline, AppTheme.primary),
          const SizedBox(height: 12),
          _section(context, 'Two-Factor Authentication',
              'Strengthen your account security by enabling two-factor '
                  'authentication via email or SMS.',
              Icons.security_outlined, Colors.blue),
          const SizedBox(height: 12),
          _section(context, 'Delete Account',
              'Permanently delete your account and all associated data. '
                  'This action cannot be undone.',
              Icons.delete_outline, Colors.red),
        ],
      ),
    );
  }

  Widget _section(BuildContext context, String title, String body,
      IconData icon, Color color) {
    final th = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: th.cardColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: GoogleFonts.nunito(
                        color: th.colorScheme.onBackground,
                        fontWeight: FontWeight.w700,
                        fontSize: 14)),
                const SizedBox(height: 6),
                Text(body,
                    style: GoogleFonts.nunito(
                        color: th.textTheme.bodySmall?.color, fontSize: 12)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
