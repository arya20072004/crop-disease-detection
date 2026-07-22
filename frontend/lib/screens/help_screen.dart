// lib/screens/help_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';
import '../core/app_theme.dart';

class HelpScreen extends StatelessWidget {
  const HelpScreen({super.key});

  Future<void> _openUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) await launchUrl(uri);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: cs.onBackground),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('Help Center',
            style: GoogleFonts.nunito(
                fontWeight: FontWeight.w800, fontSize: 20)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _helpTile(context, 'Getting Started',
              'Learn how to use Fasal Saarthi to protect your crops.',
              Icons.play_circle_outline, Colors.blue, () {}),
          _helpTile(context, 'AI Disease Detection',
              'How our AI model analyses leaf images for diseases.',
              Icons.biotech_outlined, AppTheme.primary, () {}),
          _helpTile(context, 'FAQs',
              'Frequently asked questions from farmers like you.',
              Icons.quiz_outlined, Colors.orange, () {}),
          _helpTile(context, 'Contact Support',
              'Reach our team via email at support@fasalsaarthi.in',
              Icons.email_outlined, Colors.purple,
              () => _openUrl('mailto:support@fasalsaarthi.in')),
          _helpTile(context, 'Privacy Policy',
              'Read our data privacy and usage policy.',
              Icons.policy_outlined, Colors.teal,
              () => _openUrl('https://fasalsaarthi.in/privacy')),
        ],
      ),
    );
  }

  Widget _helpTile(BuildContext context, String title, String subtitle,
      IconData icon, Color color, VoidCallback onTap) {
    final th = Theme.of(context);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: th.cardColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: ListTile(
        onTap: onTap,
        leading: Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: color.withOpacity(0.12),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(icon, color: color, size: 22),
        ),
        title: Text(title,
            style: GoogleFonts.nunito(
                color: th.colorScheme.onBackground,
                fontWeight: FontWeight.w600,
                fontSize: 14)),
        subtitle: Text(subtitle,
            style: GoogleFonts.nunito(
                color: th.textTheme.bodySmall?.color, fontSize: 12)),
        trailing: Icon(Icons.chevron_right,
            color: th.textTheme.bodySmall?.color),
      ),
    );
  }
}
