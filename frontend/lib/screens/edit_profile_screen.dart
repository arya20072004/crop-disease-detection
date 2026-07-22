// lib/screens/edit_profile_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../core/app_theme.dart';

class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final _sb        = Supabase.instance.client;
  final _formKey   = GlobalKey<FormState>();
  final _nameCtrl  = TextEditingController();
  final _phoneCtrl = TextEditingController();
  bool _loading    = false;
  bool _fetching   = true;

  @override
  void initState() {
    super.initState();
    _fetchProfile();
  }

  Future<void> _fetchProfile() async {
    final user = _sb.auth.currentUser;
    if (user == null) { setState(() => _fetching = false); return; }
    try {
      final res = await _sb
          .from('profiles')
          .select('name, phone')
          .eq('id', user.id)
          .maybeSingle();
      if (mounted) {
        _nameCtrl.text  = (res?['name']  as String?) ?? '';
        _phoneCtrl.text = (res?['phone'] as String?) ?? '';
        setState(() => _fetching = false);
      }
    } catch (_) {
      if (mounted) setState(() => _fetching = false);
    }
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    final user = _sb.auth.currentUser;
    if (user == null) return;
    try {
      await _sb.from('profiles').upsert({
        'id'   : user.id,
        'name' : _nameCtrl.text.trim(),
        'phone': _phoneCtrl.text.trim(),
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Profile updated ✓')),
        );
        Navigator.pop(context);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _phoneCtrl.dispose();
    super.dispose();
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
        title: Text(
          'Edit Profile',
          style: GoogleFonts.nunito(fontWeight: FontWeight.w800, fontSize: 20),
        ),
      ),
      body: _fetching
          ? Center(child: CircularProgressIndicator(color: cs.primary))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _field('Full Name', _nameCtrl, Icons.person_outline,
                        required: true),
                    const SizedBox(height: 16),
                    _field('Phone Number', _phoneCtrl, Icons.phone_outlined),
                    const SizedBox(height: 32),
                    ElevatedButton(
                      onPressed: _loading ? null : _save,
                      child: _loading
                          ? const SizedBox(
                              width: 22, height: 22,
                              child: CircularProgressIndicator(
                                  color: Colors.white, strokeWidth: 2))
                          : Text('Save Changes',
                              style: GoogleFonts.nunito(
                                  fontSize: 16, fontWeight: FontWeight.w700)),
                    ),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _field(String label, TextEditingController ctrl, IconData icon,
      {bool required = false}) {
    final th = Theme.of(context);
    return TextFormField(
      controller: ctrl,
      style: TextStyle(color: th.colorScheme.onBackground),
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, color: th.textTheme.bodyMedium?.color),
      ),
      validator: required
          ? (v) => (v == null || v.trim().isEmpty) ? '$label is required' : null
          : null,
    );
  }
}
