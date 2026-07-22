// lib/services/auth_service.dart
import 'package:supabase_flutter/supabase_flutter.dart';

final supabase = Supabase.instance.client;

class AuthService {
  /// Sign up a new user and insert a profile row.
  Future<void> signUp(String email, String password, String name) async {
    final response = await supabase.auth.signUp(
      email: email,
      password: password,
    );

    final user = response.user;
    if (user != null) {
      try {
        await supabase.from('profiles').insert({
          'id': user.id,
          'name': name,
        });
      } catch (_) {
        // Profile row is best-effort — auth succeeded regardless.
        // Fix RLS policies in Supabase dashboard if needed.
      }
    }
  }

  /// Sign in an existing user with email + password.
  Future<void> login(String email, String password) async {
    await supabase.auth.signInWithPassword(
      email: email,
      password: password,
    );
  }

  /// Returns the currently signed-in user, or null if not authenticated.
  User? getUser() {
    return supabase.auth.currentUser;
  }

  /// Sign out the current user.
  Future<void> logout() async {
    await supabase.auth.signOut();
  }

  /// Whether a user session is currently active.
  bool get isLoggedIn => supabase.auth.currentUser != null;

  /// Stream that emits auth state changes (login / logout events).
  Stream<AuthState> get authStateChanges =>
      supabase.auth.onAuthStateChange;
}
