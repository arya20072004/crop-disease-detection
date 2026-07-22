// lib/core/app_theme.dart
//
// Single source of truth for all app theming.
// • Static constants (primary, riskColor, etc.) used by existing screens
// • darkTheme / lightTheme ThemeData objects consumed by MaterialApp
//
import 'package:flutter/material.dart';

class AppTheme {
  // ── Brand colours ──────────────────────────────────────────────────────────
  static const Color primary       = Color(0xFF4CAF50);
  static const Color primaryGreen  = Color(0xFF4CAF50); // alias for new code
  static const Color primaryLight  = Color(0xFF81C784);
  static const Color primaryDark   = Color(0xFF388E3C);

  // ── Risk palette ───────────────────────────────────────────────────────────
  static const Color riskHigh      = Color(0xFFE53935);
  static const Color riskModerate  = Color(0xFFFF8F00);
  static const Color riskMod       = riskModerate;  // alias for legacy code
  static const Color riskLow       = Color(0xFF43A047);

  // ── Text ───────────────────────────────────────────────────────────────────
  static const Color textPrimary   = Color(0xFF212121);
  static const Color textSecondary = Color(0xFF757575);

  // ── Surface (kept for legacy screen references) ────────────────────────────
  static const Color surface       = Color(0xFF1E1E1E); // dark surface

  /// Returns a risk-appropriate colour for a 0–1 score.
  static Color riskColor(double score) {
    if (score >= 0.6) return riskHigh;
    if (score >= 0.3) return riskModerate;
    return riskLow;
  }

  /// Returns a friendly label for a risk score (0–1).
  static String riskLabelFriendly(double score) {
    if (score >= 0.6) return 'High Risk';
    if (score >= 0.3) return 'Moderate Risk';
    return 'Low Risk';
  }

  /// Returns a step-indicator colour based on intervention urgency.
  static Color stepColor(String urgency) {
    final lower = urgency.toLowerCase();
    if (lower.contains('critical') || lower.contains('immediate')) return riskHigh;
    if (lower.contains('moderate') || lower.contains('medium')) return riskModerate;
    return riskLow;
  }

  // ── Dark ThemeData ─────────────────────────────────────────────────────────
  static ThemeData get darkTheme => ThemeData(
    brightness: Brightness.dark,
    useMaterial3: true,
    colorScheme: ColorScheme.dark(
      primary: primary,
      secondary: primary,
      surface: const Color(0xFF1E1E1E),
      background: const Color(0xFF121212),
      onBackground: Colors.white,
      onSurface: Colors.white,
      onPrimary: Colors.white,
    ),
    scaffoldBackgroundColor: const Color(0xFF121212),
    cardColor: const Color(0xFF1E1E1E),
    appBarTheme: const AppBarTheme(
      backgroundColor: Color(0xFF121212),
      foregroundColor: Colors.white,
      elevation: 0,
      scrolledUnderElevation: 0,
    ),
    bottomNavigationBarTheme: const BottomNavigationBarThemeData(
      backgroundColor: Color(0xFF1E1E1E),
      selectedItemColor: primary,
      unselectedItemColor: Colors.grey,
    ),
    switchTheme: SwitchThemeData(
      thumbColor: MaterialStateProperty.resolveWith(
        (s) => s.contains(MaterialState.selected) ? primary : Colors.grey,
      ),
      trackColor: MaterialStateProperty.resolveWith(
        (s) => s.contains(MaterialState.selected)
            ? primary.withOpacity(0.5)
            : Colors.grey.withOpacity(0.3),
      ),
    ),
    cardTheme: CardThemeData(
      color: const Color(0xFF1E1E1E),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      elevation: 0,
    ),
    dividerColor: Colors.white.withOpacity(0.08),
    textTheme: TextTheme(
      headlineMedium: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
      titleMedium:    const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
      titleSmall:     const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
      bodyLarge:      const TextStyle(color: Colors.white),
      bodyMedium:     TextStyle(color: Colors.grey[400]),
      bodySmall:      TextStyle(color: Colors.grey[500]),
      labelSmall:     TextStyle(color: Colors.grey[500]),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: const Color(0xFF2A2A2A),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: primary, width: 1.5),
      ),
      hintStyle: TextStyle(color: Colors.grey[600]),
      labelStyle: TextStyle(color: Colors.grey[400]),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: primary,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        minimumSize: const Size.fromHeight(52),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: primary,
        side: const BorderSide(color: primary),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(foregroundColor: primary),
    ),
    iconTheme: const IconThemeData(color: Colors.white),
    chipTheme: ChipThemeData(
      backgroundColor: const Color(0xFF2A2A2A),
      selectedColor: primary.withOpacity(0.15),
      labelStyle: const TextStyle(color: Colors.white),
      side: const BorderSide(color: Colors.transparent),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(color: primary),
    dropdownMenuTheme: DropdownMenuThemeData(
      menuStyle: MenuStyle(
        backgroundColor: MaterialStatePropertyAll(const Color(0xFF2A2A2A)),
      ),
    ),
  );

  // ── Light ThemeData ────────────────────────────────────────────────────────
  static ThemeData get lightTheme => ThemeData(
    brightness: Brightness.light,
    useMaterial3: true,
    colorScheme: ColorScheme.light(
      primary: primary,
      secondary: primary,
      surface: Colors.white,
      background: const Color(0xFFF5F5F5),
      onBackground: Colors.black87,
      onSurface: Colors.black87,
      onPrimary: Colors.white,
    ),
    scaffoldBackgroundColor: const Color(0xFFF5F5F5),
    cardColor: Colors.white,
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.white,
      foregroundColor: Colors.black87,
      elevation: 0,
      scrolledUnderElevation: 0,
    ),
    bottomNavigationBarTheme: const BottomNavigationBarThemeData(
      backgroundColor: Colors.white,
      selectedItemColor: primary,
      unselectedItemColor: Colors.grey,
    ),
    switchTheme: SwitchThemeData(
      thumbColor: MaterialStateProperty.resolveWith(
        (s) => s.contains(MaterialState.selected) ? primary : Colors.grey,
      ),
      trackColor: MaterialStateProperty.resolveWith(
        (s) => s.contains(MaterialState.selected)
            ? primary.withOpacity(0.5)
            : Colors.grey.withOpacity(0.3),
      ),
    ),
    cardTheme: CardThemeData(
      color: Colors.white,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      elevation: 2,
      shadowColor: Colors.black12,
    ),
    dividerColor: Colors.grey.withOpacity(0.2),
    textTheme: TextTheme(
      headlineMedium: const TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
      titleMedium:    const TextStyle(color: Colors.black87, fontWeight: FontWeight.w600),
      titleSmall:     const TextStyle(color: Colors.black87, fontWeight: FontWeight.w600),
      bodyLarge:      const TextStyle(color: Colors.black87),
      bodyMedium:     TextStyle(color: Colors.grey[700]),
      bodySmall:      TextStyle(color: Colors.grey[600]),
      labelSmall:     TextStyle(color: Colors.grey[600]),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: Colors.grey[100],
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: primary, width: 1.5),
      ),
      hintStyle: TextStyle(color: Colors.grey[500]),
      labelStyle: TextStyle(color: Colors.grey[700]),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: primary,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        minimumSize: const Size.fromHeight(52),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: primary,
        side: const BorderSide(color: primary),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(foregroundColor: primary),
    ),
    iconTheme: const IconThemeData(color: Colors.black87),
    chipTheme: ChipThemeData(
      backgroundColor: Colors.grey[200]!,
      selectedColor: primary.withOpacity(0.15),
      labelStyle: const TextStyle(color: Colors.black87),
      side: const BorderSide(color: Colors.transparent),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(color: primary),
  );
}
