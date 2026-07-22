// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:provider/provider.dart' as provider;
import 'package:supabase_flutter/supabase_flutter.dart';
import 'core/core.dart';
import 'providers.dart';
import 'providers/theme_provider.dart';
import 'screens/home_screen.dart';
import 'screens/analyze_screen.dart';
import 'screens/forecast_screen.dart';
import 'screens/compare_screen.dart';
import 'screens/history_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/login_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/recommendation_screen.dart';

// ── Supabase credentials ──────────────────────────────────────────────────
const _supabaseUrl     = 'https://clwbjefmnictvolnevai.supabase.co';
const _supabaseAnonKey = 'sb_publishable_1Okhep2DcQPH6RNbLIscsw_HkD2GcJo';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Supabase.initialize(
    url: _supabaseUrl,
    anonKey: _supabaseAnonKey,
  );

  // Riverpod wraps the entire tree; Provider (for ThemeProvider) wraps inside.
  runApp(
    const ProviderScope(child: _ThemeRoot()),
  );
}

/// Sits between ProviderScope and CropDiseaseApp so ThemeProvider is
/// available to every widget via context.read/watch<ThemeProvider>().
class _ThemeRoot extends StatelessWidget {
  const _ThemeRoot();

  @override
  Widget build(BuildContext context) {
    return provider.ChangeNotifierProvider(
      create: (_) => ThemeProvider(),
      child: const CropDiseaseApp(),
    );
  }
}

class CropDiseaseApp extends ConsumerWidget {
  const CropDiseaseApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final locale        = ref.watch(localeProvider);
    final themeProvider = context.watch<ThemeProvider>();

    return MaterialApp(
      title: 'Fasal Saarthi',
      debugShowCheckedModeBanner: false,
      theme:     AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: themeProvider.themeMode,
      locale: locale,
      supportedLocales: AppLocalizations.supportedLocales,
      localizationsDelegates: const [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      home: const AuthGate(),
      routes: {
        '/analyze' : (_) => const AnalyzeScreen(),
        '/forecast': (_) => const ForecastScreen(),
        '/compare' : (_) => const CompareScreen(),
        '/history' : (_) => const HistoryScreen(),
        '/settings': (_) => const SettingsScreen(),
        '/profile' : (_) => const ProfileScreen(),
        '/recommendations': (_) => const RecommendationScreen(),
        '/login'   : (_) => const LoginScreen(),
      },
    );
  }
}

// ── Main shell with bottom navigation ────────────────────────────────────
class MainShell extends ConsumerStatefulWidget {
  const MainShell({super.key});
  @override
  ConsumerState<MainShell> createState() => _MainShellState();
}

class _MainShellState extends ConsumerState<MainShell> {
  int _idx = 0;

  void _switchTab(int i) => setState(() => _idx = i);

  List<Widget> get _screens => [
    HomeScreen(onTabSwitch: _switchTab),
    const AnalyzeScreen(),
    const ForecastScreen(),
    const CompareScreen(),
    const HistoryScreen(),
    const ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    ref.watch(localeProvider);
    final l      = AppLocalizations.of(context);
    final cs     = Theme.of(context).colorScheme;

    return Scaffold(
      body: IndexedStack(index: _idx, children: _screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _idx,
        onDestinationSelected: (i) => setState(() => _idx = i),
        backgroundColor: Theme.of(context).cardColor,
        indicatorColor: cs.primary.withOpacity(0.12),
        elevation: 8,
        shadowColor: Colors.black26,
        height: 64,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home, color: cs.primary),
            label: 'Home',
          ),
          NavigationDestination(
            icon: const Icon(Icons.camera_alt_outlined),
            selectedIcon: Icon(Icons.camera_alt, color: cs.primary),
            label: l?.translate('scan_leaf') ?? 'Scan',
          ),
          NavigationDestination(
            icon: const Icon(Icons.wb_sunny_outlined),
            selectedIcon: Icon(Icons.wb_sunny, color: cs.primary),
            label: l?.translate('next_7_days') ?? 'Forecast',
          ),
          NavigationDestination(
            icon: const Icon(Icons.bar_chart_outlined),
            selectedIcon: Icon(Icons.bar_chart, color: cs.primary),
            label: l?.translate('compare') ?? 'Compare',
          ),
          NavigationDestination(
            icon: const Icon(Icons.history_outlined),
            selectedIcon: Icon(Icons.history, color: cs.primary),
            label: l?.translate('past_risks') ?? 'History',
          ),
          NavigationDestination(
            icon: const Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person, color: cs.primary),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}

// ── Auth Gate ─────────────────────────────────────────────────────────────
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  bool _loggedIn = Supabase.instance.client.auth.currentUser != null;

  @override
  void initState() {
    super.initState();
    Supabase.instance.client.auth.onAuthStateChange.listen((data) {
      if (!mounted) return;
      setState(() { _loggedIn = data.session != null; });
    });
  }

  @override
  Widget build(BuildContext context) {
    return _loggedIn ? const MainShell() : const LoginScreen();
  }
}
