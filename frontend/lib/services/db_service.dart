// lib/services/db_service.dart
import 'package:supabase_flutter/supabase_flutter.dart';

final supabase = Supabase.instance.client;

class DBService {
  // ── Crops ────────────────────────────────────────────────────────────────

  /// Saves crop metadata to the `crops` table for the logged-in user.
  Future<void> saveCrop(Map<String, dynamic> cropData) async {
    final user = supabase.auth.currentUser;
    if (user == null) {
      print('[DBService] saveCrop: skipped — user not logged in');
      return;
    }
    final payload = {'user_id': user.id, ...cropData};
    print('[DBService] saveCrop: user=${user.id}');
    await supabase.from('crops').insert(payload);
    print('[DBService] saveCrop: insert success ✓');
  }

  // ── Predictions ───────────────────────────────────────────────────────────

  /// Saves a prediction result to the `predictions` table.
  /// Returns the new row's id (used to log prediction_history views later).
  Future<String?> savePrediction(Map<String, dynamic> predictionData) async {
    final user = supabase.auth.currentUser;
    if (user == null) {
      print('[DBService] savePrediction: skipped — user not logged in');
      throw Exception('User not logged in — cannot save prediction');
    }
    final payload = {'user_id': user.id, ...predictionData};
    print('[DBService] savePrediction: user=${user.id}');

    final res = await supabase
        .from('predictions')
        .insert(payload)
        .select('id')
        .single();
    final id = res['id']?.toString();
    print('[DBService] savePrediction: insert success ✓ id=$id');
    return id;
  }

  // ── Forecasts ─────────────────────────────────────────────────────────────

  /// Saves the top forecast risks to the `forecasts` table.
  /// Schema: id, user_id, disease, risk_today, risk_day7, highest_risk, created_at
  Future<void> saveForecast({
    required List<Map<String, dynamic>> diseases,
  }) async {
    final user = supabase.auth.currentUser;
    if (user == null) {
      print('[DBService] saveForecast: skipped — user not logged in');
      return;
    }

    final rows = diseases.map((d) {
      final daily = (d['daily'] as List?)
              ?.map((v) => (v as num).toDouble())
              .toList() ??
          <double>[];
      return {
        'user_id'     : user.id,
        'disease'     : d['disease'],
        'risk_today'  : daily.isNotEmpty ? daily.first : d['peak_risk'],
        'risk_day7'   : daily.length >= 7 ? daily[6]  : d['peak_risk'],
        'highest_risk': d['peak_risk'],
        'created_at'  : DateTime.now().toIso8601String(),
      };
    }).toList();

    print('[DBService] saveForecast: saving ${rows.length} rows for user=${user.id}');
    await supabase.from('forecasts').insert(rows);
    print('[DBService] saveForecast: insert success ✓');
  }

  // ── Predictions — read ────────────────────────────────────────────────────

  /// Fetches past predictions for the current user, newest first.
  Future<List<Map<String, dynamic>>> getPredictions() async {
    final user = supabase.auth.currentUser;
    if (user == null) {
      print('[DBService] getPredictions: skipped — user not logged in');
      return [];
    }
    final response = await supabase
        .from('predictions')
        .select()
        .eq('user_id', user.id)
        .order('created_at', ascending: false);
    return List<Map<String, dynamic>>.from(response);
  }

  // ── Recommendations ───────────────────────────────────────────────────────

  /// Fetches treatment recommendations for a given disease name.
  Future<List<Map<String, dynamic>>> getRecommendations(String disease) async {
    final response = await supabase
        .from('recommendations')
        .select()
        .eq('disease', disease);
    return List<Map<String, dynamic>>.from(response);
  }

  /// Inserts a recommendation row for [disease] if none already exists.
  /// Seeds the `recommendations` table client-side using the authenticated
  /// session — no server-side Supabase client needed.
  Future<void> saveRecommendationIfMissing(String disease) async {
    try {
      // Check whether a row already exists for this disease
      final existing = await supabase
          .from('recommendations')
          .select('id')
          .eq('disease', disease)
          .maybeSingle();

      if (existing != null) {
        print('[DBService] saveRecommendation: row already exists for "$disease"');
        return;
      }

      final rec = _recommendationLookup(disease);
      await supabase.from('recommendations').insert({
        'disease'  : disease,
        'treatment': rec['treatment'],
        'dosage'   : rec['dosage'],
        'frequency': rec['frequency'],
        'notes'    : rec['notes'],
      });
      print('[DBService] saveRecommendation: inserted for "$disease" ✓');
    } catch (e) {
      print('[DBService] saveRecommendation: error — $e');
    }
  }

  // ── Prediction History ────────────────────────────────────────────────────

  /// Records that the current user viewed a specific prediction.
  /// Writes to the `prediction_history` table.
  Future<void> savePredictionHistory(String predictionId) async {
    final user = supabase.auth.currentUser;
    if (user == null) {
      print('[DBService] savePredictionHistory: skipped — user not logged in');
      return;
    }
    try {
      await supabase.from('prediction_history').insert({
        'user_id'      : user.id,
        'prediction_id': predictionId,
        'viewed_at'    : DateTime.now().toIso8601String(),
      });
      print('[DBService] savePredictionHistory: recorded view of $predictionId ✓');
    } catch (e) {
      print('[DBService] savePredictionHistory: error — $e');
    }
  }
}

// ── Recommendation lookup ─────────────────────────────────────────────────────
// Mirrors the INTERVENTIONS dict in api.py so Flutter can seed the
// recommendations table using the authenticated Supabase session.

Map<String, String> _recommendationLookup(String disease) {
  const data = <String, Map<String, String>>{
    'Tomato___Late_blight': {
      'treatment': 'Mancozeb 75% WP',
      'dosage'   : '2.5 g/L water',
      'frequency': 'Every 7 days',
      'notes'    : 'Apply at first sign or when risk > 60%. Avoid overhead irrigation.',
    },
    'Potato___Late_blight': {
      'treatment': 'Metalaxyl + Mancozeb',
      'dosage'   : '2.5 g/L water',
      'frequency': 'Every 7–10 days',
      'notes'    : 'Preventive when risk > 50%. Destroy infected haulms.',
    },
    'Tomato___Early_blight': {
      'treatment': 'Chlorothalonil 75% WP',
      'dosage'   : '2 g/L water',
      'frequency': 'Every 10 days',
      'notes'    : 'Begin when lower leaves show spots. Remove infected debris.',
    },
    'Potato___Early_blight': {
      'treatment': 'Mancozeb 75% WP',
      'dosage'   : '2 g/L water',
      'frequency': 'Every 10–14 days',
      'notes'    : 'Start at tuber initiation stage. Ensure full canopy coverage.',
    },
    'Apple___Apple_scab': {
      'treatment': 'Captan 50% WP',
      'dosage'   : '2.5 g/L water',
      'frequency': 'Every 7–10 days during wet weather',
      'notes'    : 'Critical window: pink bud to petal fall. Re-apply after rain.',
    },
    'Tomato___Bacterial_spot': {
      'treatment': 'Copper oxychloride 50% WP',
      'dosage'   : '3 g/L water',
      'frequency': 'Every 7 days',
      'notes'    : 'Apply before or at symptom onset. Avoid working in wet foliage.',
    },
    'Squash___Powdery_mildew': {
      'treatment': 'Sulphur 80% WP',
      'dosage'   : '3 g/L water',
      'frequency': 'Every 10–14 days',
      'notes'    : 'Apply at first sign of white powder. Improve air circulation.',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
      'treatment': 'Propiconazole 25% EC',
      'dosage'   : '1 mL/L water',
      'frequency': 'Every 14 days',
      'notes'    : 'Apply at tasseling if risk is high. Use resistant varieties next season.',
    },
  };

  const defaultRec = <String, String>{
    'treatment': 'Consult local agronomist',
    'dosage'   : 'As recommended',
    'frequency': 'Every 7–14 days',
    'notes'    : 'At first confirmed symptom. Contact your nearest Krishi Kendra.',
  };

  return data[disease] ?? defaultRec;
}
