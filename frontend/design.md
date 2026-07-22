# App Design Document: Fasal Saarthi (Crop Disease EWS)

## 1. Overview
Fasal Saarthi is a Flutter-based Early Warning System for crop diseases. It enables users to scan leaves for diseases, check weather forecasts, and maintain a history of crop health. The application connects to a Supabase backend for authentication and data storage.

## 2. Technical Stack
- **Framework**: Flutter (Dart)
- **State Management**: `flutter_riverpod` (primary state tree) combined with `provider` (for specific theme provisioning).
- **Backend/Auth**: Supabase (`supabase_flutter`).
- **Networking**: `dio` and `http`.
- **Local Storage**: `hive`, `shared_preferences`.

## 3. UI/UX Design System (`AppTheme`)

The application supports dynamic **Light** and **Dark** modes to cater to field usage under bright sunlight as well as night-time viewing. It utilizes Material 3 design paradigms.

### 3.1 Color Palette
The brand revolves around agricultural themes (greens), combined with a traffic-light risk system.

*   **Brand Primary**: Green (`#4CAF50`)
*   **Primary Light/Dark Variants**: `#81C784` / `#388E3C`

**Risk / Alert Palette:**
*   **High Risk (Critical)**: Red (`#E53935`) - Used for immediate interventions and severe disease detection.
*   **Moderate Risk (Medium)**: Amber (`#xFFFF8F00`)
*   **Low Risk (Safe)**: Green (`#43A047`)

**Backgrounds & Surfaces:**
*   **Light Mode**: Background `#F5F5F5` (Off-white), Card Surface: White.
*   **Dark Mode**: Background `#121212` (Deep Charcoal), Card Surface: `#1E1E1E`.

### 3.2 Typography & Shapes
*   **Typography**: Clean, readable sans-serif fonts using `TextTheme` defaults scaled for readability.
*   **Shapes/Radii**: 
    *   Cards and main containers feature heavy rounding: `BorderRadius.circular(16)`.
    *   Buttons and Input Fields feature medium rounding: `BorderRadius.circular(12)`.
    *   Inputs use filled background colors (`#2A2A2A` in dark mode) without borders until focused.

## 4. App Navigation Structure

The app uses a `MainShell` acting as a scaffold with a `NavigationBar` (Material 3 Bottom Navigation) featuring 6 distinct tabs:

1.  **Home**: `HomeScreen`
2.  **Scan**: `AnalyzeScreen` (Uses Camera)
3.  **Forecast**: `ForecastScreen` (Next 7 Days Weather/Risk)
4.  **Compare**: `CompareScreen`
5.  **History**: `HistoryScreen` (Past Risks)
6.  **Profile**: `ProfileScreen`

An `AuthGate` widget sits above the `MainShell`. If the user is unauthenticated via Supabase, they are redirected to the `LoginScreen`.

## 5. Screen Layout Details

### 5.1 Home Screen (`HomeScreen`)
The dashboard and entry point of the app. It's designed for quick access to primary actions.

- **App Bar**: Contains the App Title ("Fasal Saarthi") and a Settings icon.
- **Hero Card**: A prominent, visually striking banner with a solid green (`primaryDark`) background. It asks "Is your crop sick?" alongside an eco/leaf icon, guiding users on what the app is for.
- **Quick Actions Grid**: A 2x2 grid (`GridView.count`) of tinted cards.
    - **Scan a Leaf** (Green) -> Navigates to Analyze Tab
    - **Next 7 Days** (Blue, `#1976D2`) -> Navigates to Forecast Tab
    - **Compare Crops** (Purple, `#7B1FA2`) -> Navigates to Compare Tab
    - **Past Results** (Orange, `#E64A19`) -> Navigates to History Tab
- **Photo Tips Section**: An educational list mapping emojis (📸, ☀️, 🤲, 📍) to practical tips for taking better leaf photos, ensuring the AI model gets high-quality input.

### 5.2 Analyze Screen (Scan Leaf)
- Designed to handle camera operations (`camera`, `image_picker`). Users will upload or take live photos of leaves. Feedback will utilize the **Risk Palette** to denote disease severity.

### 5.3 Forecast & History Screens
- Likely utilize `fl_chart` for visualizing risk trends over the next 7 days and historical data.

## 6. Directory Structure Overview
- `lib/core/`: Contains `app_theme.dart` (single source of truth for colors), constants, and localization files.
- `lib/screens/`: Contains all the individual tab views and auth screens.
- `lib/widgets/`: Reusable UI components.
- `lib/providers/`: Riverpod state notifiers and providers for business logic.
- `lib/models/` & `lib/services/`: Data models and backend communication logic.
