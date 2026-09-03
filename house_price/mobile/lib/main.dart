import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const HousePriceMobileApp());
}

class HousePriceMobileApp extends StatelessWidget {
  const HousePriceMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dự đoán giá nhà',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0D5C91)),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final _areaController = TextEditingController();
  final _accessRoadController = TextEditingController();
  final _floorsController = TextEditingController();
  final _bedroomsController = TextEditingController();
  final _bathroomsController = TextEditingController();

  // Android emulator: 10.0.2.2. iOS simulator: localhost.
  // Điện thoại thật: thay bằng LAN IP của máy đang chạy Flask, ví dụ http://192.168.1.10:5000.
  final _apiBaseUrlController = TextEditingController(text: 'http://10.0.2.2:5000');

  String _furnitureState = 'Basic';
  String _selectedModel = 'random_forest_regressor';
  bool _isLoading = false;
  String? _prediction;
  String? _modelLabel;
  String? _error;

  @override
  void dispose() {
    _areaController.dispose();
    _accessRoadController.dispose();
    _floorsController.dispose();
    _bedroomsController.dispose();
    _bathroomsController.dispose();
    _apiBaseUrlController.dispose();
    super.dispose();
  }

  String? _validateNumber(String? value) {
    final number = double.tryParse(value ?? '');
    if (number == null || number <= 0) {
      return 'Hãy nhập một số lớn hơn 0.';
    }
    return null;
  }

  Widget _numberField(String label, TextEditingController controller, {String hint = ''}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextFormField(
        controller: controller,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        validator: _validateNumber,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _prediction = null;
    });

    final baseUrl = _apiBaseUrlController.text.trim().replaceFirst(RegExp(r'/$'), '');
    final payload = {
      'Area': double.parse(_areaController.text),
      'Access Road': double.parse(_accessRoadController.text),
      'Floors': double.parse(_floorsController.text),
      'Bedrooms': double.parse(_bedroomsController.text),
      'Bathrooms': double.parse(_bathroomsController.text),
      'Furniture state': _furnitureState,
      'model': _selectedModel,
    };

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/house-price/v1/predict'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      if (response.statusCode != 200) {
        throw Exception(body['error'] ?? 'Không thể dự đoán.');
      }
      setState(() {
        _prediction = body['predicted_price_display'] as String;
        _modelLabel = body['model_label'] as String;
      });
    } catch (error) {
      setState(() => _error = 'Không thể kết nối REST API: $error');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dự đoán giá nhà'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Kết quả là giá dự đoán theo tỷ VNĐ và chỉ mang tính tham khảo.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 20),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  _numberField('Diện tích (Area)', _areaController, hint: 'Ví dụ: 80'),
                  _numberField('Độ rộng đường vào (Access Road)', _accessRoadController, hint: 'Ví dụ: 5'),
                  _numberField('Số tầng (Floors)', _floorsController, hint: 'Ví dụ: 3'),
                  _numberField('Số phòng ngủ (Bedrooms)', _bedroomsController, hint: 'Ví dụ: 3'),
                  _numberField('Số phòng tắm (Bathrooms)', _bathroomsController, hint: 'Ví dụ: 2'),
                  DropdownButtonFormField<String>(
                    value: _furnitureState,
                    decoration: const InputDecoration(
                      labelText: 'Tình trạng nội thất',
                      border: OutlineInputBorder(),
                    ),
                    items: const [
                      DropdownMenuItem(value: 'Basic', child: Text('Basic')),
                      DropdownMenuItem(value: 'Full', child: Text('Full')),
                    ],
                    onChanged: (value) => setState(() => _furnitureState = value!),
                  ),
                  const SizedBox(height: 14),
                  DropdownButtonFormField<String>(
                    value: _selectedModel,
                    decoration: const InputDecoration(
                      labelText: 'Mô hình',
                      border: OutlineInputBorder(),
                    ),
                    items: const [
                      DropdownMenuItem(value: 'random_forest_regressor', child: Text('Random Forest (khuyến nghị)')),
                      DropdownMenuItem(value: 'gradient_boosting_regressor', child: Text('Gradient Boosting')),
                      DropdownMenuItem(value: 'decision_tree_regressor', child: Text('Decision Tree')),
                      DropdownMenuItem(value: 'linear_regression', child: Text('Linear Regression')),
                      DropdownMenuItem(value: 'ridge_regression', child: Text('Ridge Regression')),
                    ],
                    onChanged: (value) => setState(() => _selectedModel = value!),
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _isLoading ? null : _predict,
                      child: Padding(
                        padding: const EdgeInsets.all(13),
                        child: _isLoading
                            ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator())
                            : const Text('Dự đoán giá nhà'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (_prediction != null) ...[
              const SizedBox(height: 22),
              Card(
                color: Colors.green.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Giá nhà dự đoán', style: TextStyle(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      Text(_prediction!, style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Colors.green.shade800)),
                      const SizedBox(height: 6),
                      Text('Mô hình: $_modelLabel'),
                    ],
                  ),
                ),
              ),
            ],
            if (_error != null) ...[
              const SizedBox(height: 20),
              Text(_error!, style: TextStyle(color: Colors.red.shade700)),
            ],
            const SizedBox(height: 28),
            ExpansionTile(
              title: const Text('Cấu hình REST API'),
              childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              children: [
                TextFormField(
                  controller: _apiBaseUrlController,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    labelText: 'API base URL',
                    hintText: 'http://10.0.2.2:5000',
                    border: OutlineInputBorder(),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
