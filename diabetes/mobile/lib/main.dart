import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const DiabetesMobileApp());
}

class DiabetesMobileApp extends StatelessWidget {
  const DiabetesMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Diabetes Check',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF007A78)),
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
  final _glucoseController = TextEditingController();
  final _bmiController = TextEditingController();
  final _ageController = TextEditingController();
  final _pregnanciesController = TextEditingController();
  final _pedigreeController = TextEditingController();

  // Android emulator: 10.0.2.2 points to the host machine's localhost.
  // For a physical phone, replace this with your computer's LAN IP address.
  final _apiBaseUrlController = TextEditingController(text: 'http://10.0.2.2:5000');

  String _selectedModel = 'decision_tree';
  bool _isLoading = false;
  String? _prediction;
  String? _modelLabel;
  double? _confidence;
  bool? _isHighRisk;
  String? _error;

  @override
  void dispose() {
    _glucoseController.dispose();
    _bmiController.dispose();
    _ageController.dispose();
    _pregnanciesController.dispose();
    _pedigreeController.dispose();
    _apiBaseUrlController.dispose();
    super.dispose();
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
      'Glucose': double.parse(_glucoseController.text),
      'BMI': double.parse(_bmiController.text),
      'Age': double.parse(_ageController.text),
      'Pregnancies': double.parse(_pregnanciesController.text),
      'DiabetesPedigreeFunction': double.parse(_pedigreeController.text),
      'model': _selectedModel,
    };

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/diabetes/v1/predict'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      final body = jsonDecode(response.body) as Map<String, dynamic>;

      if (response.statusCode != 200) {
        throw Exception(body['error'] ?? 'Không thể dự đoán.');
      }

      setState(() {
        _prediction = body['prediction'] as String;
        _modelLabel = body['model_label'] as String;
        _confidence = (body['confidence'] as num?)?.toDouble();
        _isHighRisk = body['risk_level'] == 'high';
      });
    } catch (error) {
      setState(() => _error = 'Không thể kết nối API: $error');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  String? _validateNumber(String? value, {bool allowZero = false}) {
    final number = double.tryParse(value ?? '');
    if (number == null) return 'Hãy nhập một số hợp lệ.';
    if (allowZero ? number < 0 : number <= 0) return 'Giá trị không hợp lệ.';
    return null;
  }

  Widget _numberField(String label, TextEditingController controller, {bool allowZero = false}) {
    return TextFormField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      validator: (value) => _validateNumber(value, allowZero: allowZero),
      decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final resultColor = _isHighRisk == true ? Colors.red.shade700 : Colors.green.shade700;
    final resultBackground = _isHighRisk == true ? Colors.red.shade50 : Colors.green.shade50;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dự đoán nguy cơ tiểu đường'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Kết quả chỉ hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 20),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  _numberField('Glucose', _glucoseController),
                  const SizedBox(height: 14),
                  _numberField('BMI', _bmiController),
                  const SizedBox(height: 14),
                  _numberField('Tuổi', _ageController),
                  const SizedBox(height: 14),
                  _numberField('Số lần mang thai', _pregnanciesController, allowZero: true),
                  const SizedBox(height: 14),
                  _numberField('Diabetes Pedigree Function', _pedigreeController, allowZero: true),
                  const SizedBox(height: 14),
                  DropdownButtonFormField<String>(
                    value: _selectedModel,
                    decoration: const InputDecoration(labelText: 'Mô hình', border: OutlineInputBorder()),
                    items: const [
                      DropdownMenuItem(value: 'decision_tree', child: Text('Decision Tree (khuyến nghị)')),
                      DropdownMenuItem(value: 'logistic_regression', child: Text('Logistic Regression')),
                      DropdownMenuItem(value: 'knn', child: Text('K-Nearest Neighbors')),
                      DropdownMenuItem(value: 'linear_svm', child: Text('SVM Linear')),
                      DropdownMenuItem(value: 'rbf_svm', child: Text('SVM RBF')),
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
                            : const Text('Dự đoán'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (_prediction != null) ...[
              const SizedBox(height: 22),
              Card(
                color: resultBackground,
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_prediction!, style: TextStyle(fontSize: 19, fontWeight: FontWeight.bold, color: resultColor)),
                      const SizedBox(height: 8),
                      Text('Mô hình: $_modelLabel'),
                      Text('Độ tin cậy của mô hình: ${_confidence?.toStringAsFixed(2) ?? 'N/A'}%'),
                    ],
                  ),
                ),
              ),
            ],
            if (_error != null) ...[
              const SizedBox(height: 22),
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
