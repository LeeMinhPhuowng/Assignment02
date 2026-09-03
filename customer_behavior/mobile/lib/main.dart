import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const CustomerBehaviorApp());
}

class CustomerBehaviorApp extends StatelessWidget {
  const CustomerBehaviorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dự đoán khuyến nghị',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF146CBA)),
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
  final _titleController = TextEditingController();
  final _reviewController = TextEditingController();
  final _apiBaseUrlController = TextEditingController(text: 'http://10.0.2.2:5000');

  String _selectedModel = 'logistic_regression';
  bool _isLoading = false;
  String? _recommendation;
  String? _modelLabel;
  double? _probability;
  bool? _isPositive;
  String? _error;

  @override
  void dispose() {
    _titleController.dispose();
    _reviewController.dispose();
    _apiBaseUrlController.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _isLoading = true;
      _recommendation = null;
      _error = null;
    });

    final baseUrl = _apiBaseUrlController.text.trim().replaceFirst(RegExp(r'/$'), '');
    final payload = {
      'Title': _titleController.text,
      'Review Text': _reviewController.text,
      'model': _selectedModel,
    };

    try {
      final response = await http.post(
        Uri.parse(baseUrl + '/recommendation/v1/predict'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      if (response.statusCode != 200) {
        throw Exception(body['error'] ?? 'Không thể phân loại review.');
      }
      setState(() {
        _recommendation = body['recommendation'] as String;
        _modelLabel = body['model_label'] as String;
        _probability = (body['recommendation_probability'] as num?)?.toDouble();
        _isPositive = body['result_level'] == 'positive';
      });
    } catch (error) {
      setState(() => _error = 'Không thể kết nối REST API: ' + error.toString());
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final resultColor = _isPositive == true ? Colors.green.shade800 : Colors.red.shade800;
    final resultBackground = _isPositive == true ? Colors.green.shade50 : Colors.red.shade50;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dự đoán khuyến nghị'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Nhập tiêu đề và nội dung review để dự đoán khách hàng có khả năng khuyến nghị sản phẩm hay không.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 20),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  TextFormField(
                    controller: _titleController,
                    decoration: const InputDecoration(
                      labelText: 'Tiêu đề review (không bắt buộc)',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextFormField(
                    controller: _reviewController,
                    minLines: 5,
                    maxLines: 8,
                    validator: (value) => (value ?? '').trim().isEmpty ? 'Hãy nhập nội dung review.' : null,
                    decoration: const InputDecoration(
                      labelText: 'Nội dung review',
                      hintText: 'Nhập review bằng tiếng Anh...',
                      alignLabelWithHint: true,
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 14),
                  DropdownButtonFormField<String>(
                    value: _selectedModel,
                    decoration: const InputDecoration(labelText: 'Mô hình', border: OutlineInputBorder()),
                    items: const [
                      DropdownMenuItem(value: 'logistic_regression', child: Text('Logistic Regression (khuyến nghị)')),
                      DropdownMenuItem(value: 'multinomial_naive_bayes', child: Text('Multinomial Naive Bayes')),
                      DropdownMenuItem(value: 'sgd_classifier', child: Text('SGD Classifier')),
                      DropdownMenuItem(value: 'linear_svm', child: Text('Linear SVM')),
                      DropdownMenuItem(value: 'random_forest', child: Text('Random Forest')),
                      DropdownMenuItem(value: 'decision_tree', child: Text('Decision Tree')),
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
                            : const Text('Phân loại review'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (_recommendation != null) ...[
              const SizedBox(height: 22),
              Card(
                color: resultBackground,
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_recommendation!, style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: resultColor)),
                      const SizedBox(height: 8),
                      Text('Mô hình: ' + (_modelLabel ?? '')),
                      Text(_probability == null
                          ? 'Xác suất khuyến nghị: Không có'
                          : 'Xác suất khuyến nghị: ' + _probability!.toStringAsFixed(2) + '%'),
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
