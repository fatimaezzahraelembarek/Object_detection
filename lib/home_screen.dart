import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart'; // Pour TensorFlow Lite
import 'dart:async';
import 'dart:io';

import 'package:image/image.dart' as img;

late CameraController _cameraController;
late List<CameraDescription> cameras;

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isModelReady = false;
  Timer? timer; // Pour gérer la capture périodique
  String capturedImagePath = "No image captured"; // Pour afficher le chemin de l'image capturée
  String prediction = "Waiting..."; // Prédiction du modèle
  late Interpreter interpreter;

  @override
  void initState() {
    super.initState();
    initializeCamera();
    loadModel();
  }

  // Initialisation de la caméra
  Future<void> initializeCamera() async {
    cameras = await availableCameras();
    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.medium,
    );
    await _cameraController.initialize();
    setState(() {
    });
  }

  // Chargement du modèle TFLite
  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('model/mobilenet_v1_1.0_224.tflite');
    print("TFLite model loaded successfully.");
    setState(() {
      _isModelReady = true;
    });
  }

  // Démarrer la capture périodique
  void startImageCapture() {
    timer = Timer.periodic(Duration(seconds: 2), (timer) async {
      if (_cameraController.value.isInitialized) {
        try {
          final image = await _cameraController.takePicture();
          setState(() {
            capturedImagePath = image.path;
          });
          processImage(image.path);
        } catch (e) {
          print("Erreur lors de la capture de l'image : $e");
        }
      }
    });
  }

  // Arrêter la capture
  void stopImageCapture() {
    timer?.cancel();
  }



  // Prétraiter et prédire
  Future<void> processImage(String imagePath) async {
    var input = preprocessImage(imagePath);
    // Nouvelle définition pour Mobilenet
    var output = List.filled(1001, 0.0).reshape([1, 1001]);

    // Exécuter l’inférence
    interpreter.run(input, output);
    // Interprétation des résultats
    List<double> probabilities = output[0];
    int maxIndex = 0;
    double maxProb = probabilities[0];
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    setState(() {
      prediction = output[0][0] == 1 ? "Drowsy" : "Non Drowsy";

    });

    print("Prediction: $prediction");
  }

  // Prétraitement de l'image
  List preprocessImage(String imagePath) {
    final bytes = File(imagePath).readAsBytesSync();
    final image = img.decodeImage(bytes)!;

    // Redimensionner à la taille d'entrée du modèle (par exemple : 224x224)
    final resizedImage = img.copyResize(image, width: 224, height: 224);

    // Convertir en tableau de valeurs normalisées [0, 1]
    List<List<List<double>>> input = List.generate(224, (y) {
      return List.generate(224, (x) {
        // Ici, vous remplacez l'extraction brute du pixel par la nouvelle méthode
        final pixel = resizedImage.getPixel(x, y);
        double r = pixel.r / 255.0;
        double g = pixel.g / 255.0;
        double b = pixel.b / 255.0;

        return [r, g, b];
      });
    });

    return [input];
  }

  @override
  void dispose() {
    _cameraController.dispose();
    timer?.cancel();
    interpreter.close(); // Libérer les ressources du modèle
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text("Driver Drowsiness App"),
      ),
      body: Column(
        children: [
          Expanded(
            child: CameraPreview(_cameraController),
          ),
          SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  "Prediction:",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 5),
                Text(
                  prediction,
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 18, color: Colors.blue),
                ),
                SizedBox(height: 10),
                Text(
                  "Last Captured Image Path:",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 5),
                Text(
                  capturedImagePath,
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
              ],
            ),
          ),
          SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _isModelReady ? startImageCapture : null,
                  icon: Icon(Icons.play_arrow),
                  label: Text("Start Detection"),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                    backgroundColor: _isModelReady ? Colors.green : Colors.grey,
                  ),
                ),


                ElevatedButton.icon(
                  onPressed: stopImageCapture,
                  icon: Icon(Icons.stop),
                  label: Text("Stop Detection"),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                    backgroundColor: Colors.red,
                  ),
                ),
              ],
            ),
          ),
          SizedBox(height: 20),
        ],
      ),
    );
  }
}
