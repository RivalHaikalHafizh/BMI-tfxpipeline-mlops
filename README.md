# Submission 2: BMI Prediction 
Nama: Rival Haikal Hafizh 

Username dicoding: rivalhaikalhafizh

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [BMI](https://www.kaggle.com/datasets/yasserh/bmidataset) |
| Masalah | Sering kali manusia mengalami masalah pada berat badannya ,entah itu berat badan yang berlebihan maupun kekurangan berat badan yang mana dua hal ini dapat berdampak pada penyakit yang lain ,dan bahkan banyak sekali orang yang tidak mengetahui apakah berat badannya sudah ideal atau belum . |
| Solusi machine learning | Oleh karena itu penerapan machine learning untuk mengetahui berat badan yang ideal menjadi solusi yang baik karena kita bisa mengetahui hasil dan cepat dan tepat berdasarkan BMI (Body Mass Index) |
| Metode pengolahan | Pengolahan data yang dilakukan yaitu menormalisasi data bertipe numerik dan penerapan one hot encoding pada data yang bertipe kategorik dan merubah label menjadi tipe data integer |
| Arsitektur model | Menerapkan neural network dengan menerapkan fungsi aktivasi relu pada hidden layer ,dan aktivasi softmax pada layer outputs jumlah leyer diatur menggunakan tuner  |
| Metrik evaluasi |AUC,Precision,Recall,exampleCount,TruePositives,FalsePositives,TrueNegatives,FalseNegatives dan SparseCategoricalAccuracy merupakan metrik yang digunakan pada model ini |
| Performa model | Performa yang didapatkan pada model ini cukup baik yaitu mendapatkan  val_sparse_categorical_accuracy: 0.95 dan sparse_categorical_accuracy: 0.87|
| Opsi deployment | Machine learning ini dideploy menggunakan salah satu platfrom deployment yaitu Railway. |
| Web app | [bmi-production-54d6.up.railway.app](https://bmi-production-54d6.up.railway.app/v1/models/cc-model/metadata)|
| Monitoring |Monitoring pada proyek ini menggunakan layanan prometheus. Monitoring dapat dilihat dari setiap perubahan jumlah request yang dilakukan kepada sistem dan sistem akan menampilkan status dari setiap request yang diterima dan dihubungkan dengan Grafana sehingga menghasilkan grafik yang menarik|
