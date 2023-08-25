# Laporan Proyek Machine Learning Predictive Analytics

## Domain Proyek

Proyek ini membahas mengenai permasalahan di bidang ekonomi dan bisnis. Permasalahan ini datang dari suatu perusahaan yang menangani jual beli rumah. Perusahaan melakukan pembelian rumah dari konsumen dan menjualnya kembali ke pasaran. Perusahaan tentu saja menginginkan keuntungan seoptimal mungkin. Dengan demikian, perusahaan perlu mengetahui harga rumah yang berlaku di pasar sehingga dapat membeli rumah dari konsumen dengan harga yang lebih rendah dari harga pasaran, kemudian menjualnya kembali di pasar dengan harga yang sesuai dan memberikan keuntungan. Namun, banyak sekali fitur-fitur yang perlu diperhatikan dalam penentuan harga rumah. Oleh karena itu, perusahaan membutuhkan alat otomasi prediksi harga rumah yang berlaku di pasar untuk mengejar keuntungan sebanyak-banyaknya.

Permasalahan ini sangatlah penting untuk diselesaikan. Sebagai seorang pebisnis tentu harus mengetahui strategi dalam berbisnis untuk memperoleh keuntungan yang banyak. Jika tidak, maka kerugian lah yang akan didapat. 

Menurut jurnal yang ditulis oleh [Indira Luthfiana Mulhayati 2020], dengan judul "Implementasi Machine Learning Prediksi Harga Sewa Apartemen Menggunakan Algoritma Random Forest Melalui Framework Website Flask Python", didapatkan hasil yang baik dalam memprediksi harga sewa apartemen di Jakarta menggunakan model random forest dengan akurasi 92,12%. Sedangkan untuk variabel yang sangat berpengaruh terhadap keluaran harga yaitu luas dari unit.

## Business Understanding

Model bisnis yang dilakukan oleh perusahaan adalah jual-beli rumah. Perusahaan membeli rumah dari pemilik rumah, kemudian rumah dijual kembali pada para konsumen yang membutuhkan/mencari rumah. Perusahaan menjalankan usaha berupa penjualan rumah, dan menerima pembelian rumah. Sehingga bisnis yang dijalankan merupakan distribusi rumah.

Sebagai pelaku bisnis tentu menginginkan untuk mendapatkan keuntungan sebesar-besarnya. Namun, jika harga jual terlalu mahal bagi pasaran, maka, konsumen akan melakukan pembatalan pembelian. Di lain sisi, jika harga jual terlalu murah bagi pasaran, maka, keuntungan yang akan diperoleh perusahaan amatlah kecil. 

Selain dari sisi penjualan, perusahaan perlu memikirkan strategi pada sisi pembelian rumah. Rumah yang akan dibeli oleh perusahaan harus bernilai di bawah harga pasaran. Hal ini ditujukan supaya rumah tersebut memiliki potensi laku untuk dijual karena rumah tersebut akan dijual sesuai dengan harga pasaran. Sehingga, hal ini akan berdampak baik pada perolehan keuntungan yang diterima oleh perusahaan.

Dari model bisnis yang dilakukan perusahaan, maka, perusahaan memerlukan suatu alat atau *tool* otomatis untuk memprediksi harga pasaran rumah dengan teknik *predictive modelling*. Perusahaan membutuhkan suatu alat yang dapat memprediksi harga pasaran rumah dengan ciri-ciri (fitur-fitur) tertentu. Sehingga, permasalahan bisnis pada kasus ini yaitu permasalahan regresi. Permasalahan regresi akan memberikan hasil atau nilai keluaran prediksi harga yang baik jika variabel atau fitur-fitur yang digunakan, dipilih secara tepat. Dengan demikian, keuntungan yang besar akan dapat dicapai oleh perusahaan.

### Problem Statements
- Karakteristik atau fitur apa saja yang memiliki pengaruh yang besar terhadap harga rumah di pasaran?
- Berapa harga rumah dengan karakteristik atau fitur tertentu?

### Goals
- Mengetahui karakteristik atau fitur apa saja yang memiliki pengaruh yang besar terhadap harga rumah di pasaran.
- Membuat model pembelajaran mesin yang dapat memprediksi harga rumah berdasarkan karakteristik atau fitur yang ada.
    ### Solution statements
    - Solusi yang digunakan untuk mengatasi permasalahan perusahaan jual-beli rumah adalah melakukan pengembangan model dengan tiga algoritma. Algoritma yang digunakan antara lain, decision tree, random forest regressor, dan KNN. Adapun algoritma tersebut memiliki kelemahan masing-masing. Algoritma decision tree dapat memberikan hasil yang tidak stabil meski hanya terjadi sedikit perubahan data. Algoritma random forest regressor memiliki kelemahan berupa ketidakstabilan terhadap akurasi yang dihasilkan. Algoritma random forest regressor akan menghasilkan hasil akurasi yang berbeda-beda meski diberi masukan parameter input dan data yang sama untuk eksekusi lebih dari satu kali secara berurutan. Algoritma KNN memiliki kelemahan berupa *curse dimensionality* jika dihadapkan dengan fitur yang terlalu banyak. 
    - Hyperparameter yang digunakan dalam masing-masing algoritma tersebut antara lain random_state pada algoritma decision tree, n_estimator pada algoritma random forest, n_neighbors pada algoritma KNN.
    - Metrik yang digunakan untuk mengukur seberapa baik model dalam memprediksi harga rumah di pasaran yaitu RMSE. Secara umum metrik ini digunakan untuk menghitung seberapa jauh hasil prediksi dengan nilai sebenarnya.

## Data Understanding
Data yang digunakan untuk memprediksi harga rumah sesuai harga pasaran dengan berbagai ciri-ciri atau fitur rumah, yaitu data histori penjualan rumah. Data tersebut diperoleh dari situs [Kaggle](https://www.kaggle.com/italosimoes/kc-house-data-analysis-and-predictions-95/data). Nama berkas dataset tersebut yaitu, kc_house_data.csv. Dengan format berkas berupa csv atau *(comma separated value)*. Data tersebut merupakan data riwayat penjualan rumah di King Country, Washington State, USA, pada bulan Mei 2014 hingga Mei 2015. Jumlah baris atau data yang terdapat pada dataset tersebut yaitu sebesar 21613 data. Sedangkan jumlah fitur atau kolom dari dataset tersebut adalah sebesar 21 kolom. 

### Variabel-variabel pada riwayat penjualan rumah di King Country dataset adalah sebagai berikut:
- id: notasi rumah
- date: tanggal rumah terjual
- price: harga penjualan rumah, fitur target
- bedrooms: jumlah kamar tidur pada rumah
- bathrooms: jumlah kamar mandi pada rumah, nilai .5 merupakan toilet
- sqft_living: luas *living area* dalam satuan *square feet*
- sqft_lot: luas tanah dalam satuan *square feet*
- floors: jumlah tingkat/lantai rumah
- waterfront: rumah dengan view menghadap pantai
- view: Indeks seberapa bagus pemandangan rumah tersebut
- condition: kondisinya secara keseluruhan
- grade: nilai keseluruhan yang diberikan kepada *housing-unit,* berdasarkan sistem penilaian King County
- sqft_above: luas rumah yang berada di atas permukaan tanah dalam satuan *square feet*
- sqft_basement: luas ruang-bawah-tanah dalam satuan *square feet*
- yr_built: tahun pertama kali dibangun
- yr_renovated: tahun terakhir rumah direnovasi
- zipcode: kode pos
- lat: koordinat garis lintang
- long: koordinat garis bujur
- sqft_living15: Luas *living area* untuk *nearest 15 neighbors*
- sqft_lot15: Luar area tanah untuk *nearest 15 neighbors*

Terdapat beberapa tahap yang dilakukan untuk memahami data dari dataset penjualan rumah dengan nama berkas kc_house_data.csv. Tahapan-tahapan tersebut antara lain:
1. Pengecekan Anomali Data  
     Pada tahap ini dilakukan pengidentifikasian anomali data maupun missing value terhadap data pada dataset kc_house_data.csv. Berikut penjelasan mengenai anomali yang ditemukan saat proses memahami data:  
    1.1. Anomali Data pada Atribut floor  
    Terdapat anomali data pada atribut floor. Anomali tersebut berupa nilai pecahan yang ada pada atribut tersebut seperti 3.50, dan tipe data floor berupa float. Sedangkan, banyaknya lantai pada suatu bangunan pasti bernilai bilangan bulat.  
    1.2. Anomali Data pada Atribut bathrooms  
    Dari peninjauan sekilas dataset di atas, didapatkan nilai dari fitur 'bathrooms' yaitu 0.75, 1.00, 2.00, 2.25, 2.50, 3. Anomali terjadi pada banyaknya kamar mandi yang bernilai pecahan. Meski pada keterangan fitur-fitur dijelasakan bahwa nilai .5 pada fitur bathrooms menunjukkan toilet bukan kamar mandi, namun nilai 'bathrooms' yang berkelipatan .25 seperti .25 dan .75 belum diketahui apa maksud dari nilai tersebut.  
    1.3. Anomali Data pada Atribut id  
    Dari hasil keluaran cek nilai unik pada data didapatkan bahwa atribut id memiliki nilai unik sebanyak 21436. Sedangkan total data pada dataset berjumlah 21613. Hal ini menunjukkan ketidakwajaran. Karena, nilai dari id harus unik untuk setiap data.  
    1.4. Anomali pada Tipe Data  
    Ditemukan Anomali pada tipe data beberapa atribut yang dijelaskan sebagai berikut:
    - Atribut bathrooms dan floors seharusnya bertipe data integer. Hal ini selaras dengan anomali nilai bathrooms dan floors yang sebelumnya telah dibahas, dan belum diketahui alasan terdapatnya nilai pecahan pada atribut bathrooms, dan floors.
    - Atribut yang menjelaskan mengenai luas suatu daerah seharusnya bertipe data float. Meskipun seluruh data memiliki nilai dengan bilangan bulat, seharusnya atribut yang berkaitan dengan ukuran luas bertipe data float. Hal ini dikarenakan ukuran luas bersifat kontinu. Dan pada kasus dataset ini yang terdiri dari lebih dari 20.000 data, sulit dipercaya bahwa luas dari masing-masing daerah bernilai bulat dalam satuan square-feet. Atribut-atribut yang dimaksud antara lain: sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15.  
    
    1.5. Pengecekan Nilai null pada Data  
    Pada tahap ini dilakukan pula pengecekan terhadap nilai null pada data. Namun, tidak ditemukan nilai null pada dataset seperti yang ditunjukkan pada gambar di bawah ini.
    
    1.6. Anomali Berupa Nilai 0 pada Atribut bedrooms dan bathrooms  
    Pada dataset kc_house_data.csv ditemukan nilai 0 pada variabel bathrooms maupun bedrooms. Hal ini menunjukkan ketidakwajaran data. Karena sudah sewajarnya rumah pada tahun 2014-2015 di daerah Washington, USA, setidaknya memiliki kamar tidur maupun kamar mandi.
    
2. Feature selection
    Tahap ini melakukan identifikasi dalam menentukan fitur atau variabel apa saja yang memiliki keterkaitan dengan fitur target (price). Terdapat beberapa cara untuk mendapatkan wawasan dari data. Seperti melakukan analisis univariate dengan memvisualisasikan seluruh fitur numerik (dalam kasus ini, seluruh fitur merupakan fitur numerik), dan melakukan analisis multivariate dengan memvisualisasikan keterkaitan antara satu fitur dengan fitur lainnya dalam grafik dua dimensi menggunakan fungsi pairplot() maupun dengan mengevaluasi skor keterkaitan antara satu fitur dengan fitur lainnya dengan fungsi corr().
    Berikut adalah hasil dari masing-masing proses *featur selection:*
    2.1. Univariate EDA  
    Gambar di atas memberikan wawasan sebagai berikut:
    Dari proses *univariate* EDA diperoleh informasi sebagai berikut:
    1. Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Distribusi harga rumah miring ke kanan (right-skewed).
    2. Lebih dari 50% rumah memiliki satu hingga lima kamar tidur.
    3. Sebagian besar rumah memiliki dua kamar mandi.
    4. Sebagian besar rumah memiliki luas rumah 2000 *square feet.*
    5. Sebagian besar rumah memiliki luas tanah atau *lot* sekitar satu *square feet.*
    6. Sebagian besar rumah memiliki satu atau dua tingkat/lantai.
    7. Sebagian besar rumah tidak memiliki *waterfront*.
    8. Sebagian besar rumah tidak memiliki *view*.
    9. Sebagian besar rumah memiliki nilai *condition* sebesar tiga poin.
    10. Sebagian besar rumah memiliki nilai *grade* sebesar tujuh hingga delapan poin.
    11. Sebagian besar rumah memiliki luas *above* atau *footage of house apart from basement* sebesar ratusan hingga 6100 *square feet.*
    12. Sebagian besar rumah tidak memiliki *basement*.
    13. Rumah dibangun di antara rentang tahun 1900 hingga 2015.
    14. Sebagian besar rumah tidak mengalami renovasi.
    15. Data *zipcode* rumah berada pada rentang nilai 98000 hingga 98200.
    16. Data *latitude* menggambarkan *left-skewed*.
    17. Data *longitude* menggambarkan *right-skewed*.
    18. Data variabel sqft_living15 menggambarkan *right-skewed*.
    19. Sebagian besar rumah tidak merenovasi luas *lot* atau tanah, sehingga data sqft_lot15 terbanyak berada pada nilai 0.
    20. Penjualan terbanyak berada pada bulan Mei.
    21. Penjualan pada tahun 2014 lebih banyak daripada penjualan di tahun 2015.
   
     2.2. Multivariate EDA dengan Fungsi pairplot()
    Gambar di atas merupakan baris pertama hasil menjalankan fungsi pairplot(). Keterkaitan antara tiap atribut dengan fitur target atau atribut *price* terlihat pada baris pertama visualisasi data. 
    
    Terdapat tiga penjelasan mengenai cara baca grafik antara dua atribut. Yang pertama, jika nilai pada sumbu x semakin besar dan nilai pada sumbu y semakin besar, maka kedua variabel tersebut memiliki hubungan positif. Yang kedua, jika nilai pada sumbu x semakin besar, namun nilai pada sumbu y semakin kecil, maka kedua variabel tersebut memiliki hubungan negatif. Yang ketiga, jika hasil visualisasi tidak membentuk pola maka, kedua variabel tersebut tidak berhubungan.
    
    Dari grafik di atas, fitur-fitur yang memiliki hubungan positif terhadap target (price) adalah bathrooms (kolom ke-3), sqft_living (kolom ke-4), grade (kolom ke-10), sqft_above (kolom ke-11), sqft_living15 (kolom ke-18).
    2.3. Multivariate EDA dengan Fungsi corr()
    
    Keterangan matriks:
    1. Korelasi tiap fitur terhadap fitur 'price' tertera pada baris pertama.
    2. Koefisien korelasi mendekati 1 atau -1, menunjukkan kuatnya hubungan antara dua variabel (menggambarkan korelasi positif atau negatif).
    3. Semakin mendekati 0 nilai koefisien korelasi, maka semakin kecil hubungan antara dua variabel tersebut. 
    4. Semakin pekat warna merah pada persegi menunjukkan semakin kuat korelasi positif antara dua fitur.
    5. Semakin pekat warna biru pada persegi menunjukkan semakin kuat korelasi negatif antara dua fitur.
    6. Semakin terang warna biru, semakin menunjukkan korelasi yang lemah antara dua fitur.
    7. Berikut adalah daftar fitur yang memiliki hubungan kuat dengan fitur 'price' atau memiliki warna berupa oranye atau mendekati warna merah atau memiliki nilai korelasi lebih dari 0,5: bathrooms (0,53), sqft_living (0,7), grade (0,67), sqft_above (0,61), sqft_living15 (0,59).

## Data Preparation
Pada tahap ini dilakukan pembagian data dan transformasi data. Berikut penjelasannya:
1. Feature engineering  
    Pada tahap ini, variabel baru dibuat berdasarkan data yang ada. Dalam kasus ini, variabel month dan year dibuat berdasarkan variabel date. Dengan demikian, data tersebut dapat dimanfaatkan untuk mengetahui keterkaitan month dan year terhadap harga pasar rumah. Kemudian, variabel date dihapus karena sudah digantikan dengan variabel month dan year. Selain itu, variabel id juga dihapus karena tidak ada keterkaitan nilai id dengan harga penjualan rumah.
2. Data cleaning
    Pada tahap memahami data, terdapat anomali data pada atribut bedrooms dan bathrooms. Anomali tersebut berupa ditemukannya nilai 0 pada kedua atribut tersebut. Sehingga, data dengan atribut bathrooms maupun bedrooms yang bernilai 0, perlu dihapus dari dataset.       
3. Train-Test-Split  
    Pada tahap ini data dibagi menjadi dua bagian, yaitu, data latih dan data uji. Dengan demikian, model dapat diuji seberapa bagus performa atau akurasi dalam generalisasi data baru. Selain itu, pembagian data dilakukan sebelum proses transformasi data. Hal ini ditujukan supaya transformasi hanya diterapkan pada data latih terlebih dahulu, sehingga tidak terjadi kebocoran data atau *data leakage*. Karena seharusnya model tidak memiliki informasi mengenai distribusi pada data uji. 
4. Standarisasi
    Standarisasi ditujukan supaya data tidak memiliki penyimpangan nilai yang besar. Dengan standarisasi, maka skala nilai data relatif sama, sehingga mempermudah konvergensi model algoritma. Proses yang dilakukan dalam standarisasi yakni mengurangi setiap nilai pada kumpulan data dengan nilai rata-rata, kemudian dibagi dengan deviasi standar. Pada kasus ini, data yang dimiliki berupa data numerik. Sehingga, standarisasi yang diterapkan yaitu berupa standardScaler.   
    Proses standarisasi yang pertama, baru dilakukan pada data latih.     
    
    Selanjutnya proses standarisasi juga perlu dilakukan pada data uji. Hal ini ditujukan supaya skala antara data latih dan data uji sama besar. Sehingga, proses evaluasi dapat dilakukan.  

## Modeling
Model yang digunakan untuk menyelesaikan permasalahan prediksi harga rumah yaitu terdiri dari tiga model, antara lain: decision tree, random forest regressor, K-Nearest Neighbor. Masing-masing penjelasan dari model adalah sebagai berikut:
1. Decision tree  
    Decision tree merupakan model yang mengadopsi struktur seperti pohon. Struktur tersebut terdiri dari node internal yang mewakili tes atau atribut, cabang yang mewakili hasil dari atribut, jalur dari akar ke daun yang mewakili aturan klasifikasi. Kelebihan dari decision tree yaitu mampu melakukan break-down process untuk mengambil keputusan kompleks menjadi lebih sederhana. Namun, kelemahan dari decision tree yaitu, decision tree dapat mengalami overlap ketika mendapati terlalu banyak kelas maupun kriteria. Dengan demikian, decision tree membutuhkan waktu lebih lama dalam pengambilan keputusan dan akan memakan penyimpanan sangat banyak. Tahap yang dilakukan dalam pemodelan decision tree ini adalah memanggil fungsi DecisionTreeRegressor(), dan menentukan nilai dari hiperparameter random_state. Menentukan hiperparameter random_state ditujukan supaya hasil pembagian dataset dilakukan secara konsisten atau memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka dataset akan dibagi secara tidak konsisten tiap model dijalankan. Kemudian memanggil fungsi fit() untuk melatih model dengan masukan yang diberikan. Kemudian fungsi score() digunakan untuk menghitung akurasi model pada data latih. Hasil akurasi pada data latih di model decision tree mencapai 0,994.
2. Random Forest Regressor  
    Model random forest adalah algoritma yang konsep penyelesaian masalahnya yaitu menggunakan struktur pohon yang banyak. Hasil yang prediksi yang dikeluarkan dari model random forest merupakan rata-rata prediksi seluruh pohon dalam model ensemble. Kelebihan dari algoritma ini adalah algoritma ini mampu mengatasi *missing value* maupun *noise* dalam jumlah yang besar dengan baik. Namun, kelemahan dari algoritma ini adalah interpretasi yang sulit membutuhkan tuning model yang tepat untuk data. Tahap yang dilakukan dalam pemodelan random forest regressor ini adalah memanggil fungsi RandomForestRegressor() dengan hiperparameter n_estimators. Hiperparameter n_estimators adalah hyperparameter yang digunakan untuk mengatur jumlah pohon pada algoritma yang bekerja. Dengan hiperparameter bernilai 500, didapatkan hasil akurasi *training* sebesar 0,941.
3. K-Nearest Neighbor  
    Model KNN adalah algoritma yang konsep penyelesaiannya berdasarkan nilai k tetangga terdekat. Algoritma ini menggunakan kesamaan fitur dalam memprediksi nilai data baru. Penentuan nilai k sangatlah berpengaruh terhadap akurasi model. Dengan menginisialisasikan nilai k dengan angka yang besar akan membantu menghindari overfitting. Namun, dengan menginisialisasikan nilai k terlalu tinggi akan mengakibatkan underfitting. Sedangkan, dengan menginisialisasikan nilai k terlalu rendah akan mengakibatkan overfitting. Algoritma KNN memiliki kelemahan berupa *curse dimensionality* jika menghadapi kasus dengan fitur-fitur terlalu banyak. Akan tetapi, kelebihan dari algoritma KNN adalah algoritma ini tangguh dalam menghadapi dataset latih yang *noisy* dan menghasilkan keluaran yang efektif bila data latihnya berjumlah besar. Tahap yang dilakukan dalam pemodelan KNN ini adalah memanggil fungsi KNeighborsRegressor() dengan hiperparameter n_neighbors. Hiperparameter n_neighbors adalah hyperparameter yang digunakan untuk menentukan jumlah tetangga terdekat untuk perhitungan algoritma KNN. Dengan hiperparameter bernilai 2, didapatkan hasil akurasi *training* sebesar 0,826.

Sehingga pada kasus ini performa terbaik dalam menunjukkan akurasi model terhadap kasus prediksi harga rumah dicapai oleh algoritma decision tree. Sedangkan, performa terburuk terjadi ketika menggunakan algoritma KNN.

## Evaluation
Langkah selanjutnya yaitu menilai performa generalisasi data. Matriks yang digunakan untuk mengevaluasi ketiga model pada kasus ini adalah RMSE. RMSE merupakan singkatan dari *Root Mean Square Error*. Formula dari RMSE itu sendiri adalah sebagai berikut:  

Cara perhitungan RMSE yakni mengukur kuadrat dari selisih nilai prediksi dan nilai sesungguhnya, jumlahkan masing-masing hasil hitung tersebut, kemudian diakarkan.

Adapun tahapan untuk melakukan evaluasi dengan menggunakan matriks RMSE adalah memanggil fungsi-fungsi seperti predict(), mean_squared_error(), dan np.sqrt(). Masing-masing dari fungsi tersebut ditujukan sebagai berikut:
1. Fungsi predict() digunakan untuk memprediksi hasil observasi data uji.
2. Fungsi mean_squared_error() digunakan untuk menghitung nilai akurasi dari hasil prediksi terhadap nilai sesungguhnya.
3. Fungsi np.sqrt() digunakan untuk mengakar kuadrat hasil dari perhitungan mean-squared-error.

Berikut adalah hasil evaluasi ketiga model untuk masing-masing proses latihan maupun pengujian:  

Berdasarkan matriks RMSE, hasil terbaik didapatkan oleh model Random Forest Regressor sebesar  2832,17 pada proses pelatihan dan 7036.02 pada proses pengujian. Kemudian disusul oleh K-Nearest Neighbor Regressor sebesar 4868.9 pada proses pelatihan dan 7925.34 pada proses pengujian. Sedangkan, pada model Decision Tree Regressor mendapatkan nilai RMSE sebesar 883,474 pada proses pelatihan dan 9515,05 pada proses pengujian. Dengan demikian, maka, model terbaik untuk menangani kasus prediksi harga rumah yaitu Random Forest Regressor dengan fitur-fitur yang memiliki pengaruh besar terhadap model berupa, bathrooms, sqft_living, grade, sqft_above, sqft_living15.

Selanjutnya dilakukan pengujian model dengan data uji.  
Dari proses pengujian, didapatkan hasil yang tidak jauh berbeda dari nilai asli untuk maisng-masing model. Namun, hasil uji menunjukkan model yang memiliki nilai paling dekat dengan nilai sebenarnya ditunjukkan oleh model K-Nearest Neighbor.
