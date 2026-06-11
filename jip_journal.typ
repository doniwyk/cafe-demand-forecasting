// ============================================================
// JIP (Jurnal Informatika Polinema) — Typst Template
// Compile with: typst compile jip_journal.typ
// ============================================================

// ---------- Page Setup ----------
#set page(
  paper: "a4",
  margin: (top: 30mm, left: 30mm, bottom: 20mm, right: 20mm),
  header: none,
  footer: none,
)

// ---------- Fonts & Base Style ----------
#set text(font: "Times New Roman", size: 10pt, lang: "id")
#set par(justify: true, leading: 0.65em, spacing: 0.65em, first-line-indent: 1.5em)

// ---------- Heading Styles ----------
#show heading.where(level: 1): it => {
  set par(first-line-indent: 0em)
  v(0.5em)
  text(size: 10pt, weight: "bold")[#it.body]
  v(0.2em)
}
#show heading.where(level: 2): it => {
  set par(first-line-indent: 0em)
  v(0.4em)
  text(size: 10pt, weight: "bold")[#it.body]
  v(0.2em)
}
#show heading.where(level: 3): it => {
  set par(first-line-indent: 0em)
  v(0.3em)
  text(size: 10pt, style: "italic", weight: "bold")[#it.body]
  v(0.1em)
}

// ---------- Figure / Table Captions ----------
#show figure: it => {
  v(0.5em)
  align(center, text(size: 8pt)[#it.supplement #it.numbering. #it.caption.body])
  v(0.3em)
  it.body
  v(0.5em)
}

// ============================================================
// HEADER — Journal name + ISSN
// ============================================================
#grid(
  columns: (1fr, 1fr),
  align(left, text(style: "italic", weight: "bold")[JIP (Jurnal Informatika Polinema)]),
  align(right)[ISSN: 2614-6371 #h(1em) E-ISSN: 2407-070X],
)
#line(length: 100%, stroke: 0.5pt)
#v(1em)

// ============================================================
// TITLE
// ============================================================
#align(center)[
  #text(size: 14pt, weight: "bold")[
    Penerapan Algoritma Machine Learning dalam Peramalan Permintaan Produk pada Coffee Shop Berbasis Data Historis Transaksi
  ]
]
#v(0.5em)

// ============================================================
// AUTHORS
// ============================================================
#align(center)[
  #text(weight: "bold")[ Endah Septa Sintiya#super[1]]
  #text(weight: "bold")[Yoppy Yunhasnawa#super[2]]
  #text(weight: "bold")[Doni Wahyu Kurniawan#super[3]]
]
#v(0.3em)
#align(center)[
  #super[1] #super[2]Program Studi Teknik Informatika, Jurusan Teknologi Informasi, #super[3]Politeknik Negeri Malang, Indonesia\
  #super[1]doniwyk\@gmail.com
  #super[2]xxx\@xxx.xxx
  #super[3]xxx\@xxx.xxx
]
#v(1em)
#line(length: 100%, stroke: 0.5pt)
#v(0.5em)

// ============================================================
// ABSTRACT (single column)
// ============================================================
#align(center)[#text(weight: "bold")[Abstrak]]
#v(0.3em)

Industri Food and Beverage (F&B) menghadapi tantangan inefisiensi inventaris yang signifikan, terutama pada segmen kedai kopi dengan bahan baku perishable. Husgendam Coffee, sebuah kedai kopi skala menengah di Kota Batu, masih mengandalkan sistem pengadaan manual meskipun memiliki data historis transaksi yang kaya. Penelitian ini mengembangkan model peramalan permintaan harian menggunakan algoritma Extreme Gradient Boosting (XGBoost) berbasis data Point-of-Sale periode Januari 2022 hingga Mei 2026 (94.611 item terjual). Model global tunggal melatih 61 item secara bersamaan menggunakan 31 fitur rekayasa temporal dan recency. Evaluasi menggunakan expanding window backtest 8-jendela menghasilkan MAE sebesar 0,745 cangkir per item-hari, melampaui target MAE < 1,0 dengan margin 25%. Model mengungguli baseline naive sebesar 47% dan diintegrasikan dengan struktur Bill of Materials untuk menghasilkan rekomendasi kebutuhan bahan baku. Penelitian ini menunjukkan bahwa XGBoost mampu menjadi solusi peramalan yang efektif dan applicable pada konteks UMKM F&B di Indonesia.

#v(0.4em)
#text(weight: "bold")[Kata kunci:] machine learning, peramalan permintaan, XGBoost, coffee shop, manajemen inventaris, food waste
#v(0.5em)
#line(length: 100%, stroke: 0.5pt)
#v(0.5em)

// ============================================================
// BODY — Two-Column Layout
// ============================================================
#columns(2, gutter: 5mm)[

// ---- 1. PENDAHULUAN ----
= 1. Pendahuluan

Industri Food and Beverage (F&B) menghadapi paradoks pertumbuhan: ekspansi bisnis yang pesat beriringan dengan inefisiensi lingkungan yang masif. Pada tahun 2022, sektor layanan makanan dilaporkan mengakumulasi 290 juta ton limbah makanan secara global, merepresentasikan hambatan struktural dalam pencapaian Sustainable Development Goals (SDG) 12.3 (United Nations Environment Programme, 2024). Di Indonesia, sektor HOREKA berkontribusi sebesar 20% dari total limbah pangan non-rumah tangga (Bappenas, 2021).

Kompleksitas manajemen inventaris teramplifikasi pada industri kedai kopi, di mana karakteristik bahan baku memiliki tingkat perishability yang tinggi (Chae et al., 2024; Hulaini et al., 2025). Deviasi minor dalam perencanaan stok menghasilkan dampak ganda: kerugian material akibat pembusukan bahan baku (overstock) dan hilangnya kesempatan penjualan saat kehabisan stok (stockout).

Machine Learning (ML) menawarkan pendekatan analitik prediktif yang lebih robust. Algoritma seperti Extreme Gradient Boosting (XGBoost) mampu menangkap pola non-linear kompleks tanpa asumsi parametrik yang ketat (Schmidt et al., 2022). Studi empiris menunjukkan ML umumnya mengungguli metode baseline dalam akurasi prediksi permintaan restoran (Chae et al., 2024).

Husgendam Coffee, sebagai representasi kedai kopi skala menengah di Kota Batu, masih mengandalkan sistem pengadaan manual meskipun memiliki data historis transaksi tersimpan sistematis pada sistem Point-of-Sale (PoS). Kesenjangan antara ketersediaan data dan praktik manajerial konvensional menciptakan peluang signifikan untuk intervensi teknologi berbasis ML.

Penelitian ini berfokus pada peramalan permintaan (demand forecasting) yang berbeda secara konseptual dari peramalan penjualan (sales forecasting). Permintaan merepresentasikan kuantitas produk yang diinginkan pelanggan, terlepas dari ketersediaan stok, sedangkan penjualan hanya mencerminkan kuantitas yang terealisasi dalam transaksi (Birkmaier et al., 2024).

Pemilihan XGBoost didasarkan pada tiga pertimbangan teknis: kemampuan menangani non-linearity tanpa feature engineering kompleks, robustness terhadap outlier yang umum pada data transaksi F&B, dan interpretabilitas model yang mendukung justifikasi keputusan bisnis (Alt, 2021; González Morales & Cavero Rubio, 2023).

// ---- 2. METODE ----
= 2. Metode

Penelitian ini menggunakan pendekatan Iterative and Incremental Development dalam mengembangkan sistem peramalan permintaan berbasis machine learning. Pendekatan ini dipilih karena pengembangan model ML memerlukan eksperimen berulang dalam feature engineering, pemilihan algoritma, dan hyperparameter tuning.

== 2.1 Pengumpulan Data

Data historis transaksi diperoleh dari sistem PoS Husgendam Coffee dengan periode 1 Januari 2022 hingga 25 Mei 2026, terdiri dari 94.611 data item terjual. Data diekspor dalam format CSV dengan struktur transaksi individual meliputi: tanggal, nomor struk, kategori, SKU, nama barang, dan kuantitas.

Setelah transformasi, data diagregasi ke level harian per item (item-hari), menghasilkan dataset final siap latih untuk 61 item aktif.

== 2.2 Feature Engineering

Sebanyak 31 fitur rekayasa diekstraksi dari data historis, dikelompokkan ke dalam empat kategori:

- *Fitur Temporal:* hari dalam minggu, bulan, indikator akhir pekan
- *Fitur Lag:* kuantitas terjual pada lag 1, 7, dan 14 hari
- *Fitur Recency:* Days_Since_Last_Sale — fitur dengan Mutual Information tertinggi (0,656 nats)
- *Fitur Cross-Item:* agregat penjualan kafe pada periode sebelumnya

Analisis ablasi kelompok fitur memvalidasi kontribusi masing-masing kelompok terhadap performa model.

== 2.3 Pemodelan XGBoost

Model global tunggal dipilih berdasarkan hasil Exploratory Data Analysis yang menunjukkan 26% item memiliki data historis terbatas dan zero-inflation sebesar 64%. Fungsi objektif count:poisson digunakan untuk menangani distribusi permintaan diskrit dengan dominasi nilai nol.

Konfigurasi model final menggunakan hyperparameter yang dioptimasi melalui Grid Search dengan 8 ronde backtest, meliputi: learning_rate, max_depth, n_estimators, subsample, dan colsample_bytree.

== 2.4 Evaluasi Model

Evaluasi menggunakan expanding window backtest dengan 8 jendela waktu. Metrik yang digunakan meliputi MAE, RMSE, Bias, R² overall, R² non-zero, dan wMAPE. Target kinerja ditetapkan MAE < 1,0 cangkir per item-hari.

== 2.5 Integrasi Newsvendor

Output peramalan diintegrasikan dengan model newsvendor untuk menghitung buffer persediaan optimal:

$ Q^* = hat(mu) + z_(C R) dot sigma_epsilon $

dengan CR = c_u / (c_u + c_o), di mana c_u adalah margin lost per cangkir dan c_o adalah spoilage cost per cangkir.

// ---- 3. HASIL DAN PEMBAHASAN ----
= 3. Hasil dan Pembahasan

== 3.1 Performa Model

Hasil backtest expanding window 8-jendela ditampilkan pada Tabel 1. Model mencapai MAE 0,745 cangkir per item-hari, melampaui target 25% di bawah ambang batas 1,0.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt,
    table.header(
      text(weight: "bold")[Metrik],
      text(weight: "bold")[Nilai],
      text(weight: "bold")[Interpretasi],
    ),
    [MAE], [0,745], [Deviasi ~45 cangkir/hari seluruh kafe],
    [RMSE], [1,37], [28% di bawah std. aktual (1,91)],
    [Bias], [−0,002], [Tidak ada kecenderungan sistematis],
    [R² overall], [0,48], [Prediksi nol akurat pada hari tanpa penjualan],
    [R² non-zero], [0,18], [Di atas batas teoritis 0,12],
    [wMAPE], [51,1%], [Konsekuensi skala diskrit 1–3 cangkir],
  ),
  caption: [Hasil Backtest Expanding Window 8-Jendela],
)

#h(1.5em) Nilai CV_error = 0,94 mengindikasikan ketidakpastian hampir setara dengan sinyal. Batas teoritis R²_max ≈ 1 − 0,94² ≈ 0,12; R² non-zero aktual 0,18 berada di atas batas ini, mengkonfirmasi bahwa model telah mengekstraksi informasi maksimal yang tersedia dalam data.

== 3.2 Perbandingan dengan Baseline

Model XGBoost mengungguli seluruh baseline naive dengan perbaikan MAE sebesar 47% dibandingkan baseline Day-of-Week (DOW) Median. Keterbatasan baseline DOW terletak pada ketidakmampuannya beradaptasi terhadap tren pertumbuhan 4× selama periode observasi, tidak membedakan status recency item, dan tidak memodelkan interaksi antar variabel.

== 3.3 Dominasi Fitur Recency

Analisis Mutual Information menunjukkan Days_Since_Last_Sale memiliki nilai 0,656 nats, tiga kali lipat fitur berikutnya. Studi ablasi memperkuat temuan ini: penghapusan fitur recency meningkatkan MAE sebesar 0,517 (71%).

Dominasi recency dapat dijelaskan melalui karakteristik zero-inflation 64% pada data. Fitur ini menjawab pertanyaan mendasar "apakah item akan terjual hari ini?" yang lebih sering dan lebih prediktif. Melalui fungsi objektif count:poisson, ketika Days_Since_Last_Sale bernilai tinggi, model mengasosiasikan kondisi tersebut dengan λ ≈ 0 sehingga P(Y=0) mendekati 1.

== 3.4 Integrasi Model Newsvendor

Buffer persediaan dihitung dengan service level diferensial per kelas ABC: 95% untuk Kelas A (27 item, 70% volume), 90% untuk Kelas B, dan 85% untuk Kelas C. Total buffer mingguan yang direkomendasikan adalah 730 cangkir, dengan Kelas A mendominasi 73% dari total buffer.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    table.header(
      text(weight: "bold")[Skenario],
      text(weight: "bold")[CR],
      text(weight: "bold")[z],
      text(weight: "bold")[Buffer/Minggu],
    ),
    [Stockout 5× lebih mahal], [83%], [0,96], [624 cangkir],
    [Stockout 3× lebih mahal], [75%], [0,67], [468 cangkir],
    [Stockout 2× lebih mahal], [67%], [0,44], [312 cangkir],
    [Biaya sama], [50%], [0], [0 cangkir],
  ),
  caption: [Sensitivitas Buffer terhadap Critical Ratio],
)

== 3.5 Analisis Stratifikasi per Kelas ABC

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    stroke: 0.5pt,
    table.header(
      text(weight: "bold")[Kelas],
      text(weight: "bold")[Rata-rata/Hari],
      text(weight: "bold")[MAE],
      text(weight: "bold")[MAPE],
    ),
    [A], [3,2 cangkir], [1,16], [68%],
    [B], [1,5 cangkir], [0,57], [64%],
    [C], [0,6 cangkir], [0,26], [56%],
  ),
  caption: [Metrik Evaluasi per Kelas ABC],
)

#h(1.5em) MAE meningkat seiring volume (wajar secara matematis), namun rasio MAE/volume menurun dari 0,43 ke 0,36, mengindikasikan peningkatan presisi relatif pada item high-volume. Rasio Error_Std/MAE yang konsisten sekitar 2,8 di seluruh kelas menunjukkan distribusi error yang serupa.

// ---- 4. KESIMPULAN ----
= 4. Kesimpulan

Penelitian ini berhasil menerapkan algoritma XGBoost untuk peramalan permintaan harian di Husgendam Coffee dengan hasil yang melampaui target kinerja. Model global tunggal untuk 61 item menghasilkan MAE 0,745 cangkir per item-hari (margin 25% di bawah target), RMSE 28% lebih rendah dari standar deviasi data aktual, dan bias mendekati nol (−0,002) yang mengkonfirmasi tidak adanya kecenderungan sistematis.

Model XGBoost memangkas sekitar 40% dari gap antara strategi pembaruan rekursif dan batas atas teoritis, dengan fitur recency (Days_Since_Last_Sale) sebagai prediktor paling kritis pada data zero-inflation tinggi. Integrasi dengan Bill of Materials dan model newsvendor menghasilkan rekomendasi pengadaan bahan baku yang actionable dengan buffer 730 cangkir per minggu.

Untuk penelitian lanjutan disarankan: (1) mengintegrasikan variabel eksternal seperti cuaca dan event lokal untuk meningkatkan R² non-zero, (2) mengeksplorasi ensemble XGBoost dengan model probabilistik seperti Temporal Fusion Transformer untuk item dengan zero-inflation ekstrem, dan (3) memvalidasi pipeline pada UMKM F&B lain untuk mengkonfirmasi generalisabilitas pendekatan.

// ---- DAFTAR PUSTAKA ----
= Daftar Pustaka

#set par(first-line-indent: 0em, hanging-indent: 1.5em, justify: true)

Alt, R. (2021). Digital transformation in the restaurant industry: Current developments and implications. _Journal of Smart Tourism_, 1(1), 69–74. https://doi.org/10.52255/smarttourism.2021.1.1.9

Bappenas. (2021). _Laporan Kajian Food Loss and Waste di Indonesia_.

Birkmaier, A., Imeri, A., & Reiner, G. (2024). Improving supply chain planning for perishable food: Data-driven implications for waste prevention. _Journal of Business Economics_, 94(6), 1–36. https://doi.org/10.1007/s11573-024-01191-x

Chae, B. (Kevin), Sheu, C., & Park, E. O. (2024). The value of data, machine learning, and deep learning in restaurant demand forecasting. _Decision Support Systems_, 184. https://doi.org/10.1016/j.dss.2024.114291

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 785–794. https://doi.org/10.1145/2939672.2939785

González Morales, M., & Cavero Rubio, J. A. (2023). Impact of digitalization of sales on the profitability of the restaurant industry during COVID-19. _Economies_, 11(11). https://doi.org/10.3390/economies11110283

Groene, N., & Zakharov, S. (2024). Introduction of AI-based sales forecasting: How to drive digital transformation in food and beverage outlets. _Discover Artificial Intelligence_, 4(1). https://doi.org/10.1007/s44163-023-00097-x

Mitra, A., et al. (2022). A comparative study of demand forecasting models for a multi-channel retail company. _Proceedings of ICMLCA_, 1–6.

Mustapha, A., & Sithole, M. (2025). Forecasting retail sales using machine learning models. _International Journal of Data Science_, 12(1), 45–58.

Nasseri, M., et al. (2023). Tree-based ensemble methods vs. deep learning for perishable product demand forecasting. _Computers & Industrial Engineering_, 175, 108892.

Nanda Rosya, F. (2024). Optimalisasi stok UMKM menggunakan algoritma machine learning. _Jurnal Teknologi Informasi_, 10(2), 112–120.

Rodrigues, P., et al. (2024). Machine learning-based procurement optimization for food waste reduction. _Sustainable Production and Consumption_, 45, 301–315.

Schmidt, A., et al. (2022). Machine learning based restaurant sales forecasting. _Machine Learning with Applications_, 9, 100371.

Turker, H. (2025). Integrating ML models for campus canteen food waste reduction. _Journal of Cleaner Production_, 420, 138450.

United Nations Environment Programme. (2024). _Food Waste Index Report 2024_. UNEP.

] // end columns
