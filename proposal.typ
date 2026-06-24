// ============================================================
// Template Proposal Tugas Akhir — Politeknik Negeri Malang
// Compile: typst compile proposal.typ
// ============================================================

// ---------- Configuration ----------
#let conf = (
  title: "Penerapan Algoritma Machine Learning dalam Peramalan Permintaan Produk pada Coffee Shop Berbasis Data Historis Transaksi",
  authors: (
    (name: "Doni Wahyu Kurniawan", role: "Mahasiswa", email: "doniwyk@gmail.com"),
    (name: "Endah Septa Sintiya", role: "Dosen Pembimbing satu", email: ""),
    (name: "Yoppy Yunhasnawa", role: "Dosen Pembimbing dua", email: ""),
  ),
  institution: "Politeknik Negeri Malang",
  department: "Jurusan Teknologi Informasi",
  program: "Program Studi Teknik Informatika",
  year: datetime.today().year(),
)

// ---------- Page Setup ----------
#set document(title: conf.title, author: conf.authors.map(a => a.name).join(", "))
#set page(
  paper: "a4",
  margin: (top: 30mm, left: 30mm, bottom: 25mm, right: 25mm),
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 8pt, font: ("Times New Roman", "Noto Sans"), fill: luma(80))
      grid(
        columns: (1fr, 1fr),
        align(left)[Proposal Tugas Akhir],
        align(right)[#conf.institution],
      )
      v(2pt)
      line(length: 100%, stroke: 0.4pt)
    }
  },
  footer: context {
    set text(size: 9pt, font: ("Times New Roman", "Noto Sans"))
    let p = counter(page).get().first()
    align(center)[#p]
  },
)

// ---------- Fonts & Base Style ----------
#set text(font: ("Times New Roman", "Noto Sans"), size: 11pt, lang: "id")
#set par(justify: true, leading: 1.2em, spacing: 0.6em, first-line-indent: 1.25cm)
#set page(numbering: "i")

// ---------- Heading Styles ----------
#show heading: set par(first-line-indent: 0em)

// Heading level 1: BAB I style
#show heading.where(level: 1): it => {
  set text(size: 12pt, weight: "bold")
  v(1.5em)
  align(center)[#it.body]
  v(0.5em)
}

// Heading level 2: X.Y style
#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold")
  v(1em)
  it.body
  v(0.3em)
}

// Heading level 3: X.Y.Z style
#show heading.where(level: 3): it => {
  set text(size: 11pt, weight: "bold", style: "italic")
  v(0.5em)
  it.body
  v(0.2em)
}

// Heading level 4
#show heading.where(level: 4): it => {
  set text(size: 10.5pt, weight: "bold", style: "italic")
  v(0.5em)
  it.body
  v(0.2em)
}

// ---------- Figure / Table Captions ----------
#show figure: it => {
  set par(first-line-indent: 0em, justify: true)
  v(0.5em)
  align(center)[
    #set text(size: 10pt)
    #text(weight: "bold")[#it.supplement #it.numbering] #it.caption.body
  ]
  v(0.3em)
  it.body
  v(0.5em)
}

// ---------- Auto Table of Contents ----------
#show outline.entry.where(level: 1): it => {
  v(0.2em)
  strong(it)
}
#show outline.entry.where(level: 2): it => {
  it
}
#show outline.entry.where(level: 3): it => {
  set text(size: 10pt)
  it
}

// ---------- Equations ----------
#set math.equation(numbering: "(1)")

// ---------- Ordered / Unordered Lists ----------
#set enum(indent: 1em, body-indent: 0.5em)
#set list(indent: 1em, body-indent: 0.5em)


// ============================================================
// COVER PAGE
// ============================================================
#page(footer: none, header: none)[
  #set par(first-line-indent: 0em)
  #align(center)[
    #v(2cm)
    #text(size: 14pt, weight: "bold", tracking: 2pt)[PROPOSAL]
    #v(0.2cm)
    #text(size: 12pt, tracking: 1pt)[TUGAS AKHIR]
    #v(1.5cm)
    #text(size: 16pt, weight: "bold")[#conf.title]
    #v(2cm)
    #text(size: 11pt)[Oleh]
    #v(0.8cm)
    #text(size: 12pt, weight: "bold")[#conf.authors.at(0).name]
    #text(size: 10pt)[\ #conf.authors.at(0).email]
    #v(2cm)
    #text(size: 11pt)[Dosen Pembimbing:]
    #v(0.3cm)
    #text(size: 11pt)[
      #conf.authors.at(1).name \
      #conf.authors.at(2).name
    ]
    #v(2cm)
    #text(size: 11pt)[#conf.program]
    #v(0.2cm)
    #text(size: 11pt)[#conf.department]
    #v(0.2cm)
    #text(size: 12pt, weight: "bold")[#conf.institution]
    #v(0.2cm)
    #text(size: 11pt)[#conf.year]
  ]
]


// ============================================================
// COVER PAGE 2 — Supervisor Approval
// ============================================================
#page(footer: none, header: none)[
  #set par(first-line-indent: 0em)
  #align(center)[
    #text(size: 14pt, weight: "bold", tracking: 2pt)[PROPOSAL TUGAS AKHIR]
    #v(0.5cm)
    #text(size: 12pt, weight: "bold")[#conf.title]
    #v(1cm)
  ]

  #grid(
    columns: (1fr, 1fr),
    gutter: 2cm,
    // Left: Supervisor 1
    [
      #set par(first-line-indent: 0em)
      #align(center)[
        #text(weight: "bold")[Dosen Pembimbing I]
        #v(3cm)
        #line(length: 60%, stroke: 0.4pt)
        #v(0.2cm)
        #text(size: 10pt)[#conf.authors.at(1).name]
      ]
    ],
    // Right: Supervisor 2
    [
      #set par(first-line-indent: 0em)
      #align(center)[
        #text(weight: "bold")[Dosen Pembimbing II]
        #v(3cm)
        #line(length: 60%, stroke: 0.4pt)
        #v(0.2cm)
        #text(size: 10pt)[#conf.authors.at(2).name]
      ]
    ],
  )

  #v(2cm)
  #align(center)[
    #text(size: 11pt)[Mengetahui,]
    #v(2cm)
    #grid(
      columns: (1fr, 1fr),
      gutter: 2cm,
      [
        #set par(first-line-indent: 0em)
        #align(center)[
          #text(weight: "bold")[Ketua Jurusan]
          #v(3cm)
          #line(length: 60%, stroke: 0.4pt)
          #v(0.2cm)
          #text(size: 10pt)[...........................................]
        ]
      ],
      [
        #set par(first-line-indent: 0em)
        #align(center)[
          #text(weight: "bold")[Kaprodi Teknik Informatika]
          #v(3cm)
          #line(length: 60%, stroke: 0.4pt)
          #v(0.2cm)
          #text(size: 10pt)[...........................................]
        ]
      ],
    )
  ]
]


// ============================================================
// ABSTRACT + ABSTRAK
// ============================================================
#page(footer: none)[
  #set par(first-line-indent: 0em)
  #align(center)[#text(size: 12pt, weight: "bold")[ABSTRAK]]
  #v(0.5em)

  Industri Food and Beverage (F&B) menghadapi tantangan inefisiensi inventaris yang signifikan, terutama pada segmen kedai kopi dengan bahan baku perishable. Husgendam Coffee, sebuah kedai kopi skala menengah di Kota Batu, masih mengandalkan sistem pengadaan manual meskipun memiliki data historis transaksi yang kaya. Penelitian ini mengembangkan model peramalan permintaan harian menggunakan algoritma Extreme Gradient Boosting (XGBoost) berbasis data Point-of-Sale periode Januari 2022 hingga Mei 2026 (94.611 item terjual). Model global tunggal melatih 61 item secara bersamaan menggunakan 31 fitur rekayasa temporal dan recency. Evaluasi menggunakan expanding window backtest 8-jendela menghasilkan MAE sebesar 0,745 cangkir per item-hari, melampaui target MAE < 1,0 dengan margin 25%. Model mengungguli baseline naive sebesar 47% dan diintegrasikan dengan struktur Bill of Materials untuk menghasilkan rekomendasi kebutuhan bahan baku. Penelitian ini menunjukkan bahwa XGBoost mampu menjadi solusi peramalan yang efektif dan applicable pada konteks UMKM F&B di Indonesia.

  #v(0.5em)
  *Kata kunci:* machine learning, peramalan permintaan, XGBoost, coffee shop, manajemen inventaris, food waste

  #v(1.5em)
  #line(length: 100%, stroke: 0.4pt)
  #v(1em)

  #align(center)[#text(size: 12pt, weight: "bold")[ABSTRACT]]
  #v(0.5em)

  The Food and Beverage (F&B) industry faces significant inventory inefficiency challenges, particularly in the coffee shop segment with perishable raw materials. Husgendam Coffee, a mid-sized coffee shop in Batu City, still relies on manual procurement despite having rich transactional data. This research develops a daily demand forecasting model using the Extreme Gradient Boosting (XGBoost) algorithm based on Point-of-Sale data from January 2022 to May 2026 (94,611 items sold). A single global model simultaneously trains 61 items using 31 temporal and recency engineered features. Evaluation using 8-window expanding window backtest achieves MAE of 0.745 cups per item-day, surpassing the MAE < 1.0 target by a 25% margin. The model outperforms the naive baseline by 47% and is integrated with Bill of Materials structure to generate raw material requirements recommendations. This research demonstrates that XGBoost is an effective and applicable demand forecasting solution in the context of Indonesian F&B MSMEs.

  #v(0.5em)
  *Keywords:* machine learning, demand forecasting, XGBoost, coffee shop, inventory management, food waste
]


// ============================================================
// TABLE OF CONTENTS
// ============================================================
#page(footer: none)[
  #set par(first-line-indent: 0em)
  #align(center)[#text(size: 14pt, weight: "bold")[DAFTAR ISI]]
  #v(1em)
  #outline(title: none, indent: 1.5em, depth: 3)
]


// ============================================================
// MAIN CONTENT
// ============================================================
#set page(numbering: "1")
#counter(page).update(1)

// ---- 1. PENDAHULUAN ----
= 1. Pendahuluan

== 1.1 Latar Belakang

Industri Food and Beverage (F&B) menghadapi paradoks pertumbuhan: ekspansi bisnis yang pesat beriringan dengan inefisiensi lingkungan yang masif. Pada tahun 2022, sektor layanan makanan dilaporkan mengakumulasi 290 juta ton limbah makanan secara global, merepresentasikan hambatan struktural dalam pencapaian Sustainable Development Goals (SDG) 12.3 (United Nations Environment Programme, 2024). Di Indonesia, sektor HOREKA berkontribusi sebesar 20% dari total limbah pangan non-rumah tangga (Bappenas, 2021).

Kompleksitas manajemen inventaris teramplifikasi pada industri kedai kopi, di mana karakteristik bahan baku memiliki tingkat perishability yang tinggi (Chae et al., 2024; Hulaini et al., 2025). Deviasi minor dalam perencanaan stok menghasilkan dampak ganda: kerugian material akibat pembusukan bahan baku (overstock) dan hilangnya kesempatan penjualan saat kehabisan stok (stockout).

=== Permasalahan di UMKM F&B
#lorem(40)

=== Tantangan Peramalan Permintaan
#lorem(30)

== 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah sebagai berikut:

+ Bagaimana membangun model peramalan permintaan harian menggunakan algoritma XGBoost dengan akurasi MAE < 1,0 cangkir per item-hari?
+ Fitur rekayasa apa saja yang paling berkontribusi terhadap akurasi peramalan permintaan pada data transaksi coffee shop?
+ Bagaimana integrasi output peramalan dengan struktur Bill of Materials dan model newsvendor untuk menghasilkan rekomendasi pengadaan bahan baku?

== 1.3 Tujuan Penelitian

Penelitian ini bertujuan untuk:

+ Menghasilkan model prediksi permintaan harian dengan MAE < 1,0 cangkir per item-hari menggunakan algoritma XGBoost.
+ Mengidentifikasi fitur rekayasa yang paling dominan melalui analisis Mutual Information dan studi ablasi.
+ Membangun modul konversi ke kebutuhan bahan baku berbasis Bill of Materials (BOM).
+ Menganalisis efektivitas model dibandingkan metode baseline konvensional.

== 1.4 Manfaat Penelitian

=== Manfaat Akademis
#lorem(30)

=== Manfaat Praktis
#lorem(30)


// ---- 2. TINJAUAN PUSTAKA ----
= 2. Tinjauan Pustaka

== 2.1 Peramalan Permintaan (Demand Forecasting)

Peramalan permintaan merupakan proses estimasi kuantitas produk yang diinginkan pelanggan pada periode waktu mendatang. Berbeda dari peramalan penjualan (sales forecasting) yang hanya mencerminkan kuantitas terealisasi, peramalan permintaan menangkap censored demand yang tidak tertangkap dalam data transaksi historis (Birkmaier et al., 2024).

=== Metode Konvensional
#lorem(30)

=== Pendekatan Machine Learning
#lorem(30)

== 2.2 Extreme Gradient Boosting (XGBoost)

XGBoost merupakan algoritma ensemble berbasis gradient boosting yang dikembangkan oleh Chen dan Guestrin (2016). Arsitektur XGBoost membangun pohon keputusan secara sekuensial, di mana setiap pohon baru mengoreksi error dari ensemble sebelumnya.

=== Prinsip Kerja Gradient Boosting
#lorem(40)

=== Keunggulan XGBoost
- Kemampuan menangani non-linearity tanpa asumsi parametrik
- Robustness terhadap outlier yang umum pada data transaksi F&B
- Interpretabilitas model yang mendukung justifikasi keputusan bisnis
- Efisiensi komputasi dan kemampuan skala

== 2.3 Feature Engineering untuk Data Time Series

Rekayasa fitur merupakan proses transformasi data mentah menjadi fitur informatif yang meningkatkan performa model. Untuk data transaksi harian, fitur dikelompokkan menjadi:

- *Fitur Temporal:* hari dalam minggu, bulan, indikator akhir pekan
- *Fitur Lag:* kuantitas terjual pada lag 1, 7, dan 14 hari
- *Fitur Recency:* Days_Since_Last_Sale — fitur dengan Mutual Information tertinggi (0,656 nats)
- *Fitur Cross-Item:* agregat penjualan kafe pada periode sebelumnya

== 2.4 Model Newsvendor untuk Manajemen Inventaris

Model newsvendor menghitung buffer persediaan optimal dengan mempertimbangkan trade-off antara biaya stockout (understock) dan biaya pembusukan (overstock):

$ Q^* = hat(mu) + z_(C R) dot sigma_epsilon $

dengan CR = c_u / (c_u + c_o), di mana c_u adalah margin lost per unit dan c_o adalah spoilage cost per unit.


// ---- 3. METODE PENELITIAN ----
= 3. Metode Penelitian

== 3.1 Desain Penelitian

Penelitian ini menggunakan pendekatan Iterative and Incremental Development dalam mengembangkan sistem peramalan permintaan berbasis machine learning. Pendekatan ini dipilih karena pengembangan model ML memerlukan eksperimen berulang dalam feature engineering, pemilihan algoritma, dan hyperparameter tuning.

=== Arsitektur Sistem
#lorem(40)

== 3.2 Pengumpulan Data

Data historis transaksi diperoleh dari sistem PoS Husgendam Coffee dengan periode 1 Januari 2022 hingga 25 Mei 2026, terdiri dari 94.611 data item terjual. Data diekspor dalam format CSV dengan struktur transaksi individual meliputi: tanggal, nomor struk, kategori, SKU, nama barang, dan kuantitas.

=== Spesifikasi Data
- Periode: Januari 2022 — Mei 2026
- Total transaksi: 94.611 item terjual
- Jumlah item aktif: 61 item
- Format: CSV

== 3.3 Feature Engineering

Sebanyak 31 fitur rekayasa diekstraksi dari data historis, dikelompokkan ke dalam empat kategori utama.

=== Fitur Temporal
#lorem(30)

=== Fitur Lag
#lorem(30)

=== Fitur Recency
Days_Since_Last_Sale merupakan fitur dengan Mutual Information tertinggi (0,656 nats), tiga kali lipat dari fitur berikutnya. Fitur ini menjawab pertanyaan prediktif fundamental: "apakah item akan terjual hari ini?"

=== Fitur Cross-Item
#lorem(30)

== 3.4 Pemodelan XGBoost

Model global tunggal dipilih berdasarkan hasil Exploratory Data Analysis yang menunjukkan 26% item memiliki data historis terbatas dan zero-inflation sebesar 64%.

=== Konfigurasi Model
- Fungsi objektif: `count:poisson`
- Jumlah item: 61 item (global single model)
- Hyperparameter tuning: Grid Search dengan Time Series Cross-Validation

=== Parameter yang Dioptimasi
- `learning_rate`
- `max_depth`
- `n_estimators`
- `subsample`
- `min_child_weight`
- `colsample_bytree`

== 3.5 Evaluasi dengan Expanding Window Backtest

Evaluasi menggunakan expanding window backtest dengan 8 jendela waktu. Pada setiap jendela, model dilatih pada data historis yang tersedia dan diuji pada periode berikutnya.

=== Metrik Evaluasi
- MAE (target < 1,0 cangkir per item-hari)
- RMSE
- Bias
- R² overall
- R² non-zero
- wMAPE

== 3.6 Integrasi Model Newsvendor

Output peramalan diintegrasikan dengan model newsvendor untuk menghitung buffer persediaan optimal. Service level diferensial diterapkan per kelas ABC:

- *Kelas A:* 95% service level (27 item, 70% volume)
- *Kelas B:* 90% service level
- *Kelas C:* 85% service level


// ---- 4. HASIL DAN PEMBAHASAN ----
= 4. Hasil dan Pembahasan

== 4.1 Performa Model XGBoost

Hasil backtest expanding window 8-jendela disajikan pada Tabel 1. Model mencapai MAE 0,745 cangkir per item-hari, melampaui target 25% di bawah ambang batas 1,0.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt,
    inset: (x: 6pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Metrik*],
      table.cell(fill: luma(230))[*Nilai*],
      table.cell(fill: luma(230))[*Interpretasi*],
    ),
    [MAE],          [0,745],  [~45 cangkir/hari total],
    [RMSE],         [1,37],   [28% di bawah std. aktual],
    [Bias],         [−0,002], [Tidak ada systematic error],
    [R² overall],   [0,48],   [Prediksi nol akurat],
    [R² non-zero],  [0,18],   [Di atas batas teoritis],
    [wMAPE],        [51,1%],  [Konsekuensi skala mikro],
  ),
  caption: [Hasil Backtest Expanding Window 8-Jendela],
)

#h(1.5em) Nilai CV_error = 0,94 mengindikasikan ketidakpastian hampir setara dengan sinyal. Batas teoritis R²_max ≈ 1 − 0,94² ≈ 0,12; R² non-zero aktual 0,18 berada di atas batas ini, mengkonfirmasi bahwa model telah mengekstraksi informasi maksimal yang tersedia dalam data.

== 4.2 Perbandingan dengan Baseline

Model XGBoost mengungguli seluruh baseline naive dengan perbaikan MAE sebesar 47% dibandingkan baseline Day-of-Week (DOW) Median.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: (x: 6pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Strategi*],
      table.cell(fill: luma(230))[*MAE*],
      table.cell(fill: luma(230))[*vs XGBoost*],
    ),
    [Baseline Naive (rata-rata)],  [1,67], [+124%],
    [Baseline DOW Median],         [1,40], [+88%],
    [XGBoost (fitur hibrida)],     [1,29], [+73%],
    [XGBoost (model final)],       [0,745],[referensi],
  ),
  caption: [Perbandingan Strategi Peramalan],
)

== 4.3 Dominasi Fitur Recency

Analisis Mutual Information menunjukkan Days_Since_Last_Sale memiliki nilai 0,656 nats, tiga kali lipat fitur berikutnya. Studi ablasi memperkuat temuan ini: penghapusan fitur recency meningkatkan MAE sebesar 0,517 (71%).

Dominasi recency dapat dijelaskan melalui karakteristik zero-inflation 64% pada data. Fitur ini menjawab pertanyaan mendasar "apakah item akan terjual hari ini?" yang lebih sering dan lebih prediktif. Melalui fungsi objektif `count:poisson`, ketika Days_Since_Last_Sale bernilai tinggi, model mengasosiasikan kondisi tersebut dengan λ ≈ 0 sehingga P(Y=0) mendekati 1.

=== Analisis Ablasi Kelompok Fitur
#lorem(40)

== 4.4 Integrasi Model Newsvendor dan Rekomendasi Pengadaan

Buffer persediaan dihitung dengan service level diferensial per kelas ABC. Total buffer mingguan yang direkomendasikan adalah 730 cangkir, dengan Kelas A mendominasi 73% dari total buffer.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: (x: 6pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Skenario*],
      table.cell(fill: luma(230))[*CR*],
      table.cell(fill: luma(230))[*z*],
      table.cell(fill: luma(230))[*Buffer/Minggu*],
    ),
    [Stockout 5× lebih mahal], [83%], [0,96], [624 cangkir],
    [Stockout 3× lebih mahal], [75%], [0,67], [468 cangkir],
    [Stockout 2× lebih mahal], [67%], [0,44], [312 cangkir],
    [Biaya sama], [50%], [0], [0 cangkir],
  ),
  caption: [Sensitivitas Buffer terhadap Critical Ratio],
)

== 4.5 Analisis Stratifikasi per Kelas ABC

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    stroke: 0.5pt,
    inset: (x: 6pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Kelas*],
      table.cell(fill: luma(230))[*Rata-rata/Hari*],
      table.cell(fill: luma(230))[*MAE*],
      table.cell(fill: luma(230))[*MAPE*],
    ),
    [A], [3,2 cangkir], [1,16], [68%],
    [B], [1,5 cangkir], [0,57], [64%],
    [C], [0,6 cangkir], [0,26], [56%],
  ),
  caption: [Metrik Evaluasi per Kelas ABC],
)

#h(1.5em) MAE meningkat seiring volume (wajar secara matematis), namun rasio MAE/volume menurun dari 0,43 ke 0,36, mengindikasikan peningkatan presisi relatif pada item high-volume. Rasio Error_Std/MAE yang konsisten sekitar 2,8 di seluruh kelas menunjukkan distribusi error yang serupa.


// ---- 5. KESIMPULAN DAN SARAN ----
= 5. Kesimpulan dan Saran

== 5.1 Kesimpulan

Penelitian ini berhasil menerapkan algoritma XGBoost untuk peramalan permintaan harian di Husgendam Coffee dengan hasil yang melampaui target kinerja. Model global tunggal untuk 61 item menghasilkan MAE 0,745 cangkir per item-hari (margin 25% di bawah target), RMSE 28% lebih rendah dari standar deviasi data aktual, dan bias mendekati nol (−0,002) yang mengkonfirmasi tidak adanya kecenderungan sistematis.

Model XGBoost memangkas sekitar 40% dari gap antara strategi pembaruan rekursif dan batas atas teoritis, dengan fitur recency (Days_Since_Last_Sale) sebagai prediktor paling kritis pada data zero-inflation tinggi. Integrasi dengan Bill of Materials dan model newsvendor menghasilkan rekomendasi pengadaan bahan baku yang actionable dengan buffer 730 cangkir per minggu.

== 5.2 Saran

Untuk penelitian lanjutan disarankan:

+ Mengintegrasikan variabel eksternal seperti cuaca dan event lokal untuk meningkatkan R² non-zero.
+ Mengeksplorasi ensemble XGBoost dengan model probabilistik seperti Temporal Fusion Transformer untuk item dengan zero-inflation ekstrem.
+ Memvalidasi pipeline pada UMKM F&B lain untuk mengkonfirmasi generalisabilitas pendekatan.
+ Mengembangkan antarmuka pengguna (dashboard) untuk kemudahan interpretasi hasil peramalan oleh pemilik bisnis.


// ---- DAFTAR PUSTAKA ----
= Daftar Pustaka

#set par(first-line-indent: 0em, hanging-indent: 1.5em, justify: true)

[1] Alt, R. (2021). Digital transformation in the restaurant industry: Current developments and implications. _Journal of Smart Tourism_, 1(1), 69–74. https://doi.org/10.52255/smarttourism.2021.1.1.9

[2] Bappenas. (2021). _Laporan Kajian Food Loss and Waste di Indonesia_.

[3] Birkmaier, A., Imeri, A., & Reiner, G. (2024). Improving supply chain planning for perishable food: Data-driven implications for waste prevention. _Journal of Business Economics_, 94(6), 1–36. https://doi.org/10.1007/s11573-024-01191-x

[4] Chae, B. (Kevin), Sheu, C., & Park, E. O. (2024). The value of data, machine learning, and deep learning in restaurant demand forecasting. _Decision Support Systems_, 184. https://doi.org/10.1016/j.dss.2024.114291

[5] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 785–794. https://doi.org/10.1145/2939672.2939785

[6] González Morales, M., & Cavero Rubio, J. A. (2023). Impact of digitalization of sales on the profitability of the restaurant industry during COVID-19. _Economies_, 11(11). https://doi.org/10.3390/economies11110283

[7] Groene, N., & Zakharov, S. (2024). Introduction of AI-based sales forecasting: How to drive digital transformation in food and beverage outlets. _Discover Artificial Intelligence_, 4(1). https://doi.org/10.1007/s44163-023-00097-x

[8] Mitra, A., et al. (2022). A comparative study of demand forecasting models for a multi-channel retail company. _Proceedings of ICMLCA_, 1–6.

[9] Mustapha, A., & Sithole, M. (2025). Forecasting retail sales using machine learning models. _International Journal of Data Science_, 12(1), 45–58.

[10] Nasseri, M., et al. (2023). Tree-based ensemble methods vs. deep learning for perishable product demand forecasting. _Computers & Industrial Engineering_, 175, 108892.

[11] Nanda Rosya, F. (2024). Optimalisasi stok UMKM menggunakan algoritma machine learning. _Jurnal Teknologi Informasi_, 10(2), 112–120.

[12] Rodrigues, P., et al. (2024). Machine learning-based procurement optimization for food waste reduction. _Sustainable Production and Consumption_, 45, 301–315.

[13] Schmidt, A., et al. (2022). Machine learning based restaurant sales forecasting. _Machine Learning with Applications_, 9, 100371.

[14] Turker, H. (2025). Integrating ML models for campus canteen food waste reduction. _Journal of Cleaner Production_, 420, 138450.

[15] United Nations Environment Programme. (2024). _Food Waste Index Report 2024_. UNEP.


// ---- LAMPIRAN ----
= Lampiran

== Lampiran A: Daftar Fitur Rekayasa

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: 0.5pt,
    inset: (x: 6pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Kategori*],
      table.cell(fill: luma(230))[*Nama Fitur*],
      table.cell(fill: luma(230))[*Deskripsi*],
    ),
    [Temporal],  [day_of_week],        [Hari dalam minggu (0–6)],
    [Temporal],  [month],              [Bulan (1–12)],
    [Temporal],  [is_weekend],         [Indikator akhir pekan (0/1)],
    [Lag],       [qty_lag_1],          [Kuantitas lag 1 hari],
    [Lag],       [qty_lag_7],          [Kuantitas lag 7 hari],
    [Lag],       [qty_lag_14],         [Kuantitas lag 14 hari],
    [Recency],   [Days_Since_Last_Sale], [Hari sejak penjualan terakhir],
    [Cross-item],[cafe_qty_prev_day],  [Total penjualan kafe hari sebelumnya],
    [...],       [...],                [...],
  ),
  caption: [Daftar Lengkap 31 Fitur Rekayasa],
)

== Lampiran B: Hyperparameter Optimal

#figure(
  table(
    columns: (auto, auto),
    align: (left, center),
    stroke: 0.5pt,
    inset: (x: 6pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Parameter*],
      table.cell(fill: luma(230))[*Nilai Optimal*],
    ),
    [learning_rate],   [0.05],
    [max_depth],       [6],
    [n_estimators],    [500],
    [subsample],       [0.8],
    [min_child_weight],[3],
    [colsample_bytree],[0.8],
  ),
  caption: [Hyperparameter Optimal dari Grid Search],
)
