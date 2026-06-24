// ============================================================
// TEKNOSI (Jurnal Nasional Teknologi dan Sistem Informasi)
// Layout v2 — following Template_2017v3.docx
// Compile: typst compile jip_journal_v2.typ
// ============================================================

// ---------- Page ----------
#set page(
  paper: "a4",
  margin: (top: 20mm, left: 20mm, bottom: 20mm, right: 15mm),
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 6.5pt, font: "Times New Roman", fill: luma(128))
      let authors = "Endah Septa Sintiya"
      let journal  = "JURNAL NASIONAL TEKNOLOGI DAN SISTEM INFORMASI - VOL. XX NO. XX (2020) XXX-XXX"
      align(center)[#authors / #journal]
    }
  },
  footer: context {
    set text(size: 8pt, font: "Times New Roman", fill: luma(128))
    let p = counter(page).get().first()
    if p > 1 {
      grid(columns: (1fr, 1fr),
        align(left)[#p #h(0.5em) Endah Septa Sintiya],
        align(right)[https://doi.org/xx.xxxxx/xxxxx.xxxxx],
      )
    } else {
      grid(columns: (1fr, 1fr),
        align(left)[https://doi.org/xx.xxxxx/xxxxx.xxxxx],
        align(right)[Attribution-ShareAlike 4.0 International #h(0.3em) Some rights reserved],
      )
    }
  },
)

// ---------- Base text ----------
#set text(font: "Times New Roman", size: 9pt, lang: "id")
#set par(justify: true, leading: 0.55em, spacing: 0.55em)

// ---------- Headings ----------
#show heading: set par(first-line-indent: 0pt)
#show heading.where(level: 1): it => {
  v(0.6em)
  text(size: 10.5pt, weight: "bold")[#upper(it.body)]
  v(0.3em)
}
#show heading.where(level: 2): it => {
  v(0.4em)
  text(size: 10pt, weight: "bold", style: "italic")[#it.body]
  v(0.2em)
}
#show heading.where(level: 3): it => {
  v(0.3em)
  text(size: 10pt, style: "italic")[#it.body]
  v(0.15em)
}

// ---------- Figures / Tables ----------
#show figure: it => {
  set par(first-line-indent: 0pt)
  v(0.5em)
  align(center, text(size: 9pt)[#it.supplement #it.numbering. #it.caption.body])
  v(0.3em)
  it.body
  v(0.5em)
}

// ============================================================
// PAGE 1 HEADER BLOCK (journal masthead)
// ============================================================
#set par(first-line-indent: 0pt)

// Top running head (page 1 only — mimics the template's page header)
#align(center)[
  #text(size: 8pt, font: "Times New Roman",fill: luma(128))[
    J#smallcaps[urnal] N#smallcaps[asional] T#smallcaps[eknologi dan] S#smallcaps[istem] I#smallcaps[nformasi] - Vol. xx No. xx (2020) xxx-xxx
  ]
]
#line(length: 100%, stroke: 0.4pt)
#v(0.3em)

// Masthead banner
#box(width: 100%)[
  #grid(
    columns: (18mm, 1fr, 18mm),
    gutter: 4mm,
    // Left logo
    align(center + horizon)[
      #image("Logo-Polinema.webp", width: 16mm, height: 16mm)
    ],
    // Centre text
    align(center + horizon)[
      #text(size: 7.5pt)[Terbit #emph[online] pada laman : #text(fill: blue)[http://xxxx.xxx.polinema.ac.id/]]
      #v(1mm)
      #text(size: 14pt, weight: "bold")[Jurnal Nasional Teknologi dan Sistem Informasi]
      #v(1mm)
      #text(size: 8.5pt)[| #h(4pt) ISSN (Print) 2460-3465 #h(4pt) | #h(4pt) ISSN (Online) 2476-8812 #h(4pt) |]
    ],
    // Right logo placeholder
    align(center + horizon)[
      // #rect(width: 16mm, height: 16mm, stroke: 0.5pt, fill: luma(230))[
      //   #align(center + horizon)[#text(size: 6pt)[Logo]]
      // ]
    ],
  )
]
#v(0.5em)
#line(length: 100%, stroke: 1pt)
#v(0.5em)

// Article category tag
#text(size: 8pt, fill: luma(100))[Artikel Penelitian]
#v(0.3em)

// ============================================================
// TITLE
// ============================================================
#text(size: 20pt, weight: "regular")[
  Penerapan Algoritma Machine Learning dalam Peramalan Permintaan Produk pada Coffee Shop Berbasis Data Historis Transaksi
]
#v(0.5em)

// ============================================================
// AUTHORS
// ============================================================
#text(size: 11pt, style: "italic")[
  Endah Septa Sintiya #super[a], Yoppy Yunhasnawa #super[b], Doni Wahyu Kurniawan #super[c,\*]
]
#v(0.4em)

#text(size: 9pt, style: "italic")[
  #super[a] Dosen Pembimbing satu \
  #super[b] Dosen Pembimbing dua \
  #super[c] Mahasiswa Bimbingan
]
#v(0.8em)
#line(length: 100%, stroke: 0.4pt)
#v(0.6em)

// ============================================================
// INFO + ABSTRACT SIDEBAR TABLE
// ============================================================
#grid(
  columns: (45mm, 5mm, 1fr),
  // Left: Informasi Artikel
  [
    #set text(size: 8pt)
    #rect(width: 100%, stroke: none)[
      #text(weight: "bold", size: 10pt)[#smallcaps[Informasi Artikel]]
      #line(length: 100%, stroke: 0.4pt)
      #v(0.3em)
      #text(style: "italic")[_Sejarah Artikel:_] \
      Diterima Redaksi: xx xxxxx xxxx \
      Revisi Akhir: xx xxxxx xxxx \
      Diterbitkan _Online_: xx xxxxx xxxx
      #v(0.5em)
      #text(weight: "bold", size: 10pt)[#smallcaps[Kata Kunci]]
      #line(length: 100%, stroke: 0.4pt)
      #v(0.2em)
      machine learning \
      peramalan permintaan \
      XGBoost \
      manajemen inventaris \
      coffee shop
      #v(0.5em)
      #text(weight: "bold", size: 10pt)[#smallcaps[Korespondensi]]
      #line(length: 100%, stroke: 0.4pt)
      #v(0.2em)
      E-mail: doniwyk\@gmail.com \*
    ]
  ],
  [],
  // Right: Abstract
  [
    #set text(size: 7.5pt)
    #align(center)[#text(weight: "bold", size: 10pt, tracking: 3pt)[A B S T R A C T]]
    #v(0.4em)
    #set par(justify: true, first-line-indent: 0pt)
    Industri Food and Beverage (F&B) menghadapi tantangan inefisiensi inventaris yang signifikan, terutama pada segmen kedai kopi dengan bahan baku perishable. Husgendam Coffee, sebuah kedai kopi skala menengah di Kota Batu, masih mengandalkan sistem pengadaan manual meskipun memiliki data historis transaksi Point-of-Sale (PoS) yang kaya (94.611 item terjual, Januari 2022–Mei 2026). Penelitian ini mengembangkan model peramalan permintaan harian menggunakan algoritma Extreme Gradient Boosting (XGBoost) dengan arsitektur model global tunggal yang melatih 61 item secara bersamaan menggunakan 31 fitur rekayasa temporal dan recency. Evaluasi menggunakan expanding window backtest 8-jendela menghasilkan MAE 0,745 cangkir per item-hari, melampaui target MAE < 1,0 dengan margin 25% dan mengungguli baseline naive sebesar 47%. Fitur recency (Days_Since_Last_Sale) terbukti sebagai prediktor terkritits dengan Mutual Information 0,656 nats. Integrasi dengan struktur Bill of Materials dan model newsvendor menghasilkan rekomendasi pengadaan bahan baku dengan total buffer 730 cangkir per minggu. Penelitian ini menunjukkan bahwa XGBoost merupakan solusi peramalan yang efektif dan applicable pada konteks UMKM F&B di Indonesia.
    #v(0.5em)
    #line(length: 100%, stroke: 0.4pt)
  ],
)
#v(0.8em)
#line(length: 100%, stroke: 0.4pt)
#v(0.8em)

// ============================================================
// BODY — Two Columns
// ============================================================
#set par(justify: true, leading: 0.55em, spacing: 0.55em, first-line-indent: 5mm)

#columns(2, gutter: 6mm)[

= 1. Pendahuluan

Industri Food and Beverage (F&B) menghadapi paradoks pertumbuhan: ekspansi bisnis yang pesat beriringan dengan inefisiensi lingkungan yang masif. Pada tahun 2022, sektor layanan makanan mengakumulasi 290 juta ton limbah makanan secara global, merepresentasikan hambatan struktural dalam pencapaian Sustainable Development Goals (SDG) 12.3 [1]. Di Indonesia, sektor HOREKA berkontribusi sebesar 20% dari total limbah pangan non-rumah tangga [2].

Kompleksitas manajemen inventaris teramplifikasi pada industri kedai kopi, di mana karakteristik bahan baku memiliki tingkat perishability yang tinggi [3]. Deviasi minor dalam perencanaan stok menghasilkan dampak ganda: kerugian material akibat pembusukan bahan baku (overstock) dan hilangnya kesempatan penjualan saat kehabisan stok (stockout) pada periode permintaan tinggi.

Machine Learning (ML) menawarkan pendekatan analitik prediktif yang lebih robust dibandingkan teknik statistik klasik seperti ARIMA. Algoritma seperti Extreme Gradient Boosting (XGBoost) mampu menangkap pola non-linear kompleks tanpa asumsi parametrik yang ketat [4]. Studi empiris menunjukkan ML umumnya mengungguli metode baseline dalam akurasi prediksi permintaan restoran [3].

Husgendam Coffee, sebagai representasi kedai kopi skala menengah di Kota Batu, masih mengandalkan sistem pengadaan manual meskipun memiliki data historis transaksi yang tersimpan sistematis pada sistem Point-of-Sale (PoS). Kesenjangan antara ketersediaan data dan praktik manajerial konvensional menciptakan peluang signifikan untuk intervensi teknologi berbasis ML.

Penelitian ini secara spesifik berfokus pada peramalan permintaan (demand forecasting), yang berbeda dari peramalan penjualan (sales forecasting). Permintaan merepresentasikan kuantitas yang diinginkan pelanggan terlepas dari ketersediaan stok, sedangkan penjualan hanya mencerminkan kuantitas yang terealisasi dalam transaksi [5]. Pendekatan ini memungkinkan penangkapan censored demand yang tidak tertangkap dalam data penjualan historis.

Tujuan penelitian ini adalah: (1) menghasilkan model prediksi permintaan harian dengan MAE < 1,0 cangkir per item-hari, (2) membangun modul konversi ke kebutuhan bahan baku berbasis Bill of Materials, dan (3) menganalisis efektivitas model dibandingkan metode konvensional.

= 2. Metode

Penelitian ini menggunakan pendekatan Iterative and Incremental Development. Pengembangan dilakukan dalam tiga increment utama: pengembangan model ML inti, integrasi sistem PoS dengan pipeline rekomendasi, dan fitur analitik lanjutan.

== 2.1 Pengumpulan dan Persiapan Data

Data historis transaksi diperoleh dari sistem PoS Husgendam Coffee periode 1 Januari 2022 hingga 25 Mei 2026, mencakup 94.611 data item terjual. Data diekspor dalam format CSV dengan atribut: tanggal, nomor struk, kategori, SKU, nama barang, dan kuantitas.

Setelah proses transformasi, data diagregasi ke level harian per item (_item-hari_), menghasilkan dataset final untuk 61 item aktif. Karakteristik kritis dataset mencakup zero-inflation 64%, tren pertumbuhan 4× selama periode observasi, dan 26% item dengan data historis terbatas.

== 2.2 Feature Engineering

Sebanyak 31 fitur rekayasa diekstraksi dari data historis dalam empat kelompok: fitur temporal (hari dalam minggu, bulan, indikator akhir pekan), fitur lag (kuantitas terjual pada lag 1, 7, dan 14 hari), fitur recency (Days_Since_Last_Sale), dan fitur cross-item (agregat penjualan kafe pada periode sebelumnya).

Seleksi fitur dilakukan melalui analisis Mutual Information. Hasil analisis menunjukkan Days_Since_Last_Sale sebagai fitur dominan dengan nilai 0,656 nats, tiga kali lipat fitur berikutnya.

== 2.3 Pemodelan XGBoost

Model global tunggal dipilih untuk melatih 61 item secara bersamaan, berdasarkan pertimbangan efisiensi dan kondisi data terbatas pada sebagian item. Fungsi objektif `count:poisson` digunakan untuk menangani distribusi permintaan diskrit dengan dominasi nilai nol.

Hyperparameter dioptimasi melalui Grid Search dengan Time Series Cross-Validation. Parameter yang dituning meliputi `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `min_child_weight`, dan `colsample_bytree`.

== 2.4 Evaluasi dengan Expanding Window Backtest

Evaluasi menggunakan expanding window backtest dengan 8 jendela waktu. Pada setiap jendela, model dilatih pada data historis yang tersedia dan diuji pada periode berikutnya. Metrik evaluasi meliputi MAE (target < 1,0), RMSE, Bias, R² overall, R² non-zero, dan wMAPE.

== 2.5 Integrasi Model Newsvendor

Output peramalan diintegrasikan dengan model newsvendor untuk menghitung buffer persediaan optimal. Kuantitas pengadaan dihitung sebagai:

#set par(first-line-indent: 0pt)
$ Q^* = hat(mu) + z_(C R) dot sigma_epsilon $
#set par(first-line-indent: 5mm)

dengan CR = c_u / (c_u + c_o), di mana c_u adalah biaya stockout dan c_o adalah biaya overstock per unit. Service level diferensial diterapkan per kelas ABC: 95% (Kelas A), 90% (Kelas B), dan 85% (Kelas C).

= 3. Hasil

== 3.1 Performa Model XGBoost

Hasil backtest expanding window 8-jendela disajikan pada Tabel 1. Model mencapai MAE 0,745 cangkir per item-hari, melampaui target 25% di bawah ambang batas 1,0. RMSE sebesar 1,37 berada 28% di bawah standar deviasi data aktual (1,91), mengkonfirmasi nilai prediktif yang nyata.

#set par(first-line-indent: 0pt)
#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.4pt,
    inset: (x: 5pt, y: 4pt),
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
#set par(first-line-indent: 5mm)

Nilai CV_error = 0,94 mengindikasikan ketidakpastian hampir setara dengan sinyal. Batas teoritis R²_max ≈ 1 − 0,94² ≈ 0,12; R² non-zero aktual 0,18 berada di atas batas ini, mengkonfirmasi model telah mengekstraksi informasi maksimal dari data.

== 3.2 Perbandingan dengan Baseline

Model XGBoost mengungguli seluruh baseline naive, dengan perbaikan MAE 47% terhadap baseline Day-of-Week (DOW) Median. Perbandingan strategi peramalan disajikan pada Tabel 2.

#set par(first-line-indent: 0pt)
#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.4pt,
    inset: (x: 5pt, y: 4pt),
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
#set par(first-line-indent: 5mm)

= 4. Pembahasan

== 4.1 Dominasi Fitur Recency

Analisis Mutual Information mengungkapkan Days_Since_Last_Sale memiliki nilai 0,656 nats, tiga kali lipat fitur berikutnya. Studi ablasi memperkuat temuan ini: penghapusan kelompok fitur recency meningkatkan MAE sebesar 0,410 (55%).

Dominasi recency dapat dijelaskan melalui zero-inflation 64% pada data. Terdapat dua pertanyaan prediktif dengan proporsi berbeda: "apakah item akan terjual hari ini?" (64% kasus) yang dijawab oleh fitur recency, dan "jika terjual, berapa banyak?" (36% kasus) yang dijawab oleh fitur temporal dan lag. Secara matematis, melalui fungsi objektif `count:poisson`, ketika Days_Since_Last_Sale bernilai tinggi, model mengasosiasikan kondisi tersebut dengan λ ≈ 0 sehingga P(Y=0) = e^(−λ) mendekati 1.

== 4.2 Stratifikasi per Kelas ABC

Klasifikasi ABC mengidentifikasi 27 item Kelas A (70% volume total). Metrik evaluasi per kelas disajikan pada Tabel 3.

#set par(first-line-indent: 0pt)
#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    stroke: 0.4pt,
    inset: (x: 5pt, y: 4pt),
    table.header(
      table.cell(fill: luma(230))[*Kelas*],
      table.cell(fill: luma(230))[*Rata-rata/Hari*],
      table.cell(fill: luma(230))[*MAE*],
      table.cell(fill: luma(230))[*MAPE*],
    ),
    [A], [3,2], [1,16], [68%],
    [B], [1,5], [0,57], [64%],
    [C], [0,6], [0,26], [56%],
  ),
  caption: [Metrik Evaluasi per Kelas ABC],
)
#set par(first-line-indent: 5mm)

MAE meningkat seiring volume, namun rasio MAE/volume menurun dari 0,43 (Kelas C) menjadi 0,36 (Kelas A), mengindikasikan peningkatan presisi relatif pada item high-volume. Rasio Error_Std/MAE yang konsisten ~2,8 di seluruh kelas menunjukkan distribusi error yang serupa.

== 4.3 Integrasi Newsvendor dan Rekomendasi Pengadaan

Total buffer mingguan yang direkomendasikan adalah 730 cangkir, dengan Kelas A mendominasi 73% (534 cangkir) sesuai service level 95%. Pada kondisi operasional tipikal (biaya stockout 3× biaya overstock, CR = 75%), buffer optimal adalah 468 cangkir (+36% di atas prediksi titik).

= 5. Kesimpulan

Penelitian ini berhasil menerapkan algoritma XGBoost untuk peramalan permintaan harian di Husgendam Coffee dengan hasil yang melampaui seluruh target kinerja. Model global tunggal untuk 61 item menghasilkan MAE 0,745 cangkir per item-hari (25% di bawah target), RMSE 28% lebih rendah dari standar deviasi aktual, dan bias mendekati nol (−0,002).

Fitur recency (Days_Since_Last_Sale) terbukti sebagai prediktor terkritis dengan Mutual Information 0,656 nats. Integrasi dengan Bill of Materials dan model newsvendor menghasilkan rekomendasi pengadaan bahan baku yang actionable dengan total buffer 730 cangkir per minggu.

Untuk penelitian lanjutan disarankan: (1) mengintegrasikan variabel eksternal seperti cuaca dan kalender event lokal untuk meningkatkan R² non-zero; (2) mengeksplorasi ensemble XGBoost dengan model probabilistik seperti Temporal Fusion Transformer untuk item dengan zero-inflation ekstrem; dan (3) memvalidasi pipeline pada UMKM F&B lain untuk mengkonfirmasi generalisabilitas pendekatan.

= Ucapan Terima Kasih

Penelitian ini didukung oleh Husgendam Coffee Kota Batu atas akses data transaksi dan kerja sama selama proses penelitian. Penulis juga mengucapkan terima kasih kepada Politeknik Negeri Malang atas dukungan fasilitas penelitian.

= Daftar Pustaka

#set par(first-line-indent: 0pt, hanging-indent: 4mm)
#set text(size: 9pt)

[1] United Nations Environment Programme, _Food Waste Index Report 2024_. Nairobi: UNEP, 2024.

[2] Bappenas, "Kajian Food Loss and Waste di Indonesia," Bappenas, Jakarta, Laporan Kajian, 2021.

[3] B. K. Chae, C. Sheu, and E. O. Park, "The value of data, machine learning, and deep learning in restaurant demand forecasting," _Decision Support Systems_, vol. 184, 2024, doi: 10.1016/j.dss.2024.114291.

[4] A. Schmidt, M. W. U. Kabir, and M. E. Haque, "Machine learning based restaurant sales forecasting," _Machine Learning with Applications_, vol. 9, p. 100371, 2022.

[5] A. Birkmaier, A. Imeri, and G. Reiner, "Improving supply chain planning for perishable food: Data-driven implications for waste prevention," _Journal of Business Economics_, vol. 94, no. 6, 2024, doi: 10.1007/s11573-024-01191-x.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in _Proc. ACM SIGKDD_, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[7] N. Groene and S. Zakharov, "Introduction of AI-based sales forecasting: How to drive digital transformation in food and beverage outlets," _Discover Artificial Intelligence_, vol. 4, no. 1, 2024, doi: 10.1007/s44163-023-00097-x.

[8] M. González Morales and J. A. Cavero Rubio, "Impact of digitalization of sales on the profitability of the restaurant industry during COVID-19," _Economies_, vol. 11, no. 11, 2023, doi: 10.3390/economies11110283.

[9] M. Nasseri, J. Behnamian, and A. Aghsami, "Tree-based ensemble methods vs. deep learning for perishable product demand forecasting," _Computers & Industrial Engineering_, vol. 175, p. 108892, 2023.

[10] F. Nanda Rosya, "Optimalisasi stok UMKM menggunakan algoritma machine learning," _Jurnal Teknologi Informasi_, vol. 10, no. 2, pp. 112–120, 2024.

[11] A. Mitra, S. Ghosh, and A. Das, "A comparative study of demand forecasting models for a multi-channel retail company," in _Proc. ICMLCA_, 2022, pp. 1–6.

[12] H. Turker, "Integrating ML models for campus canteen food waste reduction," _Journal of Cleaner Production_, vol. 420, p. 138450, 2025.

[13] P. Rodrigues, A. Matos, and R. Silva, "Machine learning-based procurement optimization for food waste reduction," _Sustainable Production and Consumption_, vol. 45, pp. 301–315, 2024.

[14] A. Mustapha and M. Sithole, "Forecasting retail sales using machine learning models," _International Journal of Data Science_, vol. 12, no. 1, pp. 45–58, 2025.

[15] R. Alt, "Digital transformation in the restaurant industry: Current developments and implications," _Journal of Smart Tourism_, vol. 1, no. 1, pp. 69–74, 2021, doi: 10.52255/smarttourism.2021.1.1.9.

= Biodata Penulis

#set par(first-line-indent: 0pt)
*Doni Wahyu Kurniawan* adalah mahasiswa Program Studi Teknik Informatika, Jurusan Teknologi Informasi, Politeknik Negeri Malang. Fokus penelitiannya meliputi machine learning terapan, analitik data bisnis, dan pengembangan sistem informasi untuk UMKM. Penelitian ini merupakan bagian dari tugas akhir yang berfokus pada implementasi ML untuk optimasi rantai pasok pada industri F&B skala menengah di Indonesia.

] // end columns
