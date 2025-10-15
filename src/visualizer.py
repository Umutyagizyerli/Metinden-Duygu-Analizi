#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Görselleştirme modülü.
Model sonuçlarını ve veri setine ilişkin çeşitli grafikleri görselleştirir.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from typing import Dict, List, Any, Tuple, Union, Optional
import os
from collections import Counter


class Visualizer:
    """Model sonuçları ve veri seti görselleştirmeleri için sınıf"""

    def __init__(self, output_dir: str = '../results/visualizations'):
        """
        Visualizer sınıfını başlatır.

        Parameters:
        -----------
        output_dir : str, default='../results/visualizations'
            Görselleştirmelerin kaydedileceği dizin
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Grafik stili
        plt.style.use('seaborn-v0_8-darkgrid')

        # Matplotlib ayarları
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_label_distribution(self, y_train, y_test=None, labels=None, save=True):
        """
        Etiket dağılımını görselleştirir.

        Parameters:
        -----------
        y_train : array-like
            Eğitim etiketleri
        y_test : array-like, optional
            Test etiketleri
        labels : list, optional
            Etiket isimleri
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 6))

        # Eğitim etiketi dağılımı
        train_counter = Counter(y_train)
        train_labels = sorted(train_counter.keys())
        train_counts = [train_counter[label] for label in train_labels]

        if labels is None:
            labels = [f"Sınıf {label}" for label in train_labels]

        x = np.arange(len(train_labels))
        width = 0.35

        plt.bar(x - width / 2, train_counts, width, label='Eğitim Verisi')

        # Test etiketi dağılımı (varsa)
        if y_test is not None:
            test_counter = Counter(y_test)
            test_counts = [test_counter.get(label, 0) for label in train_labels]
            plt.bar(x + width / 2, test_counts, width, label='Test Verisi')

        plt.xlabel('Sınıf')
        plt.ylabel('Örnek Sayısı')
        plt.title('Veri Seti Etiket Dağılımı')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'label_distribution.png'), dpi=300)
            plt.close()

    def plot_document_length_distribution(self, texts, label=None, bins=50, save=True):
        """
        Doküman uzunluğu dağılımını görselleştirir.

        Parameters:
        -----------
        texts : array-like of str
            Metin dokümanları
        label : str, optional
            Veri seti etiketi (örn. 'Eğitim Verisi', 'Test Verisi')
        bins : int, default=50
            Histogram bin sayısı
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 6))

        # Doküman uzunluklarını hesapla
        doc_lengths = [len(text.split()) for text in texts]

        # Histogram çiz
        sns.histplot(doc_lengths, bins=bins, kde=True, label=label)

        plt.xlabel('Kelime Sayısı')
        plt.ylabel('Doküman Sayısı')
        plt.title('Doküman Uzunluğu Dağılımı')

        if label:
            plt.legend()

        # İstatistikleri ekle
        mean_length = np.mean(doc_lengths)
        median_length = np.median(doc_lengths)
        max_length = np.max(doc_lengths)
        min_length = np.min(doc_lengths)

        stats_text = f"Ortalama: {mean_length:.1f}\nMedyan: {median_length:.1f}\nMax: {max_length}\nMin: {min_length}"
        plt.annotate(stats_text, xy=(0.75, 0.75), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'document_length_distribution.png'), dpi=300)
            plt.close()

    def plot_word_frequency(self, processed_texts, top_n=20, stop_words=None, save=True):
        """
        En sık kullanılan kelimelerin frekansını görselleştirir.

        Parameters:
        -----------
        processed_texts : array-like of str
            İşlenmiş metin dokümanları
        top_n : int, default=20
            Gösterilecek en sık kelime sayısı
        stop_words : set, optional
            Hariç tutulacak stop word'ler
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 8))

        # Tüm kelimeleri bir listede topla
        all_words = []
        for text in processed_texts:
            words = text.split()
            all_words.extend(words)

        # Stop word'leri filtrele (varsa)
        if stop_words:
            all_words = [word for word in all_words if word not in stop_words]

        # Kelime frekanslarını hesapla
        word_freq = Counter(all_words)

        # En sık kelimeler
        top_words = word_freq.most_common(top_n)
        words = [word for word, freq in top_words]
        freqs = [freq for word, freq in top_words]

        # Çubuk grafik çiz
        plt.barh(words[::-1], freqs[::-1])
        plt.xlabel('Frekans')
        plt.ylabel('Kelime')
        plt.title(f'En Sık Kullanılan {top_n} Kelime')

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'word_frequency.png'), dpi=300)
            plt.close()

        return word_freq

    def plot_word_frequency_by_class(self, texts, labels, top_n=10, class_names=None, save=True):
        """
        Sınıf bazında en sık kullanılan kelimeleri görselleştirir.

        Parameters:
        -----------
        texts : array-like of str
            Metin dokümanları
        labels : array-like
            Metin sınıfları
        top_n : int, default=10
            Her sınıf için gösterilecek en sık kelime sayısı
        class_names : list, optional
            Sınıf isimleri
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        Dict[int, Counter]
            Sınıf bazında kelime frekansları
        """
        # Benzersiz sınıfları bul
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)

        if class_names is None:
            class_names = [f"Sınıf {label}" for label in unique_labels]

        # Sınıf bazında kelime frekansları
        class_word_freq = {}

        for label in unique_labels:
            # Bu sınıftaki metinleri filtrele
            class_texts = [text for text, lbl in zip(texts, labels) if lbl == label]

            # Tüm kelimeleri bir listede topla
            all_words = []
            for text in class_texts:
                words = text.split()
                all_words.extend(words)

            # Kelime frekanslarını hesapla
            word_freq = Counter(all_words)
            class_word_freq[label] = word_freq

        # Her sınıf için ayrı grafik oluştur
        n_cols = min(2, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (label, ax) in enumerate(zip(unique_labels, axes)):
            word_freq = class_word_freq[label]

            # En sık kelimeler
            top_words = word_freq.most_common(top_n)
            words = [word for word, freq in top_words]
            freqs = [freq for word, freq in top_words]

            # Çubuk grafik çiz
            ax.barh(words[::-1], freqs[::-1])
            ax.set_xlabel('Frekans')
            ax.set_ylabel('Kelime')
            ax.set_title(f'{class_names[i]} - En Sık {top_n} Kelime')

        # Kullanılmayan grafikleri gizle
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'word_frequency_by_class.png'), dpi=300)
            plt.close()

        return class_word_freq

    def plot_feature_distribution(self, X, feature_names=None, top_n=10, save=True):
        """
        Özniteliklerin dağılımını görselleştirir.

        Parameters:
        -----------
        X : array-like or sparse matrix
            Öznitelik matrisi
        feature_names : list, optional
            Öznitelik isimleri
        top_n : int, default=10
            Gösterilecek en yüksek ortalama değere sahip öznitelik sayısı
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 8))

        # Sparse matris ise yoğun matrise dönüştür
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Ortalama değerleri hesapla
        feature_means = np.mean(X_dense, axis=0)

        # En yüksek ortalama değere sahip öznitelikleri bul
        top_indices = np.argsort(feature_means)[-top_n:]

        if feature_names is None:
            feature_names = [f"Öznitelik {i}" for i in range(X_dense.shape[1])]

        top_features = [feature_names[i] for i in top_indices]
        top_means = [feature_means[i] for i in top_indices]

        # Çubuk grafik çiz
        plt.barh(top_features[::-1], top_means[::-1])
        plt.xlabel('Ortalama Değer')
        plt.ylabel('Öznitelik')
        plt.title(f'En Yüksek Ortalama Değere Sahip {top_n} Öznitelik')

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_distribution.png'), dpi=300)
            plt.close()

    def plot_dimensionality_reduction(self, X, y, method='tsne', perplexity=30, n_components=2,
                                      class_names=None, random_state=42, save=True):
        """
        Boyut indirgeme ile öznitelikleri 2D veya 3D uzayda görselleştirir.

        Parameters:
        -----------
        X : array-like or sparse matrix
            Öznitelik matrisi
        y : array-like
            Etiketler
        method : str, default='tsne'
            Boyut indirgeme yöntemi ('tsne', 'pca', 'svd')
        perplexity : int, default=30
            t-SNE parametresi
        n_components : int, default=2
            İndirgenecek boyut sayısı (2 veya 3)
        class_names : list, optional
            Sınıf isimleri
        random_state : int, default=42
            Rastgele durum
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        np.ndarray
            İndirgenmiş veri
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components 2 veya 3 olmalıdır.")

        # Sparse matris ise yoğun matrise dönüştür
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Boyut indirgeme işlemi
        if method.lower() == 'tsne':
            print("t-SNE uygulanıyor...")
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            X_reduced = reducer.fit_transform(X_dense)
        elif method.lower() == 'pca':
            print("PCA uygulanıyor...")
            reducer = PCA(n_components=n_components, random_state=random_state)
            X_reduced = reducer.fit_transform(X_dense)
        elif method.lower() == 'svd':
            print("TruncatedSVD uygulanıyor...")
            reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
            X_reduced = reducer.fit_transform(X)
        else:
            raise ValueError("Geçersiz method. 'tsne', 'pca' veya 'svd' olmalıdır.")

        # Benzersiz sınıflar
        unique_labels = sorted(set(y))

        if class_names is None:
            class_names = [f"Sınıf {label}" for label in unique_labels]

        # 2D görselleştirme
        if n_components == 2:
            plt.figure(figsize=(12, 10))

            for i, label in enumerate(unique_labels):
                plt.scatter(X_reduced[y == label, 0], X_reduced[y == label, 1],
                            label=class_names[i], alpha=0.7, edgecolors='w')

            plt.xlabel('Bileşen 1')
            plt.ylabel('Bileşen 2')
            plt.title(f'{method.upper()} ile Boyut İndirgeme (2D)')
            plt.legend()
            plt.grid(True)

            if save:
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{method.lower()}_2d.png'), dpi=300)
                plt.close()

        # 3D görselleştirme
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            for i, label in enumerate(unique_labels):
                ax.scatter(X_reduced[y == label, 0], X_reduced[y == label, 1], X_reduced[y == label, 2],
                           label=class_names[i], alpha=0.7)

            ax.set_xlabel('Bileşen 1')
            ax.set_ylabel('Bileşen 2')
            ax.set_zlabel('Bileşen 3')
            ax.set_title(f'{method.upper()} ile Boyut İndirgeme (3D)')
            ax.legend()

            if save:
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{method.lower()}_3d.png'), dpi=300)
                plt.close()

        return X_reduced

    def plot_correlation_matrix(self, X, feature_names=None, top_n=20, save=True):
        """
        Öznitelikler arasındaki korelasyon matrisini görselleştirir.

        Parameters:
        -----------
        X : array-like or sparse matrix
            Öznitelik matrisi
        feature_names : list, optional
            Öznitelik isimleri
        top_n : int, default=20
            Gösterilecek en önemli öznitelik sayısı
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        pd.DataFrame
            Korelasyon matrisi
        """
        # Sparse matris ise yoğun matrise dönüştür
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Veri sayısı çok fazlaysa, rastgele seç
        if X_dense.shape[1] > 1000:
            print("Öznitelik sayısı çok fazla, rastgele örnekleme yapılıyor...")
            indices = np.random.choice(X_dense.shape[1], size=1000, replace=False)
            X_dense = X_dense[:, indices]
            if feature_names is not None:
                feature_names = [feature_names[i] for i in indices]

        # Öznitelik isimleri
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(X_dense.shape[1])]

        # En değişken öznitelikleri bul
        std_devs = np.std(X_dense, axis=0)
        top_indices = np.argsort(std_devs)[-top_n:]

        # İlgili öznitelikleri seç
        X_top = X_dense[:, top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]

        # Korelasyon matrisi
        df = pd.DataFrame(X_top, columns=top_feature_names)
        corr_matrix = df.corr()

        # Görselleştir
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True,
                    fmt='.2f', linewidths=0.5)
        plt.title(f'En Değişken {top_n} Öznitelik Arasındaki Korelasyon Matrisi')

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=300)
            plt.close()

        return corr_matrix

    def plot_learning_curves(self, train_sizes, train_scores, test_scores, title="Öğrenme Eğrisi", save=True):
        """
        Öğrenme eğrilerini görselleştirir.

        Parameters:
        -----------
        train_sizes : array-like
            Eğitim örnek boyutları
        train_scores : array-like
            Eğitim skorları
        test_scores : array-like
            Test/validasyon skorları
        title : str, default="Öğrenme Eğrisi"
            Grafik başlığı
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        plt.figure(figsize=(10, 6))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim skoru")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validasyon skoru")

        plt.title(title)
        plt.xlabel("Eğitim örnekleri sayısı")
        plt.ylabel("Skor")
        plt.grid(True)
        plt.legend(loc="best")

        if save:
            filename = title.lower().replace(" ", "_") + ".png"
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()

    def plot_model_comparison_radar(self, metrics_dict, model_names=None, metric_labels=None, save=True):
        """
        Model karşılaştırmasını radar grafik ile görselleştirir.

        Parameters:
        -----------
        metrics_dict : Dict[str, Dict[str, float]]
            Her model için metrik değerleri içeren sözlük
        model_names : list, optional
            Model isimleri
        metric_labels : list, optional
            Metrik etiketleri
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        if model_names is None:
            model_names = list(metrics_dict.keys())

        if metric_labels is None:
            # İlk modelin metriklerini kullan
            first_model = next(iter(metrics_dict.values()))
            metric_labels = list(first_model.keys())

        n_models = len(model_names)
        n_metrics = len(metric_labels)

        # Açıları hesapla
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Kapalı bir poligon oluşturmak için ilk açıyı tekrarla

        # Radar grafik
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Her bir model için
        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name][metric] for metric in metric_labels]
            values += values[:1]  # Kapalı bir poligon oluşturmak için ilk değeri tekrarla

            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)

        # Metrik etiketleri
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)

        # y-eksen sınırları
        ax.set_ylim(0, 1)

        # Izgara çizgileri
        ax.grid(True)

        # Başlık ve lejant
        plt.title('Model Performans Karşılaştırması (Radar Grafik)')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_comparison_radar.png'), dpi=300)
            plt.close()

    def plot_confusion_matrix_heatmap(self, cm, class_names=None, title="Confusion Matrix", save=True):
        """
        Confusion matrix'i heatmap olarak görselleştirir.

        Parameters:
        -----------
        cm : array-like
            Confusion matrix
        class_names : list, optional
            Sınıf isimleri
        title : str, default="Confusion Matrix"
            Grafik başlığı
        save : bool, default=True
            True ise, görselleştirmeyi dosyaya kaydeder

        Returns:
        --------
        None
        """
        plt.figure(figsize=(10, 8))

        # Normalize et
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Sınıf isimleri
        if class_names is None:
            class_names = [f"Sınıf {i}" for i in range(cm.shape[0])]

        # Heatmap
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.title(title)

        if save:
            filename = title.lower().replace(" ", "_") + ".png"
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()

    def generate_summary_report(self, data_info, model_results, output_path=None):
        """
        Özet rapor oluşturur ve HTML dosyası olarak kaydeder.

        Parameters:
        -----------
        data_info : Dict[str, Any]
            Veri seti bilgileri
        model_results : Dict[str, Dict[str, Any]]
            Model sonuçları
        output_path : str, optional
            Rapor dosyasının kaydedileceği yol

        Returns:
        --------
        None
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'summary_report.html')

        # HTML içeriği
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Duygu Analizi Projesi - Özet Rapor</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .section {{ margin-bottom: 30px; }}
                .chart {{ margin: 20px 0; }}
                .footer {{ margin-top: 50px; font-size: 12px; color: #888; }}
            </style>
        </head>
        <body>
            <h1>Duygu Analizi Projesi - Özet Rapor</h1>
            <div class="section">
                <h2>Veri Seti Bilgileri</h2>
                <table>
                    <tr>
                        <th>Özellik</th>
                        <th>Değer</th>
                    </tr>
                    <tr>
                        <td>Eğitim Seti Boyutu</td>
                        <td>{data_info.get('train_size', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Test Seti Boyutu</td>
                        <td>{data_info.get('test_size', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Öznitelik Sayısı</td>
                        <td>{data_info.get('n_features', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Sınıf Sayısı</td>
                        <td>{data_info.get('n_classes', 'N/A')}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>Model Performans Karşılaştırması</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>ROC AUC</th>
                        <th>Eğitim Süresi (s)</th>
                        <th>Tahmin Süresi (s)</th>
                    </tr>
        """

        # Model sonuçlarını ekle
        for model_name, results in model_results.items():
            html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{results.get('accuracy', 'N/A'):.4f}</td>
                        <td>{results.get('precision', 'N/A'):.4f}</td>
                        <td>{results.get('recall', 'N/A'):.4f}</td>
                        <td>{results.get('f1_score', 'N/A'):.4f}</td>
                        <td>{results.get('roc_auc', 'N/A') if results.get('roc_auc') is not None else 'N/A'}</td>
                        <td>{results.get('training_time', 'N/A') if results.get('training_time') is not None else 'N/A'}</td>
                        <td>{results.get('prediction_time', 'N/A'):.4f}</td>
                    </tr>
            """

        # En iyi modeli bul (F1 skoruna göre)
        best_model = max(model_results.items(), key=lambda x: x[1].get('f1_score', 0))
        best_model_name = best_model[0]
        best_model_f1 = best_model[1].get('f1_score', 0)

        html_content += f"""
                </table>
                <p><strong>En iyi model (F1 skoruna göre):</strong> {best_model_name} (F1 = {best_f1:.4f})</p>
            </div>

            <div class="section">
                <h2>Görsel Sonuçlar</h2>
                <p>Detaylı görsel sonuçları 'visualizations' klasöründe bulabilirsiniz.</p>

                <div class="chart">
                    <h3>Model Karşılaştırması</h3>
                    <img src="model_comparison.png" alt="Model Karşılaştırması" style="max-width: 100%;">
                </div>

                <div class="chart">
                    <h3>Eğitim ve Tahmin Süreleri</h3>
                    <img src="model_time_comparison.png" alt="Eğitim ve Tahmin Süreleri" style="max-width: 100%;">
                </div>

                <!-- Diğer görseller... -->
            </div>

            <div class="footer">
                <p>Bu rapor otomatik olarak oluşturulmuştur. Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
        </body>
        </html>
        """

        # HTML dosyasını kaydet
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Özet rapor oluşturuldu: {output_path}")


if __name__ == "__main__":
    # Test amaçlı basit bir çalıştırma örneği
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    # Veriyi yükle
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2,
                                                        random_state=42)

    # Visualizer'ı başlat
    visualizer = Visualizer(output_dir='../results_test/visualizations')

    # Etiket dağılımını görselleştir
    visualizer.plot_label_distribution(y_train, y_test, labels=categories)

    # Doküman uzunluğu dağılımını görselleştir
    visualizer.plot_document_length_distribution(X_train, label='Eğitim Verisi')

    # Kelime frekanslarını görselleştir (ön işleme uygulanmış metinler üzerinde)
    # Basit bir ön işleme
    processed_texts = [' '.join([word.lower() for word in text.split() if len(word) > 2]) for text in X_train]
    visualizer.plot_word_frequency(processed_texts, top_n=20)

    # Sınıf bazında kelime frekanslarını görselleştir
    visualizer.plot_word_frequency_by_class(processed_texts, y_train, class_names=categories)

    # Özellikleri çıkar
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Öznitelik dağılımını görselleştir
    feature_names = vectorizer.get_feature_names_out()
    visualizer.plot_feature_distribution(X_train_tfidf, feature_names, top_n=20)

    # Boyut indirgeme ile görselleştir
    visualizer.plot_dimensionality_reduction(X_train_tfidf, y_train, method='tsne',
                                             class_names=categories, n_components=2)

    # Korelasyon matrisini görselleştir
    visualizer.plot_correlation_matrix(X_train_tfidf, feature_names, top_n=15)

    print("Tüm görselleştirmeler tamamlandı.")