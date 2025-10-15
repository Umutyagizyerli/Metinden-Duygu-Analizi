#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model değerlendirme modülü.
Sınıflandırıcı modellerinin performansını değerlendirir ve karşılaştırır.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
import os
from typing import Dict, List, Any, Tuple, Union, Optional
import time
import joblib


class ModelEvaluator:
    """Sınıflandırıcı modellerini değerlendirme sınıfı"""

    def __init__(self, output_dir: str = '../results'):
        """
        ModelEvaluator sınıfını başlatır.

        Parameters:
        -----------
        output_dir : str, default='../results'
            Sonuçların kaydedileceği dizin
        """
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        self.viz_dir = os.path.join(output_dir, 'visualizations')

        # Dizinleri oluştur
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        # Sonuçlar
        self.model_results = {}

    def evaluate_model(self, model, X_test, y_test, model_name: str,
                       X_train=None, y_train=None, feature_names=None):
        """
        Tek bir modeli değerlendirir.

        Parameters:
        -----------
        model : object
            Değerlendirilecek model (predict ve predict_proba metodları olmalı)
        X_test : array-like or sparse matrix
            Test öznitelikleri
        y_test : array-like
            Test etiketleri
        model_name : str
            Model adı
        X_train : array-like or sparse matrix, optional
            Eğitim öznitelikleri (öğrenme eğrileri için)
        y_train : array-like, optional
            Eğitim etiketleri (öğrenme eğrileri için)
        feature_names : array-like, optional
            Özellik isimleri (öznitelik önemleri için)

        Returns:
        --------
        Dict[str, Any]
            Değerlendirme metrikleri ve sonuçları
        """
        print(f"\n{model_name} modeli değerlendiriliyor...")

        # Tahminler
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time

        # Olasılık tahminleri (model destekliyorsa)
        try:
            y_pred_proba = model.predict_proba(X_test)
            has_proba = True
        except (AttributeError, NotImplementedError):
            y_pred_proba = None
            has_proba = False

        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Sınıflandırma raporu
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC AUC (ikili sınıflandırma ise)
        if has_proba:
            if len(np.unique(y_test)) == 2:  # İkili sınıflandırma
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                except IndexError:
                    fpr, tpr = None, None
                    roc_auc = None
            else:  # Çoklu sınıflandırma
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    fpr, tpr = None, None
                except (ValueError, IndexError):
                    roc_auc = None
                    fpr, tpr = None, None
        else:
            roc_auc = None
            fpr = tpr = None

        # Öznitelik önemleri (model destekliyorsa)
        if hasattr(model, 'get_feature_importance') and callable(getattr(model, 'get_feature_importance')):
            feature_importance = model.get_feature_importance()
        else:
            feature_importance = None

        # Eğitim ve tahmin süreleri
        try:
            training_time = getattr(model, 'training_time', None)
        except (AttributeError, TypeError):
            training_time = None

        # Sonuçları topla
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm,
            'classification_report': report,
            'feature_importance': feature_importance,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'has_proba': has_proba,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Sonuçları kaydet
        self.model_results[model_name] = results

        # Konsola yazdır
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        if training_time is not None:
            print(f"Eğitim süresi: {training_time:.2f} saniye")
        print(f"Tahmin süresi: {prediction_time:.2f} saniye")

        # Öğrenme eğrileri
        if X_train is not None and y_train is not None:
            # Öğrenme eğrisi hesapla (memory hatalarını önlemek için boyut küçültme)
            max_samples = min(10000, X_train.shape[0])
            if X_train.shape[0] > max_samples:
                from sklearn.model_selection import train_test_split
                X_sample, _, y_sample, _ = train_test_split(
                    X_train, y_train, train_size=max_samples, random_state=42, stratify=y_train
                )
            else:
                X_sample, y_sample = X_train, y_train

            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    model.model if hasattr(model, 'model') else model,
                    X_sample, y_sample,
                    cv=5,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 5)
                )

                results['learning_curve'] = {
                    'train_sizes': train_sizes,
                    'train_scores': train_scores,
                    'test_scores': test_scores
                }
            except Exception as e:
                print(f"Öğrenme eğrisi hesaplanamadı: {str(e)}")
                results['learning_curve'] = None

        return results

    def compare_models(self, models_dict: Dict[str, Any], X_test, y_test,
                       X_train=None, y_train=None, feature_names=None):
        """
        Birden fazla modeli değerlendirir ve karşılaştırır.

        Parameters:
        -----------
        models_dict : Dict[str, Any]
            Modeller sözlüğü (anahtar: model adı, değer: model nesnesi)
        X_test : array-like or sparse matrix
            Test öznitelikleri
        y_test : array-like
            Test etiketleri
        X_train : array-like or sparse matrix, optional
            Eğitim öznitelikleri (öğrenme eğrileri için)
        y_train : array-like, optional
            Eğitim etiketleri (öğrenme eğrileri için)
        feature_names : array-like, optional
            Özellik isimleri (öznitelik önemleri için)

        Returns:
        --------
        pd.DataFrame
            Karşılaştırma metrikleri DataFrame'i
        """
        for model_name, model in models_dict.items():
            self.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names
            )

        # Karşılaştırma DataFrame'i oluştur
        comparison_data = []
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'ROC AUC': results['roc_auc'] if results['roc_auc'] is not None else np.nan,
                'Training Time (s)': results['training_time'] if results['training_time'] is not None else np.nan,
                'Prediction Time (s)': results['prediction_time']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # En iyi modeli bul (F1 skoruna göre)
        best_model_idx = comparison_df['F1 Score'].idxmax()
        best_model_name = comparison_df.loc[best_model_idx, 'Model']

        print("\nModel Karşılaştırması:")
        print(comparison_df.to_string(index=False))
        print(f"\nEn iyi model (F1 skoruna göre): {best_model_name}")

        # CSV olarak kaydet
        comparison_df.to_csv(os.path.join(self.metrics_dir, 'model_comparison.csv'), index=False)
        print(f"Model karşılaştırması kaydedildi: {os.path.join(self.metrics_dir, 'model_comparison.csv')}")

        return comparison_df

    def plot_confusion_matrices(self, save=True):
        """
        Tüm modeller için confusion matrix'leri görselleştirir.

        Parameters:
        -----------
        save : bool, default=True
            True ise, görselleştirmeleri dosyaya kaydeder

        Returns:
        --------
        None
        """
        n_models = len(self.model_results)
        if n_models == 0:
            print("Görselleştirilecek model sonucu yok.")
            return

        # Her model için bir confusion matrix çiz
        for model_name, results in self.model_results.items():
            plt.figure(figsize=(8, 6))
            cm = results['confusion_matrix']

            # Normalize et
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Görselleştir
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Gerçek Etiket')
            plt.xlabel('Tahmin Edilen Etiket')

            if save:
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
                plt.close()

    def plot_roc_curves(self, save=True):
        """
        ROC eğrilerini görselleştirir (ikili sınıflandırma için).

        Parameters:
        -----------
        save : bool, default=True
            True ise, görselleştirmeleri dosyaya kaydeder

        Returns:
        --------
        None
        """
        # ROC eğrisi olan modelleri filtrele
        valid_models = [(name, results) for name, results in self.model_results.items()
                        if results['fpr'] is not None and results['tpr'] is not None]

        if not valid_models:
            print("ROC eğrisi çizilebilecek model sonucu yok.")
            return

        plt.figure(figsize=(10, 8))

        for model_name, results in valid_models:
            fpr = results['fpr']
            tpr = results['tpr']
            roc_auc = results['roc_auc']

            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        # Referans çizgisi
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'roc_curves.png'), dpi=300)
            plt.close()

    def plot_precision_recall_curves(self, save=True):
        """
        Precision-Recall eğrilerini görselleştirir.

        Parameters:
        -----------
        save : bool, default=True
            True ise, görselleştirmeleri dosyaya kaydeder

        Returns:
        --------
        None
        """
        # Olasılık tahmini yapabilen modelleri filtrele
        valid_models = [(name, results) for name, results in self.model_results.items()
                        if results['has_proba'] and results['y_pred_proba'] is not None]

        if not valid_models:
            print("Precision-Recall eğrisi çizilebilecek model sonucu yok.")
            return

        plt.figure(figsize=(10, 8))

        for model_name, results in valid_models:
            y_test = results['y_pred']  # Gerçek değerler
            y_pred_proba = results['y_pred_proba']

            # İkili sınıflandırma durumu
            if y_pred_proba.shape[1] == 2:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
                avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])

                plt.plot(recall, precision, lw=2,
                         label=f'{model_name} (AP = {avg_precision:.3f})')
            else:
                # Çoklu sınıflandırma durumu - her sınıf için ayrı eğriler çiz
                n_classes = y_pred_proba.shape[1]
                # Çok fazla sınıf varsa, ilk 5 sınıfı göster
                if n_classes > 5:
                    n_classes = 5

                for i in range(n_classes):
                    y_test_bin = (y_test == i).astype(int)
                    precision, recall, _ = precision_recall_curve(y_test_bin, y_pred_proba[:, i])
                    avg_precision = average_precision_score(y_test_bin, y_pred_proba[:, i])

                    plt.plot(recall, precision, lw=2,
                             label=f'{model_name} - Class {i} (AP = {avg_precision:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'precision_recall_curves.png'), dpi=300)
            plt.close()

    def plot_feature_importance(self, top_n: int = 20, save=True):
        """
        Öznitelik önemlerini görselleştirir.

        Parameters:
        -----------
        top_n : int, default=20
            Gösterilecek en önemli öznitelik sayısı
        save : bool, default=True
            True ise, görselleştirmeleri dosyaya kaydeder

        Returns:
        --------
        None
        """
        # Öznitelik önemi olan modelleri filtrele
        valid_models = [(name, results) for name, results in self.model_results.items()
                        if results['feature_importance'] is not None]

        if not valid_models:
            print("Öznitelik önemi görselleştirilebilecek model sonucu yok.")
            return

        for model_name, results in valid_models:
            feature_importance = results['feature_importance']

            # Öznitelik önemi bir sözlük ise
            if isinstance(feature_importance, dict) and not isinstance(next(iter(feature_importance.values())), dict):
                # En önemli N özniteliği al
                top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
                feature_names = [item[0] for item in top_features]
                importances = [item[1] for item in top_features]

                plt.figure(figsize=(12, 8))
                colors = ['#1f77b4' if i >= 0 else '#d62728' for i in importances]
                y_pos = np.arange(len(feature_names))

                plt.barh(y_pos, importances, align='center', color=colors)
                plt.yticks(y_pos, feature_names)
                plt.xlabel('Öznitelik Önemi')
                plt.title(f'En Önemli {top_n} Öznitelik - {model_name}')

                if save:
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.viz_dir, f'feature_importance_{model_name}.png'), dpi=300)
                    plt.close()
            elif isinstance(feature_importance, dict) and isinstance(next(iter(feature_importance.values())), dict):
                # Çoklu sınıflandırma durumu - her sınıf için ayrı görselleştir
                for class_name, class_importances in feature_importance.items():
                    # En önemli N özniteliği al
                    top_features = sorted(class_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
                    feature_names = [item[0] for item in top_features]
                    importances = [item[1] for item in top_features]

                    plt.figure(figsize=(12, 8))
                    colors = ['#1f77b4' if i >= 0 else '#d62728' for i in importances]
                    y_pos = np.arange(len(feature_names))

                    plt.barh(y_pos, importances, align='center', color=colors)
                    plt.yticks(y_pos, feature_names)
                    plt.xlabel('Öznitelik Önemi')
                    plt.title(f'En Önemli {top_n} Öznitelik - {model_name} - {class_name}')

                    if save:
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.viz_dir, f'feature_importance_{model_name}_{class_name}.png'),
                                    dpi=300)
                        plt.close()

    def plot_learning_curves(self, save=True):
        """
        Öğrenme eğrilerini görselleştirir.

        Parameters:
        -----------
        save : bool, default=True
            True ise, görselleştirmeleri dosyaya kaydeder

        Returns:
        --------
        None
        """
        # Öğrenme eğrisi olan modelleri filtrele
        valid_models = [(name, results) for name, results in self.model_results.items()
                        if 'learning_curve' in results and results['learning_curve'] is not None]

        if not valid_models:
            print("Öğrenme eğrisi görselleştirilebilecek model sonucu yok.")
            return

        for model_name, results in valid_models:
            lc_data = results['learning_curve']
            train_sizes = lc_data['train_sizes']
            train_scores = lc_data['train_scores']
            test_scores = lc_data['test_scores']

            plt.figure(figsize=(10, 6))

            # Eğitim skoru
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Eğitim skoru")

            # Test skoru
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Çapraz doğrulama skoru")

            plt.title(f"Öğrenme Eğrisi - {model_name}")
            plt.xlabel("Eğitim örnekleri sayısı")
            plt.ylabel("F1 Skoru")
            plt.grid(True)
            plt.legend(loc="best")

            if save:
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, f'learning_curve_{model_name}.png'), dpi=300)
                plt.close()

    def plot_model_comparison(self, save=True):
        """
        Model karşılaştırmasını görselleştirir.

        Parameters:
        -----------
        save : bool, default=True
            True ise, görselleştirmeleri dosyaya kaydeder

        Returns:
        --------
        None
        """
        if not self.model_results:
            print("Karşılaştırılacak model sonucu yok.")
            return

        # Karşılaştırma DataFrame'i oluştur
        comparison_data = []
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'ROC AUC': results['roc_auc'] if results['roc_auc'] is not None else np.nan
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Metrikleri çiz
        plt.figure(figsize=(12, 8))

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        if not comparison_df['ROC AUC'].isna().all():
            metrics.append('ROC AUC')

        comparison_df_melted = pd.melt(comparison_df, id_vars=['Model'], value_vars=metrics,
                                       var_name='Metric', value_name='Score')

        sns.barplot(x='Metric', y='Score', hue='Model', data=comparison_df_melted)
        plt.title('Model Performans Karşılaştırması')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'model_comparison.png'), dpi=300)
            plt.close()

        # Eğitim ve tahmin sürelerini çiz
        time_comparison_data = []
        for model_name, results in self.model_results.items():
            time_comparison_data.append({
                'Model': model_name,
                'Training Time (s)': results['training_time'] if results['training_time'] is not None else 0,
                'Prediction Time (s)': results['prediction_time']
            })

        time_comparison_df = pd.DataFrame(time_comparison_data)

        plt.figure(figsize=(10, 6))
        time_comparison_df_melted = pd.melt(time_comparison_df, id_vars=['Model'],
                                            value_vars=['Training Time (s)', 'Prediction Time (s)'],
                                            var_name='Metric', value_name='Time (s)')

        sns.barplot(x='Model', y='Time (s)', hue='Metric', data=time_comparison_df_melted)
        plt.title('Model Eğitim ve Tahmin Süreleri')
        plt.xticks(rotation=45)
        plt.legend(title='Metrik')
        plt.yscale('log')  # Logaritmik ölçek (sürelerde büyük farklar olabilir)

        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'model_time_comparison.png'), dpi=300)
            plt.close()

    def generate_all_visualizations(self):
        """
        Tüm görselleştirmeleri oluşturur.

        Returns:
        --------
        None
        """
        print("\nTüm görselleştirmeler oluşturuluyor...")

        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_feature_importance()
        self.plot_learning_curves()
        self.plot_model_comparison()

        print(f"Tüm görselleştirmeler {self.viz_dir} dizinine kaydedildi.")

    def save_results(self, output_path=None):
        """
        Değerlendirme sonuçlarını kaydeder.

        Parameters:
        -----------
        output_path : str, optional
            Sonuçların kaydedileceği dosya yolu. Belirtilmezse, default dizin kullanılır.

        Returns:
        --------
        None
        """
        if not self.model_results:
            print("Kaydedilecek sonuç yok.")
            return

        if output_path is None:
            output_path = os.path.join(self.metrics_dir, 'evaluation_results.joblib')

        # Dizinin varlığını kontrol et
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Numpy array'leri ve pandas DataFrame'leri liste veya sözlüğe dönüştür
        serializable_results = {}
        for model_name, results in self.model_results.items():
            serializable_model_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_model_results[key] = value.tolist() if value is not None else None
                elif isinstance(value, pd.DataFrame):
                    serializable_model_results[key] = value.to_dict() if value is not None else None
                else:
                    serializable_model_results[key] = value

            serializable_results[model_name] = serializable_model_results

        joblib.dump(serializable_results, output_path)
        print(f"Değerlendirme sonuçları kaydedildi: {output_path}")

    def load_results(self, input_path):
        """
        Kaydedilmiş değerlendirme sonuçlarını yükler.

        Parameters:
        -----------
        input_path : str
            Yüklenecek sonuç dosyası yolu

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Yüklenmiş sonuçlar
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Sonuç dosyası bulunamadı: {input_path}")

        self.model_results = joblib.load(input_path)
        print(f"Değerlendirme sonuçları yüklendi: {input_path}")

        return self.model_results


if __name__ == "__main__":
    # Test amaçlı basit bir çalıştırma örneği
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    # Veriyi yükle
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

    # Özellikleri çıkar
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target

    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelleri eğit
    models = {
        'Naive Bayes': MultinomialNB().fit(X_train, y_train),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train),
        'Linear SVM': LinearSVC(random_state=42).fit(X_train, y_train)
    }

    # ModelEvaluator'ı başlat
    evaluator = ModelEvaluator(output_dir='../results_test')

    # Modelleri karşılaştır
    comparison_df = evaluator.compare_models(models, X_test, y_test, X_train, y_train)

    # Görselleştirmeleri oluştur
    evaluator.generate_all_visualizations()

    # Sonuçları kaydet
    evaluator.save_results()