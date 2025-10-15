#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Maximum Entropy (Maxent) model modülü.
Metin sınıflandırma için Maxent (Logistic Regression) modellerini içerir.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import os
import time
from typing import Dict, Any, Optional, Tuple, Union


class MaxentModel:
    """Maximum Entropy (Logistic Regression) sınıflandırıcı modeli"""

    def __init__(self, **kwargs):
        """
        MaxentModel sınıfını başlatır.

        Parameters:
        -----------
        **kwargs : Dict[str, Any]
            Model parametreleri, örn. C, solver, penalty, vb.
        """
        self.model_params = kwargs
        self.model = LogisticRegression(**self.model_params)
        self.training_time = 0
        self.prediction_time = 0
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        """
        Modeli eğitir.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Eğitim örnekleri
        y : array-like, shape (n_samples,)
            Hedef değerler
        feature_names : array-like, optional
            Özellik isimleri

        Returns:
        --------
        MaxentModel
            Eğitilmiş model
        """
        self.feature_names = feature_names

        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time

        print("Maxent (Logistic Regression) model eğitimi tamamlandı.")
        print(f"Eğitim süresi: {self.training_time:.2f} saniye")

        # İterasyonlar (sadece belirli solverlar için geçerli)
        if hasattr(self.model, 'n_iter_'):
            print(f"İterasyon sayısı: {self.model.n_iter_}")

        return self

    def predict(self, X):
        """
        Tahmin yapar.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Tahmin edilecek örnekler

        Returns:
        --------
        array-like
            Tahmin edilen etiketler
        """
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time

        print("Maxent (Logistic Regression) model tahmini tamamlandı.")
        print(f"Tahmin süresi: {self.prediction_time:.2f} saniye")

        return predictions

    def predict_proba(self, X):
        """
        Olasılık tahminleri yapar.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Olasılığı tahmin edilecek örnekler

        Returns:
        --------
        array-like, shape (n_samples, n_classes)
            Her sınıf için tahmin edilen olasılıklar
        """
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """
        Öznitelik önemlerini (katsayılar) döndürür.

        Returns:
        --------
        Dict[str, np.ndarray]
            Her sınıf için öznitelik-önem çiftleri
        """
        if not hasattr(self.model, 'coef_'):
            return {}

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.model.coef_[0].shape[0])]
        else:
            feature_names = self.feature_names

        # İkili sınıflandırma durumu
        if self.model.coef_.shape[0] == 1:
            return {feature: coef for feature, coef in zip(feature_names, self.model.coef_[0])}
        # Çoklu sınıflandırma durumu
        else:
            result = {}
            for i, class_coef in enumerate(self.model.coef_):
                class_name = f"Class_{i}"
                if hasattr(self.model, 'classes_'):
                    class_name = f"Class_{self.model.classes_[i]}"

                result[class_name] = {feature: coef for feature, coef in zip(feature_names, class_coef)}

            return result

    def get_params(self):
        """
        Model parametrelerini döndürür.

        Returns:
        --------
        Dict[str, Any]
            Model parametreleri
        """
        return self.model.get_params()

    def save(self, model_path):
        """
        Modeli kaydeder.

        Parameters:
        -----------
        model_path : str
            Modelin kaydedileceği dosya yolu

        Returns:
        --------
        None
        """
        # Dizinin varlığını kontrol et
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model_data = {
            'model': self.model,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, model_path)
        print(f"Maxent (Logistic Regression) model kaydedildi: {model_path}")

    @classmethod
    def load(cls, model_path):
        """
        Kaydedilmiş modeli yükler.

        Parameters:
        -----------
        model_path : str
            Yüklenecek model dosyası yolu

        Returns:
        --------
        MaxentModel
            Yüklenmiş model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        model_data = joblib.load(model_path)

        instance = cls()
        instance.model = model_data['model']
        instance.training_time = model_data.get('training_time', 0)
        instance.prediction_time = model_data.get('prediction_time', 0)
        instance.feature_names = model_data.get('feature_names', None)

        print(f"Maxent (Logistic Regression) model yüklendi: {model_path}")
        return instance


def optimize_maxent(X_train, y_train, X_val=None, y_val=None,
                    cv: int = 5, verbose: int = 1) -> Tuple[MaxentModel, Dict[str, Any]]:
    """
    GridSearchCV ile Maxent (Logistic Regression) modelini optimize eder.

    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Eğitim öznitelikleri
    y_train : array-like
        Eğitim etiketleri
    X_val : array-like or sparse matrix, optional
        Doğrulama öznitelikleri (None ise CV kullanılır)
    y_val : array-like, optional
        Doğrulama etiketleri (None ise CV kullanılır)
    cv : int, default=5
        Çapraz doğrulama katlama sayısı
    verbose : int, default=1
        Detay seviyesi

    Returns:
    --------
    Tuple[MaxentModel, Dict[str, Any]]
        En iyi model ve optimizasyon sonuçları
    """
    print("Maxent (Logistic Regression) model optimizasyonu başlatılıyor...")

    # Parametre ızgarası
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 500]
    }

    # Solver-penalty uyumsuzluğunu engelle
    # Farklı solver'lar farklı penalty'leri destekler
    from sklearn.model_selection import ParameterGrid
    valid_param_grid = []
    for params in ParameterGrid(param_grid):
        solver = params['solver']
        penalty = params['penalty']

        # liblinear: l1, l2
        if solver == 'liblinear' and penalty not in ['l1', 'l2']:
            continue
        # newton-cg, sag, lbfgs: l2, None
        elif solver in ['newton-cg', 'sag', 'lbfgs'] and penalty not in ['l2', None]:
            continue
        # saga: l1, l2, elasticnet, None
        # else: solver is saga, all penalties are valid

        valid_param_grid.append(params)

    # GridSearchCV
    grid_search = GridSearchCV(
        LogisticRegression(),
        valid_param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=verbose
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    optimization_time = time.time() - start_time

    # En iyi parametreler ve skor
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Maxent (Logistic Regression) model optimizasyonu tamamlandı.")
    print(f"Optimizasyon süresi: {optimization_time:.2f} saniye")
    print(f"En iyi parametreler: {best_params}")
    print(f"En iyi CV skoru: {best_score:.4f}")

    # En iyi parametrelerle modeli oluştur ve eğit
    best_model = MaxentModel(**best_params)
    best_model.fit(X_train, y_train)

    # Doğrulama seti varsa, performansı değerlendir
    if X_val is not None and y_val is not None:
        val_predictions = best_model.predict(X_val)
        from sklearn.metrics import accuracy_score, f1_score
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions, average='weighted')
        print(f"Doğrulama seti performansı:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  F1 score: {val_f1:.4f}")

    # Sonuçları hazırla
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': grid_search.cv_results_,
        'optimization_time': optimization_time
    }

    return best_model, results


if __name__ == "__main__":
    # Test amaçlı basit bir çalıştırma örneği
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    # Veriyi yükle
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

    # Özellikleri çıkar
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target

    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli başlat ve eğit
    model = MaxentModel(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=100,
        random_state=42
    )

    # Feature names
    feature_names = vectorizer.get_feature_names_out()

    # Modeli eğit
    model.fit(X_train, y_train, feature_names=feature_names)

    # Tahmin yap
    predictions = model.predict(X_test)

    # Sonuçları değerlendir
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Öznitelik önemleri
    importances = model.get_feature_importance()
    if isinstance(importances, dict) and not isinstance(next(iter(importances.values())), dict):
        # İkili sınıflandırma durumu
        top_n = 10
        top_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        print(f"\nEn önemli {top_n} öznitelik:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")