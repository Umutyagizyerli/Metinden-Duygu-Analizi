#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Naive Bayes model modülü.
Metin sınıflandırma için Naive Bayes modellerini içerir.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional, Tuple, Union
import joblib
import os
import time


class NaiveBayesModel:
    """Naive Bayes sınıflandırıcı modeli"""

    def __init__(self, variant: str = 'multinomial', **kwargs):
        """
        NaiveBayesModel sınıfını başlatır.

        Parameters:
        -----------
        variant : str, default='multinomial'
            Kullanılacak Naive Bayes varyantı ('multinomial' veya 'complement')
        **kwargs : Dict[str, Any]
            Model parametreleri
        """
        self.variant = variant.lower()
        self.model_params = kwargs
        self.model = None
        self.training_time = 0
        self.prediction_time = 0

        if self.variant == 'multinomial':
            self.model = MultinomialNB(**self.model_params)
        elif self.variant == 'complement':
            self.model = ComplementNB(**self.model_params)
        else:
            raise ValueError("Geçersiz Naive Bayes varyantı. 'multinomial' veya 'complement' olmalıdır.")

    def fit(self, X, y):
        """
        Modeli eğitir.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Eğitim örnekleri
        y : array-like, shape (n_samples,)
            Hedef değerler

        Returns:
        --------
        NaiveBayesModel
            Eğitilmiş model
        """
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        print(f"Naive Bayes ({self.variant}) model eğitimi tamamlandı.")
        print(f"Eğitim süresi: {self.training_time:.2f} saniye")
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
        print(f"Naive Bayes ({self.variant}) model tahmini tamamlandı.")
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

    def get_params(self):
        """
        Model parametrelerini döndürür.

        Returns:
        --------
        Dict[str, Any]
            Model parametreleri
        """
        params = self.model.get_params()
        params['variant'] = self.variant
        return params

    def save(self, model_path: str):
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
            'variant': self.variant,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

        joblib.dump(model_data, model_path)
        print(f"Naive Bayes model kaydedildi: {model_path}")

    @classmethod
    def load(cls, model_path: str):
        """
        Kaydedilmiş modeli yükler.

        Parameters:
        -----------
        model_path : str
            Yüklenecek model dosyası yolu

        Returns:
        --------
        NaiveBayesModel
            Yüklenmiş model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        model_data = joblib.load(model_path)

        instance = cls(variant=model_data['variant'])
        instance.model = model_data['model']
        instance.training_time = model_data['training_time']
        instance.prediction_time = model_data['prediction_time']

        print(f"Naive Bayes model yüklendi: {model_path}")
        return instance


def optimize_naive_bayes(X_train, y_train, X_val=None, y_val=None,
                         variant: str = 'multinomial',
                         cv: int = 5, verbose: int = 1) -> Tuple[NaiveBayesModel, Dict[str, Any]]:
    """
    GridSearchCV ile Naive Bayes modelini optimize eder.

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
    variant : str, default='multinomial'
        Kullanılacak Naive Bayes varyantı ('multinomial' veya 'complement')
    cv : int, default=5
        Çapraz doğrulama katlama sayısı
    verbose : int, default=1
        Detay seviyesi

    Returns:
    --------
    Tuple[NaiveBayesModel, Dict[str, Any]]
        En iyi model ve optimizasyon sonuçları
    """
    print(f"Naive Bayes ({variant}) model optimizasyonu başlatılıyor...")

    # Model sınıfını seç
    if variant == 'multinomial':
        model_class = MultinomialNB
    elif variant == 'complement':
        model_class = ComplementNB
    else:
        raise ValueError("Geçersiz Naive Bayes varyantı. 'multinomial' veya 'complement' olmalıdır.")

    # Parametre ızgarası
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        model_class(),
        param_grid,
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

    print(f"Naive Bayes ({variant}) model optimizasyonu tamamlandı.")
    print(f"Optimizasyon süresi: {optimization_time:.2f} saniye")
    print(f"En iyi parametreler: {best_params}")
    print(f"En iyi CV skoru: {best_score:.4f}")

    # En iyi parametrelerle modeli oluştur ve eğit
    best_model = NaiveBayesModel(variant=variant, **best_params)
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

    # Veriyi yükle
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    # Özellikleri çıkar
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(twenty_train.data)
    y_train = twenty_train.target

    # Modeli başlat ve eğit
    model = NaiveBayesModel(variant='multinomial', alpha=1.0)
    model.fit(X_train, y_train)

    # Test et
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    X_test = vectorizer.transform(twenty_test.data)
    y_test = twenty_test.target

    predictions = model.predict(X_test)

    # Sonuçları değerlendir
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)