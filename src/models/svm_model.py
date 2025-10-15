#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SVM model modülü.
Metin sınıflandırma için Support Vector Machine (SVM) modellerini içerir.
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import joblib
import os
import time
from typing import Dict, Any, Optional, Tuple, Union, List


class SVMModel:
    """Support Vector Machine sınıflandırıcı modeli"""

    def __init__(self, variant: str = 'linear', probability: bool = True, **kwargs):
        """
        SVMModel sınıfını başlatır.

        Parameters:
        -----------
        variant : str, default='linear'
            SVM varyantı ('linear', 'kernel')
        probability : bool, default=True
            Olasılık tahminlerinin etkinleştirilip etkinleştirilmeyeceği
        **kwargs : Dict[str, Any]
            Model parametreleri
        """
        self.variant = variant.lower()
        self.probability = probability
        self.model_params = kwargs
        self.training_time = 0
        self.prediction_time = 0
        self.feature_names = None

        if self.variant == 'linear':
            self.base_model = LinearSVC(**self.model_params)
            # LinearSVC olasılık desteklemez, bu yüzden CalibratedClassifierCV kullanır
            if self.probability:
                self.model = CalibratedClassifierCV(self.base_model)
            else:
                self.model = self.base_model
        elif self.variant == 'kernel':
            self.model = SVC(probability=self.probability, **self.model_params)
            self.base_model = self.model
        else:
            raise ValueError("Geçersiz SVM varyantı. 'linear' veya 'kernel' olmalıdır.")

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
        SVMModel
            Eğitilmiş model
        """
        self.feature_names = feature_names

        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time

        print(f"SVM ({self.variant}) model eğitimi tamamlandı.")
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

        print(f"SVM ({self.variant}) model tahmini tamamlandı.")
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
        if not self.probability:
            raise RuntimeError("Model olasılık tahminleri için yapılandırılmamış. 'probability=True' ile başlatın.")

        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """
        Öznitelik önemlerini (katsayılar) döndürür.
        Linear SVM için geçerlidir.

        Returns:
        --------
        Dict[str, Union[float, List[float]]]
            Öznitelik-önem çiftleri
        """
        if self.variant != 'linear':
            return {}

        if not hasattr(self.base_model, 'coef_'):
            return {}

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.base_model.coef_[0].shape[0])]
        else:
            feature_names = self.feature_names

        # İkili sınıflandırma durumu
        if self.base_model.coef_.shape[0] == 1:
            return {feature: float(coef) for feature, coef in zip(feature_names, self.base_model.coef_[0])}
        # Çoklu sınıflandırma durumu
        else:
            result = {}
            for i, class_coef in enumerate(self.base_model.coef_):
                class_name = f"Class_{i}"
                if hasattr(self.base_model, 'classes_'):
                    class_name = f"Class_{self.base_model.classes_[i]}"

                result[class_name] = {feature: float(coef) for feature, coef in zip(feature_names, class_coef)}

            return result

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
        params['probability'] = self.probability
        return params

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
            'base_model': self.base_model,
            'variant': self.variant,
            'probability': self.probability,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, model_path)
        print(f"SVM ({self.variant}) model kaydedildi: {model_path}")

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
        SVMModel
            Yüklenmiş model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        model_data = joblib.load(model_path)

        instance = cls(
            variant=model_data['variant'],
            probability=model_data['probability']
        )
        instance.model = model_data['model']
        instance.base_model = model_data['base_model']
        instance.training_time = model_data.get('training_time', 0)
        instance.prediction_time = model_data.get('prediction_time', 0)
        instance.feature_names = model_data.get('feature_names', None)

        print(f"SVM ({instance.variant}) model yüklendi: {model_path}")
        return instance


def optimize_svm(X_train, y_train, X_val=None, y_val=None,
                 variant: str = 'linear',
                 cv: int = 5, verbose: int = 1) -> Tuple[SVMModel, Dict[str, Any]]:
    """
    GridSearchCV ile SVM modelini optimize eder.

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
    variant : str, default='linear'
        SVM varyantı ('linear' veya 'kernel')
    cv : int, default=5
        Çapraz doğrulama katlama sayısı
    verbose : int, default=1
        Detay seviyesi

    Returns:
    --------
    Tuple[SVMModel, Dict[str, Any]]
        En iyi model ve optimizasyon sonuçları
    """
    print(f"SVM ({variant}) model optimizasyonu başlatılıyor...")

    if variant == 'linear':
        # LinearSVC için parametre ızgarası
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'loss': ['hinge', 'squared_hinge'],
            'penalty': ['l1', 'l2'],
            'dual': [True, False],
            'tol': [1e-4, 1e-3],
            'max_iter': [1000, 2000, 5000]
        }

        # Bazı kombinasyonlar geçersizdir (örn. L1 penalty ve dual=True)
        from sklearn.model_selection import ParameterGrid
        valid_param_grid = []
        for params in ParameterGrid(param_grid):
            if params['penalty'] == 'l1' and params['dual'] is True:
                continue
            if params['penalty'] == 'l1' and params['loss'] == 'hinge':
                continue

            valid_param_grid.append(params)

        base_model = LinearSVC(random_state=42)

        # GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            valid_param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=verbose
        )

    elif variant == 'kernel':
        # SVC için parametre ızgarası
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            'degree': [2, 3, 4],  # poly kernel için
            'coef0': [0.0, 0.1, 0.5],  # poly ve sigmoid kernel için
            'class_weight': [None, 'balanced']
        }

        base_model = SVC(probability=True, random_state=42)

        # GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=verbose
        )

    else:
        raise ValueError("Geçersiz SVM varyantı. 'linear' veya 'kernel' olmalıdır.")

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    optimization_time = time.time() - start_time

    # En iyi parametreler ve skor
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"SVM ({variant}) model optimizasyonu tamamlandı.")
    print(f"Optimizasyon süresi: {optimization_time:.2f} saniye")
    print(f"En iyi parametreler: {best_params}")
    print(f"En iyi CV skoru: {best_score:.4f}")

    # En iyi parametrelerle modeli oluştur
    if variant == 'linear':
        # probability=True ile CalibratedClassifierCV kullanır
        best_model = SVMModel(variant=variant, probability=True, **best_params)
    else:
        best_model = SVMModel(variant=variant, probability=True, **best_params)

    # Modeli eğit
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

    # Linear SVM Modeli
    print("\nLinear SVM test ediliyor...")
    linear_model = SVMModel(
        variant='linear',
        C=1.0,
        loss='squared_hinge',
        penalty='l2',
        dual=True,
        random_state=42
    )

    # Feature names
    feature_names = vectorizer.get_feature_names_out()

    # Modeli eğit
    linear_model.fit(X_train, y_train, feature_names=feature_names)

    # Tahmin yap
    linear_predictions = linear_model.predict(X_test)

    # Sonuçları değerlendir
    from sklearn.metrics import accuracy_score, classification_report

    linear_accuracy = accuracy_score(y_test, linear_predictions)
    linear_report = classification_report(y_test, linear_predictions)

    print(f"Linear SVM Accuracy: {linear_accuracy:.4f}")
    print("Linear SVM Classification Report:")
    print(linear_report)

    # Kernel SVM Modeli
    print("\nKernel SVM test ediliyor...")
    kernel_model = SVMModel(
        variant='kernel',
        C=10.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )

    # Modeli eğit
    kernel_model.fit(X_train, y_train)

    # Tahmin yap
    kernel_predictions = kernel_model.predict(X_test)

    # Sonuçları değerlendir
    kernel_accuracy = accuracy_score(y_test, kernel_predictions)
    kernel_report = classification_report(y_test, kernel_predictions)

    print(f"Kernel SVM Accuracy: {kernel_accuracy:.4f}")
    print("Kernel SVM Classification Report:")
    print(kernel_report)