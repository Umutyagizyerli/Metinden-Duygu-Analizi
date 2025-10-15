#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Özellik çıkarma modülü.
Metin verilerinden makine öğrenimi algoritmaları için özellikler çıkarır.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Dict, List, Tuple, Optional
import joblib
import os


class FeatureExtractor:
    """Metinlerden özellik çıkarmak için sınıf"""

    def __init__(self, method: str = 'tfidf',
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_features: Optional[int] = 5000,
                 min_df: Union[int, float] = 2,
                 max_df: Union[int, float] = 0.95):
        """
        FeatureExtractor sınıfını başlatır.

        Parameters:
        -----------
        method : str, default='tfidf'
            Özellik çıkarma yöntemi ('tfidf', 'count', 'both')
        ngram_range : tuple, default=(1, 2)
            N-gram aralığı, örn. (1, 2) unigram ve bigram'ları içerir
        max_features : int, optional, default=5000
            Maksimum özellik sayısı
        min_df : union[int, float], default=2
            Bir terimin bir özellik olarak kabul edilmesi için gereken minimum döküman sayısı
            1'den küçük değerler oran olarak kabul edilir
        max_df : union[int, float], default=0.95
            Bir terimin bir özellik olarak kabul edilmesi için gereken maksimum döküman sayısı
            1'den küçük değerler oran olarak kabul edilir
        """
        self.method = method.lower()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        if self.method not in ['tfidf', 'count', 'both']:
            raise ValueError("Geçersiz method. 'tfidf', 'count' veya 'both' olmalıdır.")

        # Vektörizerleri başlat
        if self.method in ['tfidf', 'both']:
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=True
            )

        if self.method in ['count', 'both']:
            self.count_vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df
            )

    def fit(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractor':
        """
        Vektörizerleri metinlere göre eğitir.

        Parameters:
        -----------
        texts : Union[List[str], pd.Series]
            Eğitim metinleri

        Returns:
        --------
        FeatureExtractor
            Eğitilmiş FeatureExtractor nesnesi
        """
        if self.method in ['tfidf', 'both']:
            self.tfidf_vectorizer.fit(texts)
            print(f"TF-IDF Vectorizer eğitildi. Özellik sayısı: {len(self.tfidf_vectorizer.get_feature_names_out())}")

        if self.method in ['count', 'both']:
            self.count_vectorizer.fit(texts)
            print(f"Count Vectorizer eğitildi. Özellik sayısı: {len(self.count_vectorizer.get_feature_names_out())}")

        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> Dict[str, np.ndarray]:
        """
        Metinleri özellik vektörlerine dönüştürür.

        Parameters:
        -----------
        texts : Union[List[str], pd.Series]
            Dönüştürülecek metinler

        Returns:
        --------
        Dict[str, np.ndarray]
            Her bir yöntem için özellik matrisleri içeren sözlük
        """
        result = {}

        if self.method in ['tfidf', 'both']:
            result['tfidf'] = self.tfidf_vectorizer.transform(texts)

        if self.method in ['count', 'both']:
            result['count'] = self.count_vectorizer.transform(texts)

        return result

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> Dict[str, np.ndarray]:
        """
        Vektörizerleri eğitir ve metinleri dönüştürür.

        Parameters:
        -----------
        texts : Union[List[str], pd.Series]
            Eğitilecek ve dönüştürülecek metinler

        Returns:
        --------
        Dict[str, np.ndarray]
            Her bir yöntem için özellik matrisleri içeren sözlük
        """
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Her vektörizer için özellik isimlerini döndürür.

        Returns:
        --------
        Dict[str, List[str]]
            Her bir yöntem için özellik isimleri içeren sözlük
        """
        result = {}

        if self.method in ['tfidf', 'both']:
            result['tfidf'] = self.tfidf_vectorizer.get_feature_names_out().tolist()

        if self.method in ['count', 'both']:
            result['count'] = self.count_vectorizer.get_feature_names_out().tolist()

        return result

    def save(self, output_dir: str) -> None:
        """
        Eğitilmiş vektörizerleri kaydeder.

        Parameters:
        -----------
        output_dir : str
            Vektörizerlerin kaydedileceği dizin

        Returns:
        --------
        None
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.method in ['tfidf', 'both']:
            joblib.dump(self.tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))

        if self.method in ['count', 'both']:
            joblib.dump(self.count_vectorizer, os.path.join(output_dir, 'count_vectorizer.joblib'))

        print(f"Vektörizerler {output_dir} dizinine kaydedildi.")

    @classmethod
    def load(cls, input_dir: str, method: str = 'both') -> 'FeatureExtractor':
        """
        Kaydedilmiş vektörizerleri yükler.

        Parameters:
        -----------
        input_dir : str
            Vektörizerlerin yükleneceği dizin
        method : str, default='both'
            Yüklenecek vektörizer yöntemi ('tfidf', 'count', 'both')

        Returns:
        --------
        FeatureExtractor
            Yüklenmiş vektörizerlerle oluşturulmuş FeatureExtractor nesnesi
        """
        extractor = cls(method=method)

        if method in ['tfidf', 'both']:
            tfidf_path = os.path.join(input_dir, 'tfidf_vectorizer.joblib')
            if os.path.exists(tfidf_path):
                extractor.tfidf_vectorizer = joblib.load(tfidf_path)
                print(
                    f"TF-IDF Vectorizer yüklendi. Özellik sayısı: {len(extractor.tfidf_vectorizer.get_feature_names_out())}")
            else:
                raise FileNotFoundError(f"TF-IDF Vectorizer dosyası bulunamadı: {tfidf_path}")

        if method in ['count', 'both']:
            count_path = os.path.join(input_dir, 'count_vectorizer.joblib')
            if os.path.exists(count_path):
                extractor.count_vectorizer = joblib.load(count_path)
                print(
                    f"Count Vectorizer yüklendi. Özellik sayısı: {len(extractor.count_vectorizer.get_feature_names_out())}")
            else:
                raise FileNotFoundError(f"Count Vectorizer dosyası bulunamadı: {count_path}")

        return extractor


def extract_features(train_texts: Union[List[str], pd.Series],
                     test_texts: Union[List[str], pd.Series],
                     method: str = 'tfidf',
                     ngram_range: Tuple[int, int] = (1, 2),
                     max_features: Optional[int] = 5000) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], FeatureExtractor]:
    """
    Eğitim ve test metinlerinden özellikler çıkarır.

    Parameters:
    -----------
    train_texts : Union[List[str], pd.Series]
        Eğitim metinleri
    test_texts : Union[List[str], pd.Series]
        Test metinleri
    method : str, default='tfidf'
        Özellik çıkarma yöntemi ('tfidf', 'count', 'both')
    ngram_range : Tuple[int, int], default=(1, 2)
        N-gram aralığı, örn. (1, 2) unigram ve bigram'ları içerir
    max_features : int, optional, default=5000
        Maksimum özellik sayısı

    Returns:
    --------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], FeatureExtractor]
        Eğitim özellikleri, test özellikleri ve eğitilmiş FeatureExtractor
    """
    # FeatureExtractor'ı başlat
    extractor = FeatureExtractor(
        method=method,
        ngram_range=ngram_range,
        max_features=max_features
    )

    # Eğitim metinlerini kullanarak fit_transform
    print("Eğitim metinlerinden özellikler çıkarılıyor...")
    train_features = extractor.fit_transform(train_texts)

    # Test metinlerini kullanarak transform
    print("Test metinlerinden özellikler çıkarılıyor...")
    test_features = extractor.transform(test_texts)

    # İstatistikler
    for feature_type, feature_matrix in train_features.items():
        print(f"{feature_type.upper()} özellikleri:")
        print(f"  Eğitim: {feature_matrix.shape[0]} örnek, {feature_matrix.shape[1]} özellik")
        print(f"  Test: {test_features[feature_type].shape[0]} örnek, {test_features[feature_type].shape[1]} özellik")

    return train_features, test_features, extractor


if __name__ == "__main__":
    # Test amaçlı basit bir çalıştırma örneği
    sample_texts = [
        "Bu bir örnek metindir.",
        "Duygu analizi için ön işleme uygulanacaktır.",
        "Metin özellik çıkarımı test ediliyor.",
        "Sentiment analizi için öznitelik çıkarımı önemlidir."
    ]

    extractor = FeatureExtractor(method='both', ngram_range=(1, 2), max_features=100)
    features = extractor.fit_transform(sample_texts)

    for feature_type, feature_matrix in features.items():
        print(f"\n{feature_type.upper()} özellikleri:")
        print(f"Matris boyutu: {feature_matrix.shape}")
        print("Özellik isimleri (ilk 10):", extractor.get_feature_names()[feature_type][:10])