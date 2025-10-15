#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metin ön işleme modülü.
Ham metni temizler, normalleştirir ve duygu analizi için hazır hale getirir.
Zemberek kütüphanesi kullanılarak Türkçe lemmatization (kökleme) işlemi yapılır.
"""

import re
import unicodedata
import pandas as pd
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from typing import List, Dict, Union, Optional, Callable
from tqdm import tqdm
import warnings

# JPype ve Zemberek importları
try:
    import jpype
    from jpype import JClass, JString, getDefaultJVMPath

    ZEMBEREK_AVAILABLE = True
except ImportError:
    warnings.warn("JPype veya Zemberek kütüphaneleri bulunamadı. Lemmatization devre dışı bırakılacak.")
    ZEMBEREK_AVAILABLE = False

# NLTK kaynaklarını kontrol et ve gerekirse indir
try:
    stopwords.words('turkish')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Metin ön işleme için sınıf"""

    def __init__(self, language: str = 'turkish',
                 remove_stopwords: bool = True,
                 apply_lemmatization: bool = True,
                 min_word_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None,
                 zemberek_jar_path: Optional[str] = None):
        """
        TextPreprocessor sınıfını başlatır.

        Parameters:
        -----------
        language : str, default='turkish'
            İşlenecek metinlerin dili
        remove_stopwords : bool, default=True
            Stopword'leri kaldırma işleminin uygulanıp uygulanmayacağı
        apply_lemmatization : bool, default=True
            Lemmatization işleminin uygulanıp uygulanmayacağı
        min_word_length : int, default=2
            Kelimelerin minimum uzunluğu
        custom_stopwords : List[str], optional
            Ek olarak kaldırılacak stopword'ler
        zemberek_jar_path : str, optional
            Zemberek JAR dosyasının yolu. Belirtilmezse, varsayılan yoldan aranır.
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.apply_lemmatization = apply_lemmatization and ZEMBEREK_AVAILABLE
        self.min_word_length = min_word_length

        # Kelime tokenizer'ı oluştur (sadece kelimeleri eşleştirir, noktalama işaretlerini atlar)
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Stopwords
        if self.remove_stopwords:
            try:
                # Dil kodu düzeltme
                lang_code = self._get_language_code(language)
                self.stop_words = set(stopwords.words(lang_code))
                print(f"{lang_code} stopwords yüklendi. Toplam {len(self.stop_words)} adet.")

                # Özel stopwords'leri ekle
                if custom_stopwords:
                    self.stop_words.update(custom_stopwords)
            except LookupError:
                print(f"'{language}' dili için stopwords bulunamadı. Basit stopwords listesi kullanılacak.")
                self.stop_words = set(self._basic_turkish_stopwords())
                if custom_stopwords:
                    self.stop_words.update(custom_stopwords)
        else:
            self.stop_words = set()

        # Zemberek kurulumu (eğer apply_lemmatization=True ise)
        if self.apply_lemmatization:
            self._setup_zemberek(zemberek_jar_path)
        else:
            self.morphology = None

    def _setup_zemberek(self, zemberek_jar_path: Optional[str] = None):
        """
        Zemberek kütüphanesini ayarlar.

        Parameters:
        -----------
        zemberek_jar_path : str, optional
            Zemberek JAR dosyasının yolu. Belirtilmezse, varsayılan yollardan aranır.
        """
        # JAR dosyasının yolunu bul
        if zemberek_jar_path is None:
            # Varsayılan yollar içerisinde ara
            possible_paths = [
                "./zemberek-full.jar",
                "../zemberek-full.jar",
                "zemberek-full.jar",
                os.path.join(os.path.expanduser("~"), "zemberek-full.jar"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "zemberek-full.jar")
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    zemberek_jar_path = path
                    break

            if zemberek_jar_path is None:
                warnings.warn(
                    "Zemberek JAR dosyası bulunamadı. Lemmatization devre dışı bırakılacak. "
                    "Zemberek JAR dosyasını https://github.com/ahmetaa/zemberek-nlp/releases "
                    "adresinden indirip, projenin kök dizinine kopyalayabilirsiniz."
                )
                self.apply_lemmatization = False
                return

        # JVM başlatma
        try:
            if not jpype.isJVMStarted():
                jpype.startJVM(getDefaultJVMPath(), f"-Djava.class.path={zemberek_jar_path}", convertStrings=False)

            # Zemberek sınıflarını yükleme
            TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")

            # Türkçe morfoloji analizi için gerekli nesneyi oluştur
            self.morphology = TurkishMorphology.createWithDefaults()

            print("Zemberek başarıyla yüklendi. Lemmatization aktif.")
        except Exception as e:
            warnings.warn(f"Zemberek yüklenirken hata oluştu: {str(e)}. Lemmatization devre dışı bırakılacak.")
            self.apply_lemmatization = False
            self.morphology = None

    def _get_language_code(self, language: str) -> str:
        """
        Dil adını NLTK'nın kullandığı dil koduna dönüştürür.

        Parameters:
        -----------
        language : str
            Dil adı

        Returns:
        --------
        str
            NLTK dil kodu
        """
        # Dil kodu eşleştirmeleri
        language_map = {
            'türkçe': 'turkish',
            'tr': 'turkish',
            'turkish': 'turkish',
            'ingilizce': 'english',
            'en': 'english',
            'english': 'english'
        }

        return language_map.get(language.lower(), language.lower())

    def _basic_turkish_stopwords(self) -> List[str]:
        """
        Temel Türkçe stopwords listesi.

        Returns:
        --------
        List[str]
            Temel Türkçe stopwords listesi
        """
        return [
            "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey", "biz", "bu", "çok",
            "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her",
            "hiç", "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl", "ne", "neden", "nerde",
            "nerede", "nereye", "niçin", "niye", "o", "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya",
            "yani", "bir", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
        ]

    def normalize_text(self, text: str) -> str:
        """
        Metni normalleştirir.

        Parameters:
        -----------
        text : str
            İşlenecek metin

        Returns:
        --------
        str
            Normalleştirilmiş metin
        """
        if not isinstance(text, str):
            return ""

        # Küçük harfe çevir
        text = text.lower()

        # Unicode normalizasyonu
        text = unicodedata.normalize('NFKD', text)

        # URL'leri kaldır
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # HTML etiketlerini kaldır
        text = re.sub(r'<.*?>', '', text)

        # Noktalama işaretlerini ve sayıları kaldır
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Metni tokenize eder.

        Parameters:
        -----------
        text : str
            Tokenize edilecek metin

        Returns:
        --------
        List[str]
            Token listesi
        """
        # RegexpTokenizer ile tokenizasyon (sadece kelimeleri alır, noktalama işaretlerini atlar)
        return self.tokenizer.tokenize(text)

    def lemmatize_word(self, word: str) -> str:
        """
        Kelimenin kökünü bulur (lemmatization).

        Parameters:
        -----------
        word : str
            Kökü bulunacak kelime

        Returns:
        --------
        str
            Kelimenin kökü veya orijinal kelime (lemmatization başarısız olursa)
        """
        if not self.apply_lemmatization or self.morphology is None:
            return word

        try:
            # Zemberek ile kök bulma
            analyses = self.morphology.analyze(JString(word))
            if analyses.size() > 0:
                return str(analyses.get(0).getLemmas()[0])
            return word
        except Exception as e:
            # Hata durumunda orijinal kelimeyi döndür
            return word

    def remove_short_words(self, tokens: List[str]) -> List[str]:
        """
        Kısa kelimeleri kaldırır.

        Parameters:
        -----------
        tokens : List[str]
            Token listesi

        Returns:
        --------
        List[str]
            Filtrelenmiş token listesi
        """
        return [token for token in tokens if len(token) >= self.min_word_length]

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Stopword'leri kaldırır.

        Parameters:
        -----------
        tokens : List[str]
            Token listesi

        Returns:
        --------
        List[str]
            Filtrelenmiş token listesi
        """
        if not self.remove_stopwords:
            return tokens

        return [token for token in tokens if token.lower() not in self.stop_words]

    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Lemmatization (kökleme) işlemi uygular.

        Parameters:
        -----------
        tokens : List[str]
            Token listesi

        Returns:
        --------
        List[str]
            Lemmatize edilmiş token listesi
        """
        if not self.apply_lemmatization or self.morphology is None:
            return tokens

        return [self.lemmatize_word(token) for token in tokens]

    def preprocess_text(self, text: str) -> str:
        """
        Tam metin ön işleme sürecini uygular.

        Parameters:
        -----------
        text : str
            İşlenecek ham metin

        Returns:
        --------
        str
            İşlenmiş metin
        """
        # Normalleştirme
        normalized_text = self.normalize_text(text)

        # Tokenize etme
        tokens = self.tokenize(normalized_text)

        # Kısa kelimeleri kaldırma
        tokens = self.remove_short_words(tokens)

        # Stopword'leri kaldırma
        tokens = self.filter_stopwords(tokens)

        # Lemmatization uygulama
        if self.apply_lemmatization and self.morphology is not None:
            tokens = self.apply_lemmatization(tokens)

        # Tokenları birleştirerek metin oluşturma
        processed_text = ' '.join(tokens)

        return processed_text

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str,
                             processed_column: str = 'processed_text',
                             inplace: bool = False) -> pd.DataFrame:
        """
        DataFrame'deki metinlere ön işleme uygular.

        Parameters:
        -----------
        df : pd.DataFrame
            İşlenecek DataFrame
        text_column : str
            Ham metin sütunu adı
        processed_column : str, default='processed_text'
            İşlenmiş metin sütunu adı
        inplace : bool, default=False
            True ise orijinal DataFrame'i değiştirir, False ise kopyasını döndürür

        Returns:
        --------
        pd.DataFrame
            İşlenmiş DataFrame
        """
        if not inplace:
            df = df.copy()

        # İlerleme çubuğu ile tüm metinlere ön işleme uygula
        tqdm.pandas(desc="Metin ön işleme")
        df[processed_column] = df[text_column].progress_apply(self.preprocess_text)

        return df


def preprocess_data(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    text_column: str = 'text',
                    language: str = 'turkish',
                    zemberek_jar_path: Optional[str] = None) -> tuple:
    """
    Eğitim ve test verilerine ön işleme uygular.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Eğitim veri seti
    test_df : pd.DataFrame
        Test veri seti
    text_column : str, default='text'
        Ham metin sütunu adı
    language : str, default='turkish'
        İşlenecek metinlerin dili
    zemberek_jar_path : str, optional
        Zemberek JAR dosyasının yolu

    Returns:
    --------
    tuple
        (processed_train_df, processed_test_df) - İşlenmiş eğitim ve test veri setleri
    """
    # TextPreprocessor'ı başlat
    preprocessor = TextPreprocessor(
        language=language,
        remove_stopwords=True,
        apply_lemmatization=True,
        min_word_length=2,
        zemberek_jar_path=zemberek_jar_path
    )

    # Eğitim ve test verilerine ön işleme uygula
    print("Eğitim verilerine ön işleme uygulanıyor...")
    processed_train_df = preprocessor.preprocess_dataframe(
        train_df,
        text_column=text_column,
        processed_column='processed_text'
    )

    print("Test verilerine ön işleme uygulanıyor...")
    processed_test_df = preprocessor.preprocess_dataframe(
        test_df,
        text_column=text_column,
        processed_column='processed_text'
    )

    # İstatistikler
    print("\nÖn işleme sonrası istatistikler:")
    print(f"Eğitim veri seti: {processed_train_df.shape[0]} satır")
    print(f"Test veri seti: {processed_test_df.shape[0]} satır")

    # İşlenmiş veri örnekleri göster
    print("\nİşlenmiş metin örnekleri:")
    sample_idx = min(3, len(processed_train_df))
    for i in range(sample_idx):
        print(f"\nÖrnek {i + 1}:")
        print(f"Orijinal: {processed_train_df[text_column].iloc[i]}")
        print(f"İşlenmiş: {processed_train_df['processed_text'].iloc[i]}")

    return processed_train_df, processed_test_df


def close_jvm():
    """JVM'yi kapatır."""
    if ZEMBEREK_AVAILABLE and jpype.isJVMStarted():
        jpype.shutdownJVM()


if __name__ == "__main__":
    # Test amaçlı basit bir çalıştırma örneği
    sample_text = "Bu bir örnek metindir! Duygu analizi için ön işleme uygulanacaktır. https://example.com"

    # Zemberek JAR dosyasının yolunu belirtin
    zemberek_jar_path = "./zemberek-full.jar"  # Bu yolu kendi sisteminize göre değiştirin

    try:
        # Ön işleme sınıfını başlat
        preprocessor = TextPreprocessor(
            language='turkish',
            zemberek_jar_path=zemberek_jar_path,
            apply_lemmatization=True
        )

        # Ön işleme uygula
        processed_text = preprocessor.preprocess_text(sample_text)

        print(f"Orijinal: {sample_text}")
        print(f"İşlenmiş: {processed_text}")
    finally:
        # JVM'yi kapat
        close_jvm()