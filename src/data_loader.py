
"""
Veri yükleme modülü.
CSV formatındaki eğitim ve test veri setlerini yükler.
"""

import os
import pandas as pd
from typing import Tuple, Optional


def load_data(train_path: str = "C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\train.csv",
              test_path: str = "C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\test.csv",
              text_column: str = 'text',
              label_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Eğitim ve test veri setlerini CSV dosyalarından yükler.

    Parameters:
    -----------
    train_path : str, default="C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\train.csv"
        Eğitim veri seti CSV dosyasının yolu
    test_path : str, default="C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\test.csv"
        Test veri seti CSV dosyasının yolu
    text_column : str, default='text'
        Metin verisinin bulunduğu sütun adı
    label_column : str, default='label'
        Etiket (duygu) verisinin bulunduğu sütun adı

    Returns:
    --------
    tuple
        (train_df, test_df) - Eğitim ve test dataframe'leri
    """
    # Dosyaların varlığını kontrol et
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Eğitim veri seti dosyası bulunamadı: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test veri seti dosyası bulunamadı: {test_path}")

    # CSV dosyalarını oku
    try:
        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
    except Exception as e:
        raise Exception(f"CSV dosyalarını okurken hata oluştu: {str(e)}")

    # Gerekli sütunların varlığını kontrol et
    required_columns = [text_column, label_column]
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Eğitim veri setinde '{col}' sütunu bulunamadı")
        if col not in test_df.columns:
            raise ValueError(f"Test veri setinde '{col}' sütunu bulunamadı")

    print(f"Eğitim veri seti yüklendi: {train_df.shape[0]} satır, {train_df.shape[1]} sütun")
    print(f"Test veri seti yüklendi: {test_df.shape[0]} satır, {test_df.shape[1]} sütun")

    # Etiket dağılımlarını göster
    print("\nEğitim veri seti etiket dağılımı:")
    print(train_df[label_column].value_counts())
    print("\nTest veri seti etiket dağılımı:")
    print(test_df[label_column].value_counts())

    # Dataset bilgisi varsa, bu bilgiyi de göster
    if 'dataset' in train_df.columns:
        print("\nEğitim veri seti dataset dağılımı:")
        print(train_df['dataset'].value_counts())

    if 'dataset' in test_df.columns:
        print("\nTest veri seti dataset dağılımı:")
        print(test_df['dataset'].value_counts())

    return train_df, test_df


def get_train_test_data(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        text_column: str = 'text',
                        label_column: str = 'label') -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Dataframe'lerden metin ve etiketleri ayırır.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Eğitim veri seti dataframe'i
    test_df : pd.DataFrame
        Test veri seti dataframe'i
    text_column : str, default='text'
        Metin verisinin bulunduğu sütun adı
    label_column : str, default='label'
        Etiket (duygu) verisinin bulunduğu sütun adı

    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test) - Metin ve etiket serileri
    """
    X_train = train_df[text_column]
    y_train = train_df[label_column]
    X_test = test_df[text_column]
    y_test = test_df[label_column]

    # Veri kontrolü
    if X_train.isnull().sum() > 0:
        print(f"Uyarı: Eğitim metinlerinde {X_train.isnull().sum()} adet eksik değer var.")
    if X_test.isnull().sum() > 0:
        print(f"Uyarı: Test metinlerinde {X_test.isnull().sum()} adet eksik değer var.")

    return X_train, y_train, X_test, y_test


def save_processed_data(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        output_dir: str = 'C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\processed') -> None:
    """
    İşlenmiş veri setlerini kaydeder.

    Parameters:
    -----------
    train_df : pd.DataFrame
        İşlenmiş eğitim veri seti dataframe'i
    test_df : pd.DataFrame
        İşlenmiş test veri seti dataframe'i
    output_dir : str, default='C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\processed'
        Çıktı dizini

    Returns:
    --------
    None
    """
    # Dizinin varlığını kontrol et, yoksa oluştur
    os.makedirs(output_dir, exist_ok=True)

    # İşlenmiş veri setlerini kaydet
    train_output_path = os.path.join(output_dir, 'processed_train.csv')
    test_output_path = os.path.join(output_dir, 'processed_test.csv')

    train_df.to_csv(train_output_path, index=False, encoding='utf-8')
    test_df.to_csv(test_output_path, index=False, encoding='utf-8')

    print(f"İşlenmiş eğitim veri seti kaydedildi: {train_output_path}")
    print(f"İşlenmiş test veri seti kaydedildi: {test_output_path}")


if __name__ == "__main__":
    # Test amaçlı basit bir çalıştırma örneği
    try:
        train_df, test_df = load_data(
            train_path="C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\train.csv",
            test_path="C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\test.csv",
            text_column='text',
            label_column='label'
        )
        X_train, y_train, X_test, y_test = get_train_test_data(train_df, test_df)
        print("Veri yükleme başarılı!")

        # Dataset sütunu varsa, dataset dağılımını incele
        if 'dataset' in train_df.columns:
            print("\nDataset bazında etiket dağılımı:")
            print(train_df.groupby(['dataset', 'label']).size().unstack())
    except Exception as e:
        print(f"Hata: {str(e)}")