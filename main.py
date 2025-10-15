#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ana çalıştırma dosyası.
Projenin tüm adımlarını çalıştırır ve sonuçları kaydeder.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Any, Tuple, Union, Optional
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Proje modüllerini import et
from src.data_loader import load_data, get_train_test_data, save_processed_data
from src.preprocessor import preprocess_data, TextPreprocessor, close_jvm
from src.feature_extractor import extract_features, FeatureExtractor
from src.models.naive_bayes_model import NaiveBayesModel, optimize_naive_bayes
from src.models.decision_tree_model import DecisionTreeModel, optimize_decision_tree
from src.models.maxent_model import MaxentModel, optimize_maxent
from src.models.svm_model import SVMModel, optimize_svm
from src.models.random_forest_model import RandomForestModel, optimize_random_forest
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer


def parse_arguments():
    """
    Komut satırı argümanlarını ayrıştırır.

    Returns:
    --------
    argparse.Namespace
        Argüman değerleri
    """
    parser = argparse.ArgumentParser(description='Duygu Analizi Projesi')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Veri dizini (default: ./data)')

    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Sonuçlar dizini (default: ./results)')

    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Eğitim veri seti dosyası (default: train.csv)')

    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Test veri seti dosyası (default: test.csv)')

    parser.add_argument('--text_column', type=str, default='text',
                        help='Metin sütunu adı (default: text)')

    parser.add_argument('--label_column', type=str, default='label',
                        help='Etiket sütunu adı (default: label)')

    parser.add_argument('--language', type=str, default='turkish',
                        help='Metin dili (default: turkish)')

    parser.add_argument('--feature_method', type=str, default='tfidf',
                        choices=['tfidf', 'count', 'both'],
                        help='Özellik çıkarma yöntemi (default: tfidf)')

    parser.add_argument('--max_features', type=int, default=5000,
                        help='Maksimum özellik sayısı (default: 5000)')

    parser.add_argument('--ngram_range', type=str, default='1,2',
                        help='N-gram aralığı, virgülle ayrılmış (default: 1,2)')

    parser.add_argument('--optimize', action='store_true',
                        help='Hiperparametre optimizasyonu uygula')

    parser.add_argument('--models', type=str, default='all',
                        help='Çalıştırılacak modeller, virgülle ayrılmış (naive_bayes, decision_tree, maxent, svm, random_forest, all)')

    parser.add_argument('--cv', type=int, default=5,
                        help='Çapraz doğrulama katlama sayısı (default: 5)')

    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Ön işleme adımını atla')

    parser.add_argument('--skip_feature_extraction', action='store_true',
                        help='Özellik çıkarma adımını atla')

    parser.add_argument('--skip_model_training', action='store_true',
                        help='Model eğitim adımını atla')

    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Model değerlendirme adımını atla')

    parser.add_argument('--skip_visualization', action='store_true',
                        help='Görselleştirme adımını atla')

    parser.add_argument('--zemberek_jar_path', type=str, default='./zemberek-full.jar',
                        help='Zemberek JAR dosyasının yolu (default: ./zemberek-full.jar)')

    parser.add_argument('--apply_lemmatization', action='store_true', default=True,
                        help='Lemmatization uygula (default: True)')

    return parser.parse_args()


def main():
    """
    Ana fonksiyon.
    Projenin tüm adımlarını çalıştırır.
    """
    try:
        # Argümanları ayrıştır
        args = parse_arguments()

        # Dizin yolları
        data_dir = args.data_dir
        results_dir = args.results_dir
        models_dir = os.path.join(results_dir, 'models')
        metrics_dir = os.path.join(results_dir, 'metrics')
        viz_dir = os.path.join(results_dir, 'visualizations')

        # Dizinleri oluştur
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        # Dosya yolları
        train_path = "C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\train.csv"
        test_path = "C:\\Users\\karad\\Desktop\\MyPythonProject\\duygu_analizi\\data\\test.csv"

        # N-gram aralığını ayır
        ngram_range = tuple(map(int, args.ngram_range.split(',')))

        # Çalıştırılacak modelleri belirle
        all_models = ['naive_bayes', 'decision_tree', 'maxent', 'svm', 'random_forest']
        if args.models.lower() == 'all':
            models_to_run = all_models
        else:
            models_to_run = [model.strip().lower() for model in args.models.split(',')]

        # Başlangıç zamanı
        start_time = time.time()

        print("=" * 80)
        print("Duygu Analizi Projesi")
        print("=" * 80)

        # 1. Veri Yükleme
        print("\n1. Veri Yükleme")
        print("-" * 80)

        try:
            train_df, test_df = load_data(
                train_path=train_path,
                test_path=test_path,
                text_column=args.text_column,
                label_column=args.label_column
            )

            X_train_raw, y_train, X_test_raw, y_test = get_train_test_data(
                train_df, test_df,
                text_column=args.text_column,
                label_column=args.label_column
            )

            print("Veri yükleme başarılı!")
        except Exception as e:
            print(f"Veri yükleme hatası: {str(e)}")
            sys.exit(1)

        # 2. Ön İşleme
        print("\n2. Metin Ön İşleme")
        print("-" * 80)

        if not args.skip_preprocessing:
            try:
                processed_train_df, processed_test_df = preprocess_data(
                    train_df, test_df,
                    text_column=args.text_column,
                    language=args.language,
                    zemberek_jar_path=args.zemberek_jar_path
                )

                # İşlenmiş verileri kaydet
                save_processed_data(processed_train_df, processed_test_df,
                                    output_dir=os.path.join(data_dir, 'processed'))

                # İşlenmiş metinleri al
                X_train = processed_train_df['processed_text']
                X_test = processed_test_df['processed_text']

                print("Metin ön işleme başarılı!")
            except Exception as e:
                print(f"Metin ön işleme hatası: {str(e)}")
                sys.exit(1)
        else:
            print("Ön işleme adımı atlandı.")
            X_train = X_train_raw
            X_test = X_test_raw

        # 3. Özellik Çıkarma
        print("\n3. Özellik Çıkarma")
        print("-" * 80)

        if not args.skip_feature_extraction:
            try:
                train_features, test_features, extractor = extract_features(
                    X_train, X_test,
                    method=args.feature_method,
                    ngram_range=ngram_range,
                    max_features=args.max_features
                )

                # Extractor'ı kaydet
                extractor.save(os.path.join(models_dir, 'feature_extractor'))

                # Özellik matrislerini belirle
                if args.feature_method == 'tfidf':
                    X_train_features = train_features['tfidf']
                    X_test_features = test_features['tfidf']
                    feature_type = 'tfidf'
                elif args.feature_method == 'count':
                    X_train_features = train_features['count']
                    X_test_features = test_features['count']
                    feature_type = 'count'
                else:  # both
                    # Varsayılan olarak tfidf kullan
                    X_train_features = train_features['tfidf']
                    X_test_features = test_features['tfidf']
                    feature_type = 'tfidf'

                # Özellik isimlerini al
                feature_names = extractor.get_feature_names()[feature_type]

                print("Özellik çıkarma başarılı!")
            except Exception as e:
                print(f"Özellik çıkarma hatası: {str(e)}")
                sys.exit(1)
        else:
            print("Özellik çıkarma adımı atlandı. Özellik matrisleri mevcut olmalıdır.")
            # Bu durumda özelliklerin daha önce çıkarıldığı ve kaydedildiği varsayılır
            try:
                extractor = FeatureExtractor.load(os.path.join(models_dir, 'feature_extractor'))

                # Özellik matrislerini yükle veya yeniden oluştur
                train_features = extractor.transform(X_train)
                test_features = extractor.transform(X_test)

                if args.feature_method == 'tfidf':
                    X_train_features = train_features['tfidf']
                    X_test_features = test_features['tfidf']
                    feature_type = 'tfidf'
                elif args.feature_method == 'count':
                    X_train_features = train_features['count']
                    X_test_features = test_features['count']
                    feature_type = 'count'
                else:  # both
                    X_train_features = train_features['tfidf']
                    X_test_features = test_features['tfidf']
                    feature_type = 'tfidf'

                feature_names = extractor.get_feature_names()[feature_type]
            except Exception as e:
                print(f"Kaydedilmiş özellikleri yükleme hatası: {str(e)}")
                print("Devam etmek için özellik çıkarma adımını çalıştırmanız gerekiyor.")
                sys.exit(1)

        # 4. Model Eğitimi
        print("\n4. Model Eğitimi")
        print("-" * 80)

        models = {}

        if not args.skip_model_training:
            for model_name in models_to_run:
                print(f"\n4.{models_to_run.index(model_name) + 1}. {model_name.upper()} Model Eğitimi")

                model_path = os.path.join(models_dir, f"{model_name}_model.joblib")

                try:
                    if model_name == 'naive_bayes':
                        if args.optimize:
                            print("Naive Bayes model optimizasyonu yapılıyor...")
                            model, results = optimize_naive_bayes(
                                X_train_features, y_train,
                                variant='multinomial',
                                cv=args.cv
                            )

                            # Optimizasyon sonuçlarını kaydet
                            joblib.dump(results, os.path.join(metrics_dir, 'naive_bayes_optimization.joblib'))
                        else:
                            print("Varsayılan parametrelerle Naive Bayes model eğitimi yapılıyor...")
                            model = NaiveBayesModel(variant='multinomial')
                            model.fit(X_train_features, y_train)

                        models['Naive Bayes'] = model

                    elif model_name == 'decision_tree':
                        if args.optimize:
                            print("Decision Tree model optimizasyonu yapılıyor...")
                            model, results = optimize_decision_tree(
                                X_train_features, y_train,
                                cv=args.cv
                            )

                            # Optimizasyon sonuçlarını kaydet
                            joblib.dump(results, os.path.join(metrics_dir, 'decision_tree_optimization.joblib'))
                        else:
                            print("Varsayılan parametrelerle Decision Tree model eğitimi yapılıyor...")
                            model = DecisionTreeModel(
                                criterion='entropy',
                                max_depth=20,
                                min_samples_split=5,
                                random_state=42
                            )
                            model.fit(X_train_features, y_train, feature_names=feature_names)

                        models['Decision Tree'] = model

                    elif model_name == 'maxent':
                        if args.optimize:
                            print("Maxent (Logistic Regression) model optimizasyonu yapılıyor...")
                            model, results = optimize_maxent(
                                X_train_features, y_train,
                                cv=args.cv
                            )

                            # Optimizasyon sonuçlarını kaydet
                            joblib.dump(results, os.path.join(metrics_dir, 'maxent_optimization.joblib'))
                        else:
                            print("Varsayılan parametrelerle Maxent model eğitimi yapılıyor...")
                            model = MaxentModel(
                                C=1.0,
                                penalty='l2',
                                solver='lbfgs',
                                max_iter=1000,
                                random_state=42
                            )
                            model.fit(X_train_features, y_train, feature_names=feature_names)

                        models['Maxent'] = model

                    elif model_name == 'svm':
                        if args.optimize:
                            print("SVM model optimizasyonu yapılıyor...")
                            model, results = optimize_svm(
                                X_train_features, y_train,
                                variant='linear',
                                cv=args.cv
                            )

                            # Optimizasyon sonuçlarını kaydet
                            joblib.dump(results, os.path.join(metrics_dir, 'svm_optimization.joblib'))
                        else:
                            print("Varsayılan parametrelerle SVM model eğitimi yapılıyor...")
                            model = SVMModel(
                                variant='linear',
                                C=1.0,
                                penalty='l2',
                                dual=True,
                                random_state=42
                            )
                            model.fit(X_train_features, y_train, feature_names=feature_names)

                        models['SVM'] = model

                    elif model_name == 'random_forest':
                        if args.optimize:
                            print("Random Forest model optimizasyonu yapılıyor...")
                            model, results = optimize_random_forest(
                                X_train_features, y_train,
                                cv=args.cv
                            )

                            # Optimizasyon sonuçlarını kaydet
                            joblib.dump(results, os.path.join(metrics_dir, 'random_forest_optimization.joblib'))
                        else:
                            print("Varsayılan parametrelerle Random Forest model eğitimi yapılıyor...")
                            model = RandomForestModel(
                                n_estimators=200,
                                max_depth=20,
                                min_samples_split=5,
                                random_state=42
                            )
                            model.fit(X_train_features, y_train, feature_names=feature_names)

                        models['Random Forest'] = model

                    # Modeli kaydet
                    model.save(model_path)
                    print(f"{model_name} model eğitimi başarılı! Model kaydedildi: {model_path}")

                except Exception as e:
                    print(f"{model_name} model eğitimi hatası: {str(e)}")
                    continue
        else:
            print("Model eğitim adımı atlandı. Eğitilmiş modeller yükleniyor...")

            # Modelleri yükle
            for model_name in models_to_run:
                model_path = os.path.join(models_dir, f"{model_name}_model.joblib")

                try:
                    if model_name == 'naive_bayes':
                        model = NaiveBayesModel.load(model_path)
                        models['Naive Bayes'] = model

                    elif model_name == 'decision_tree':
                        model = DecisionTreeModel.load(model_path)
                        models['Decision Tree'] = model

                    elif model_name == 'maxent':
                        model = MaxentModel.load(model_path)
                        models['Maxent'] = model

                    elif model_name == 'svm':
                        model = SVMModel.load(model_path)
                        models['SVM'] = model

                    elif model_name == 'random_forest':
                        model = RandomForestModel.load(model_path)
                        models['Random Forest'] = model

                    print(f"{model_name} model yüklendi.")
                except Exception as e:
                    print(f"{model_name} model yükleme hatası: {str(e)}")
                    continue

        # 5. Model Değerlendirme
        print("\n5. Model Değerlendirme")
        print("-" * 80)

        if not args.skip_evaluation and models:
            try:
                # ModelEvaluator'ı başlat
                evaluator = ModelEvaluator(output_dir=results_dir)

                # Modelleri karşılaştır
                comparison_df = evaluator.compare_models(
                    models, X_test_features, y_test,
                    X_train=X_train_features, y_train=y_train,
                    feature_names=feature_names
                )

                # Sonuçları kaydet
                evaluator.save_results()

                print("Model değerlendirme başarılı!")
            except Exception as e:
                print(f"Model değerlendirme hatası: {str(e)}")
        else:
            print("Model değerlendirme adımı atlandı veya eğitilmiş model bulunamadı.")

        # 6. Görselleştirmeler
        print("\n6. Görselleştirmeler")
        print("-" * 80)

        if not args.skip_visualization:
            try:
                # ModelEvaluator görselleştirmeleri
                if 'evaluator' in locals():
                    evaluator.generate_all_visualizations()

                # Ek görselleştirmeler
                visualizer = Visualizer(output_dir=viz_dir)

                # Etiket dağılımını görselleştir
                visualizer.plot_label_distribution(y_train, y_test)

                # Doküman uzunluğu dağılımını görselleştir
                visualizer.plot_document_length_distribution(X_train_raw, label='Eğitim Verisi')
                visualizer.plot_document_length_distribution(X_test_raw, label='Test Verisi')

                # İşlenmiş metinler varsa kelime frekanslarını görselleştir
                if not args.skip_preprocessing:
                    processed_texts = processed_train_df['processed_text']
                    visualizer.plot_word_frequency(processed_texts, top_n=20)
                    visualizer.plot_word_frequency_by_class(processed_texts, y_train, top_n=10)

                # Öznitelik dağılımını görselleştir
                if 'feature_names' in locals():
                    visualizer.plot_feature_distribution(X_train_features, feature_names, top_n=20)

                # t-SNE ile boyut indirgeme (eğer veri boyutu uygunsa)
                try:
                    if X_train_features.shape[0] > 1000:
                        # Rastgele örnekleme yap
                        from sklearn.model_selection import train_test_split
                        X_sample, _, y_sample, _ = train_test_split(
                            X_train_features, y_train, train_size=1000, random_state=42, stratify=y_train
                        )
                    else:
                        X_sample, y_sample = X_train_features, y_train

                    visualizer.plot_dimensionality_reduction(X_sample, y_sample, method='tsne', n_components=2)
                except Exception as e:
                    print(f"t-SNE görselleştirme hatası: {str(e)}")

                # Veri seti bilgileri ve model sonuçları
                if 'evaluator' in locals() and hasattr(evaluator, 'model_results'):
                    data_info = {
                        'train_size': X_train.shape[0],
                        'test_size': X_test.shape[0],
                        'n_features': X_train_features.shape[1] if hasattr(X_train_features, 'shape') else 'N/A',
                        'n_classes': len(np.unique(y_train))
                    }

                    # Özet rapor oluştur
                    visualizer.generate_summary_report(data_info, evaluator.model_results)

                print("Görselleştirmeler başarılı!")
            except Exception as e:
                print(f"Görselleştirme hatası: {str(e)}")
        else:
            print("Görselleştirme adımı atlandı.")

        # Bitiş zamanı ve toplam süre
        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "=" * 80)
        print(f"İşlem tamamlandı! Toplam süre: {total_time:.2f} saniye")
        print("=" * 80)

    finally:
        # JVM'yi kapat
        try:
            close_jvm()
            print("JVM kapatıldı.")
        except:
            pass


if __name__ == "__main__":
    main()