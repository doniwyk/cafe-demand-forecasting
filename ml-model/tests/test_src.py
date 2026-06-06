import pandas as pd
import numpy as np
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    BOM_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
)
from src.utils.gpu import get_xgboost_params, is_gpu_available
from src.models.features import create_features
from src.models.forecaster import (
    load_and_prep_data,
    train_and_predict,
    train_models,
    load_models,
    predict,
    generate_future_features,
)
from src.models.raw_materials import RawMaterialProcessor
from src.data.merger import translate_indonesian_to_english, clean_numeric_columns
from src.data.cleaner import SalesDataCleaner
from src.evaluation.metrics import (
    weighted_mape,
    compute_metrics,
    classify_abc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_sales_df():
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    items = ["Espresso", "Black"]
    rows = []
    for d in dates:
        for item in items:
            rows.append({"Date": d, "Item": item, "Quantity_Sold": np.random.randint(0, 10)})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_bom_csv(tmp_path):
    path = tmp_path / "menu_bom.csv"
    path.write_text("Tipe,Item,Bahan,Qty,Unit\nbeverage,Espresso,Beans Arabika,19.0,gr\nbeverage,Black,Beans Arabika,19.0,gr\n")
    return path


@pytest.fixture
def tiny_condiment_csv(tmp_path):
    path = tmp_path / "condiment_bom.csv"
    path.write_text("Condiment,Condiment_Qty,Condiment_Unit,Sub_Ingredient,Qty_per_condiment_unit,Sub_Unit\nBSJ - Creamer,1000.0,gr,Creamer Bubuk,1000.0,gr\nBSJ - Creamer,1000.0,gr,Cleo Galon,2000.0,ml\n")
    return path


@pytest.fixture
def tiny_sales_csv(tmp_path):
    path = tmp_path / "sales_data_cleaned.csv"
    path.write_text(
        "Date,Item,Quantity\n2025-01-06,Espresso,5\n2025-01-06,Black,3\n"
        "2025-01-07,Espresso,7\n2025-01-07,Black,4\n"
    )
    return path


# ---------------------------------------------------------------------------
# config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_feature_columns_not_empty(self):
        assert len(FEATURE_COLUMNS) > 0

    def test_daily_has_daily_lags(self):
        assert "Lag_1" in FEATURE_COLUMNS
        assert "Roll_Mean_7" in FEATURE_COLUMNS
        assert "Roll_Mean_28" in FEATURE_COLUMNS

    def test_paths_exist(self):
        assert BOM_DIR.exists()
        assert MODELS_DIR.exists()


# ---------------------------------------------------------------------------
# gpu tests
# ---------------------------------------------------------------------------

class TestGPU:
    def test_gpu_detection_runs(self):
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_xgboost_params(self):
        params = get_xgboost_params()
        assert "tree_method" in params
        assert isinstance(params["n_jobs"], int)


# ---------------------------------------------------------------------------
# features tests
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_create_features_daily(self, daily_sales_df):
        result = create_features(daily_sales_df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    def test_create_features_no_nans_in_features(self, daily_sales_df):
        result = create_features(daily_sales_df)
        for col in FEATURE_COLUMNS:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_lag_values_daily(self, daily_sales_df):
        result = create_features(daily_sales_df)
        assert "Lag_1" in result.columns
        assert "Roll_Mean_7" in result.columns
        assert "EWMA_28" in result.columns


# ---------------------------------------------------------------------------
# forecaster tests
# ---------------------------------------------------------------------------

class TestForecaster:
    def test_load_and_prep_data_daily(self, tiny_sales_csv):
        df = load_and_prep_data(tiny_sales_csv)
        assert "Quantity_Sold" in df.columns
        assert "Date" in df.columns
        assert "Item" in df.columns
        assert len(df) > 0

    def test_train_and_predict_daily(self, daily_sales_df):
        features = create_features(daily_sales_df)
        result = train_and_predict(features, n_test_periods=2)
        assert "Predicted" in result.columns
        assert (result["Predicted"] >= 0).all()

    def test_train_models_and_predict(self, daily_sales_df, tmp_path):
        features = create_features(daily_sales_df)
        model_dir = tmp_path / "models"
        item_models, global_model, dow_factors = train_models(features, model_dir)

        assert isinstance(item_models, dict)
        assert len(dow_factors) > 0
        assert (model_dir / "global_model.pkl").exists()
        assert (model_dir / "item_models.pkl").exists()
        assert (model_dir / "dow_factors.json").exists()

        loaded = load_models(model_dir)
        assert isinstance(loaded, tuple)
        assert len(loaded) == 3

    def test_generate_future_features_daily(self, daily_sales_df):
        features = create_features(daily_sales_df)
        future = generate_future_features(features, future_weeks=4)
        assert len(future) > 0
        assert "Quantity_Sold" in future.columns

    def test_predict_with_loaded_models(self, daily_sales_df, tmp_path):
        features = create_features(daily_sales_df)
        model_dir = tmp_path / "models"
        item_models, global_model, dow_factors = train_models(features, model_dir)

        future = generate_future_features(features, future_weeks=2)
        preds = predict(future, model_dir=model_dir)
        assert "Predicted" in preds.columns
        assert (preds["Predicted"] >= 0).all()


# ---------------------------------------------------------------------------
# raw_materials tests
# ---------------------------------------------------------------------------

class TestRawMaterials:
    def test_expand_condiment(self, tiny_bom_csv, tiny_condiment_csv, tiny_sales_csv):
        processor = RawMaterialProcessor(tiny_sales_csv, tiny_bom_csv, tiny_condiment_csv)
        result = processor._expand_condiment("BSJ - Creamer", 100.0, "gr")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_normalize_material_name(self, tiny_bom_csv, tiny_condiment_csv, tiny_sales_csv):
        processor = RawMaterialProcessor(tiny_sales_csv, tiny_bom_csv, tiny_condiment_csv)
        assert processor._normalize_material_name("cleo galon") == "Cleo Galon"
        assert processor._normalize_material_name("skm") == "SKM"

    def test_cache_used(self, tiny_bom_csv, tiny_condiment_csv, tiny_sales_csv):
        processor = RawMaterialProcessor(tiny_sales_csv, tiny_bom_csv, tiny_condiment_csv)
        r1 = processor._expand_condiment("BSJ - Creamer", 100.0, "gr")
        r2 = processor._expand_condiment("BSJ - Creamer", 100.0, "gr")
        assert r1 == r2
        assert len(processor.expansion_cache) == 1


# ---------------------------------------------------------------------------
# data pipeline tests
# ---------------------------------------------------------------------------

class TestDataPipeline:
    def test_translate_indonesian_columns(self):
        df = pd.DataFrame({"Tanggal": ["01/01/25"], "Barang": ["Kopi"]})
        result = translate_indonesian_to_english(df)
        assert "Date" in result.columns
        assert "Item" in result.columns

    def test_clean_numeric_columns(self):
        df = pd.DataFrame({"Quantity": ["10", "abc", "5.5"], "Price": ["1000", "2000", "3000"]})
        result = clean_numeric_columns(df)
        assert pd.isna(result["Quantity"].iloc[1])
        assert result["Quantity"].iloc[0] == 10.0

    def test_cleaner_identifies_discontinued(self, tiny_bom_csv, tiny_sales_csv):
        cleaner = SalesDataCleaner(tiny_sales_csv, tiny_bom_csv)
        assert len(cleaner.active_items) == 2  # Espresso, Black


# ---------------------------------------------------------------------------
# evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_weighted_mape(self):
        y_true = pd.Series([10, 20, 30])
        y_pred = pd.Series([11, 19, 33])
        result = weighted_mape(y_true, y_pred)
        assert 0 < result < 100

    def test_compute_metrics(self):
        y_true = pd.Series([10, 20, 30, 40])
        y_pred = pd.Series([11, 19, 31, 38])
        m = compute_metrics(y_true, y_pred)
        assert "r2" in m
        assert "wmape" in m
        assert "mae" in m
        assert 0 <= m["r2"] <= 1
        assert m["mae"] >= 0

    def test_classify_abc(self):
        df = pd.DataFrame({
            "Item": ["A"] * 70 + ["B"] * 20 + ["C"] * 10,
            "Quantity_Sold": [10] * 70 + [5] * 20 + [1] * 10,
        })
        result = classify_abc(df)
        assert "Class" in result.columns
        assert set(result["Class"].unique()).issubset({"A", "B", "C"})

    def test_classify_abc_class_distribution(self):
        df = pd.DataFrame({
            "Item": ["A"] * 70 + ["B"] * 20 + ["C"] * 10,
            "Quantity_Sold": [10] * 70 + [5] * 20 + [1] * 10,
        })
        result = classify_abc(df)
        assert result.loc["A", "Class"] in ("A", "B")  # right at boundary
