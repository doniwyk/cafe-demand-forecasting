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
    FEATURE_COLUMNS_DAILY,
    get_feature_columns,
    REBRANDING_DATE,
    INDONESIAN_HOLIDAYS,
    RAMADAN_RANGES,
    PROCESSED_DIR,
    BOM_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
)
from src.utils.gpu import get_xgboost_params, is_gpu_available
from src.models.features import add_calendar_features, create_features
from src.models.forecaster import (
    load_and_prep_data,
    get_min_train_records,
    train_and_predict,
    train_models,
    load_models,
    predict,
    generate_future_features,
    FREQ_MAP,
)
from src.models.raw_materials import RawMaterialProcessor
from src.data.merger import translate_indonesian_to_english, clean_numeric_columns
from src.data.cleaner import SalesDataCleaner, PACKAGE_MAP
from src.evaluation.metrics import (
    weighted_mape,
    volume_accuracy,
    compute_metrics,
    classify_abc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weekly_sales_df():
    dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
    items = ["Espresso", "Black"]
    rows = []
    for d in dates:
        for item in items:
            rows.append({"Date": d, "Item": item, "Quantity_Sold": np.random.randint(1, 20)})
    return pd.DataFrame(rows)


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
        assert len(FEATURE_COLUMNS_DAILY) > 0

    def test_get_feature_columns_weekly(self):
        cols = get_feature_columns("weekly")
        assert cols == FEATURE_COLUMNS

    def test_get_feature_columns_daily(self):
        cols = get_feature_columns("daily")
        assert cols == FEATURE_COLUMNS_DAILY

    def test_daily_has_daily_lags(self):
        assert "Lag_7" in FEATURE_COLUMNS_DAILY
        assert "Lag_14" in FEATURE_COLUMNS_DAILY
        assert "Lag_1" in FEATURE_COLUMNS_DAILY

    def test_weekly_has_weekly_lags(self):
        assert "Lag_1" in FEATURE_COLUMNS
        assert "Lag_2" in FEATURE_COLUMNS
        assert "Lag_4" in FEATURE_COLUMNS

    def test_freq_map(self):
        assert FREQ_MAP["daily"] == "D"
        assert FREQ_MAP["weekly"] == "W-MON"

    def test_paths_exist(self):
        assert BOM_DIR.exists()
        assert MODELS_DIR.exists()

    def test_rebranding_date_is_string(self):
        assert isinstance(REBRANDING_DATE, str)
        pd.to_datetime(REBRANDING_DATE)

    def test_holidays_are_valid_dates(self):
        for h in INDONESIAN_HOLIDAYS:
            pd.to_datetime(h)

    def test_ramadan_ranges_are_valid(self):
        for start, end in RAMADAN_RANGES:
            s = pd.to_datetime(start)
            e = pd.to_datetime(end)
            assert s <= e

    def test_get_min_train_records(self):
        assert get_min_train_records("daily") == 180
        assert get_min_train_records("weekly") == 40


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
    def test_add_calendar_features(self):
        df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=10, freq="D"), "Quantity_Sold": [5] * 10})
        result = add_calendar_features(df)
        for col in ["Month", "Week", "Year", "DOY", "DOW", "Is_Holiday", "Is_Ramadan"]:
            assert col in result.columns

    def test_create_features_weekly(self, weekly_sales_df):
        result = create_features(weekly_sales_df, frequency="weekly")
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    def test_create_features_daily(self, daily_sales_df):
        result = create_features(daily_sales_df, frequency="daily")
        for col in FEATURE_COLUMNS_DAILY:
            assert col in result.columns

    def test_create_features_no_nans_in_features(self, weekly_sales_df):
        result = create_features(weekly_sales_df, frequency="weekly")
        for col in FEATURE_COLUMNS:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_lag_values_weekly(self, weekly_sales_df):
        result = create_features(weekly_sales_df, frequency="weekly")
        assert "Lag_1" in result.columns
        assert "Lag_2" in result.columns
        assert "Lag_4" in result.columns

    def test_lag_values_daily(self, daily_sales_df):
        result = create_features(daily_sales_df, frequency="daily")
        assert "Lag_1" in result.columns
        assert "Lag_7" in result.columns
        assert "Lag_14" in result.columns

    def test_holiday_flag_set(self):
        holiday = INDONESIAN_HOLIDAYS[0]
        df = pd.DataFrame({"Date": pd.to_datetime([holiday]), "Quantity_Sold": [1]})
        result = add_calendar_features(df)
        assert result["Is_Holiday"].iloc[0] == 1


# ---------------------------------------------------------------------------
# forecaster tests
# ---------------------------------------------------------------------------

class TestForecaster:
    def test_load_and_prep_data_weekly(self, tiny_sales_csv, capsys):
        df = load_and_prep_data(tiny_sales_csv, frequency="weekly")
        assert "Date" in df.columns
        assert "Quantity_Sold" in df.columns
        assert "Item" in df.columns
        assert len(df) > 0

    def test_load_and_prep_data_daily(self, tiny_sales_csv):
        df = load_and_prep_data(tiny_sales_csv, frequency="daily")
        assert "Quantity_Sold" in df.columns

    def test_train_and_predict_weekly(self, weekly_sales_df):
        features = create_features(weekly_sales_df, frequency="weekly")
        result = train_and_predict(features, n_test_periods=4, frequency="weekly")
        assert "Predicted" in result.columns
        assert "Quantity_Sold" in result.columns
        assert (result["Predicted"] >= 0).all()

    def test_train_and_predict_daily(self, daily_sales_df):
        features = create_features(daily_sales_df, frequency="daily")
        result = train_and_predict(features, n_test_periods=2, frequency="daily")
        assert "Predicted" in result.columns
        assert (result["Predicted"] >= 0).all()

    def test_train_models_and_predict(self, weekly_sales_df, tmp_path):
        features = create_features(weekly_sales_df, frequency="weekly")
        model_dir = tmp_path / "models"
        item_models, global_model, dow_factors = train_models(features, model_dir, frequency="weekly")

        assert isinstance(item_models, dict)
        assert len(dow_factors) > 0
        assert (model_dir / "global_model.pkl").exists()
        assert (model_dir / "item_models.pkl").exists()
        assert (model_dir / "dow_factors.json").exists()

        loaded = load_models(model_dir)
        assert isinstance(loaded, tuple)
        assert len(loaded) == 3

    def test_generate_future_features_weekly(self, weekly_sales_df):
        features = create_features(weekly_sales_df, frequency="weekly")
        future = generate_future_features(features, future_weeks=4, frequency="weekly")
        assert len(future) > 0
        assert "Quantity_Sold" in future.columns

    def test_generate_future_features_daily(self, daily_sales_df):
        features = create_features(daily_sales_df, frequency="daily")
        future = generate_future_features(features, future_weeks=4, frequency="daily")
        assert len(future) > 0

    def test_predict_with_loaded_models(self, weekly_sales_df, tmp_path):
        features = create_features(weekly_sales_df, frequency="weekly")
        model_dir = tmp_path / "models"
        item_models, global_model, dow_factors = train_models(features, model_dir, frequency="weekly")

        future = generate_future_features(features, future_weeks=2, frequency="weekly")
        preds = predict(future, model_dir=model_dir, frequency="weekly")
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

    def test_package_map_structure(self):
        for key, components in PACKAGE_MAP.items():
            assert isinstance(components, list)
            assert all(isinstance(c, tuple) and len(c) == 2 for c in components)


# ---------------------------------------------------------------------------
# evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_weighted_mape(self):
        y_true = pd.Series([10, 20, 30])
        y_pred = pd.Series([11, 19, 33])
        result = weighted_mape(y_true, y_pred)
        assert 0 < result < 100

    def test_volume_accuracy(self):
        y_true = pd.Series([100, 200])
        y_pred = pd.Series([100, 200])
        result = volume_accuracy(y_true, y_pred)
        assert result == 100.0

    def test_volume_accuracy_off_by_10pct(self):
        y_true = pd.Series([100])
        y_pred = pd.Series([110])
        result = volume_accuracy(y_true, y_pred)
        assert abs(result - 90.0) < 1

    def test_compute_metrics(self):
        y_true = pd.Series([10, 20, 30, 40])
        y_pred = pd.Series([11, 19, 31, 38])
        m = compute_metrics(y_true, y_pred)
        assert "r2" in m
        assert "wmape" in m
        assert "mae" in m
        assert "volume_accuracy" in m
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
