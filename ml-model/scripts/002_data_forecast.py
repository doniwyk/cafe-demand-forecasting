import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD & AGGREGATE TO WEEKLY
# ==========================================
def load_and_prep_data(filepath):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[~df['Item'].str.strip().str.lower().str.startswith('add')]
    df_weekly = df.set_index('Date').groupby('Item').resample('W-MON')['Quantity_Sold'].sum().reset_index()
    return df_weekly

# ==========================================
# 2. CALENDAR & REBRANDING FEATURES
# ==========================================
def add_calendar_features(df):
    data = df.copy()
    data = data.sort_values('Date').reset_index(drop=True)

    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
    data['Year'] = data['Date'].dt.year
    data['DOY'] = data['Date'].dt.dayofyear
    data['DOW'] = data['Date'].dt.weekday

    # Yearly seasonality
    data['Sin_Week'] = np.sin(2 * np.pi * data['Week'] / 52)
    data['Cos_Week'] = np.cos(2 * np.pi * data['Week'] / 52)

    # Trend
    data['Weeks_Since_Start'] = (data['Date'] - data['Date'].min()).dt.days // 7

    # Payday & weekend before payday
    data['Is_Payday_Week'] = data['Date'].dt.day.apply(lambda x: 1 if x <= 7 or x >= 25 else 0)
    data['Is_Weekend_Before_Payday'] = ((data['DOW'] >= 4) & (data['Date'].dt.day >= 25)).astype(int)

    # Major Indonesian Holidays 2024–2026
    holidays = [
        '2024-01-01','2024-04-10','2024-04-11','2024-04-12','2024-05-01','2024-05-23',
        '2024-06-01','2024-06-17','2024-12-25',
        '2025-01-01','2025-03-01','2025-03-30','2025-03-31','2025-05-01','2025-05-12',
        '2025-06-01','2025-06-07','2025-12-25',
        '2026-01-01','2026-02-17','2026-03-09','2026-05-01','2026-05-21','2026-12-25'
    ]
    data['Is_Holiday'] = data['Date'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)

    # Ramadan (approx)
    data['Is_Ramadan'] = 0
    for year in [2024, 2025, 2026]:
        mask = (data['Date'] >= f'{year}-02-28') & (data['Date'] <= f'{year}-04-10')
        data.loc[mask, 'Is_Ramadan'] = 1

    # Rebranding
    data['Is_Post_Rebranding'] = (data['Date'] >= '2025-05-01').astype(int)
    data['Weeks_Since_Rebrand'] = ((data['Date'] - pd.to_datetime('2025-05-01')).dt.days / 7).clip(lower=0)

    return data

# ==========================================
# 3. ADVANCED PER-ITEM FEATURES
# ==========================================
def create_features(df):
    data = add_calendar_features(df)
    data = data.sort_values(['Item', 'Date'])
    data['Item_Code'] = data['Item'].astype('category').cat.codes

    for item in data['Item'].unique():
        mask = data['Item'] == item
        g = data.loc[mask, 'Quantity_Sold']

        # Lags
        data.loc[mask, 'Lag_1'] = g.shift(1)
        data.loc[mask, 'Lag_2'] = g.shift(2)
        data.loc[mask, 'Lag_4'] = g.shift(4)

        # Rolling
        shifted = g.shift(1)
        data.loc[mask, 'Roll_Mean_4']  = shifted.rolling(4, min_periods=1).mean()
        data.loc[mask, 'Roll_Mean_12'] = shifted.rolling(12, min_periods=1).mean()
        data.loc[mask, 'Roll_Std_4']   = shifted.rolling(4, min_periods=1).std()
        data.loc[mask, 'Roll_Q95_4']   = shifted.rolling(4, min_periods=1).quantile(0.95)

        # EWMA
        data.loc[mask, 'EWMA_4']  = shifted.ewm(span=4, adjust=False).mean()
        data.loc[mask, 'EWMA_12'] = shifted.ewm(span=12, adjust=False).mean()

        # Momentum
        data.loc[mask, 'Diff_1']  = g.diff(1)
        data.loc[mask, 'Accel_2'] = data.loc[mask, 'Diff_1'].diff(1)

        # Recent vs older trend
        recent = shifted.rolling(4).mean()
        older  = g.shift(5).rolling(8).mean()
        data.loc[mask, 'Recent_vs_Old_Trend'] = (recent / (older + 1)).clip(0, 10)

        # Post-rebrand surge ratio (the Black Ice killer)
        pre_mean  = g[data.loc[mask, 'Date'] < '2025-05-01'].mean()
        post_mean = g[data.loc[mask, 'Date'] >= '2025-05-01'].mean()
        surge = (post_mean / (pre_mean + 1)) if pd.notna(post_mean) else 1.0
        data.loc[mask, 'Post_Rebrand_Surge_Ratio'] = surge

    data = data.fillna(0)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

# ==========================================
# 4. PER-ITEM TRAINING + FIXED DOW ADJUSTMENT
# ==========================================
def train_and_predict(df_features):
    split_date = df_features['Date'].max() - pd.Timedelta(weeks=12)
    train = df_features[df_features['Date'] < split_date].copy()
    test  = df_features[df_features['Date'] >= split_date].copy()

    print(f"Training: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"Testing : {test['Date'].min().date()} → {test['Date'].max().date()}")

    # FIXED DOW FACTOR
    dow_pattern = train.groupby(['Item', train['Date'].dt.weekday])['Quantity_Sold'].mean().reset_index()
    item_avg = train.groupby('Item')['Quantity_Sold'].mean().reset_index().rename(columns={'Quantity_Sold': 'item_avg'})
    dow_pattern = dow_pattern.merge(item_avg, on='Item')
    dow_pattern['dow_factor'] = dow_pattern['Quantity_Sold'] / dow_pattern['item_avg']
    dow_factor_dict = dow_pattern.pivot(index='Item', columns='Date', values='dow_factor')
    dow_factor_dict = dow_factor_dict.fillna(1.0).to_dict('index')   # {item: {0:1.1, 1:0.9, ...}}

    # Feature list
    features = [
        'Item_Code','Month','Week','Year','DOY','DOW','Sin_Week','Cos_Week','Weeks_Since_Start',
        'Is_Payday_Week','Is_Weekend_Before_Payday','Is_Holiday','Is_Ramadan','Is_Post_Rebranding',
        'Weeks_Since_Rebrand','Post_Rebrand_Surge_Ratio','Recent_vs_Old_Trend',
        'Lag_1','Lag_2','Lag_4','Roll_Mean_4','Roll_Mean_12','Roll_Std_4','Roll_Q95_4',
        'EWMA_4','EWMA_12','Diff_1','Accel_2'
    ]

    # Global fallback
    print("Training global fallback model...")
    global_model = XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=1.5,
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
    )
    global_model.fit(train[features], train['Quantity_Sold'])

    # Per-item models
    print("Training per-item models...")
    predictions = []
    for item in test['Item'].unique():
        train_item = train[train['Item'] == item]
        test_item  = test[test['Item'] == item].copy()

        if len(train_item) >= 40:
            model = XGBRegressor(
                objective='reg:tweedie', tweedie_variance_power=1.5,
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=42
            )
            model.fit(train_item[features], train_item['Quantity_Sold'], verbose=False)
            pred = model.predict(test_item[features])
        else:
            pred = global_model.predict(test_item[features])

        test_item['Raw_Pred'] = np.maximum(0, pred)
        test_item['DOW'] = test_item['Date'].dt.weekday

        # Apply correct DOW factor
        factors = dow_factor_dict.get(item, {i: 1.0 for i in range(7)})
        test_item['dow_factor'] = test_item['DOW'].map(factors).fillna(1.0)
        test_item['Predicted'] = (test_item['Raw_Pred'] * test_item['dow_factor']).round(0)
        test_item['Predicted'] = np.maximum(0, test_item['Predicted'])

        predictions.append(test_item)

    return pd.concat(predictions).sort_values(['Item', 'Date'])

# ==========================================
# 5. ABC ANALYSIS
# ==========================================
def generate_abc_analysis(df):
    y_true = df['Quantity_Sold']
    y_pred = df['Predicted']

    print("\n" + "="*90)
    print("FINAL MODEL PERFORMANCE – BLACK ICE FIXED")
    print("="*90)
    print(f"Global R²        : {r2_score(y_true, y_pred):.4f}")
    print(f"Global wMAPE     : {100 * abs(y_true - y_pred).sum() / y_true.sum():.2f}%")
    print(f"Global MAE       : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"Total Vol Acc    : {100 * (1 - abs(y_true.sum() - y_pred.sum()) / y_true.sum()):.2f}%")

    # ABC classification
    item_vol = df.groupby('Item')['Quantity_Sold'].sum().sort_values(ascending=False)
    total = item_vol.sum()
    item_vol = pd.DataFrame({'Vol': item_vol,
                             'Cum': item_vol.cumsum(),
                             'Pct': item_vol.cumsum() / total})
    item_vol['Class'] = item_vol['Pct'].apply(lambda x: 'A' if x <= 0.70 else ('B' if x <= 0.90 else 'C'))
    df['Class'] = df['Item'].map(item_vol['Class'])

    print("\nABC BY CLASS")
    print("-"*60)
    for c in ['A', 'B', 'C']:
        sub = df[df['Class'] == c]
        if len(sub) == 0: continue
        wmape = 100 * abs(sub['Quantity_Sold'] - sub['Predicted']).sum() / sub['Quantity_Sold'].sum()
        volacc = 100 * (1 - abs(sub['Quantity_Sold'].sum() - sub['Predicted'].sum()) / sub['Quantity_Sold'].sum())
        print(f"{c}-Class | Items: {sub['Item'].nunique():2d} | wMAPE: {wmape:5.1f}% | Vol.Acc: {volacc:5.1f}%")

    print("\nTOP 10 CLASS A ITEMS")
    print("-"*60)
    top = df[df['Class'] == 'A'].groupby('Item')[['Quantity_Sold','Predicted']].sum()
    top = top.sort_values('Quantity_Sold', ascending=False).head(10)
    top['Acc%'] = (100 * (1 - abs(top['Predicted'] - top['Quantity_Sold']) / top['Quantity_Sold'])).round(1)
    print(top[['Quantity_Sold','Predicted','Acc%']])

# ==========================================
# 6. RUN
# ==========================================
if __name__ == "__main__":
    df_raw = load_and_prep_data('../data/processed/daily_item_sales.csv')
    df_feat = create_features(df_raw)
    test_pred = train_and_predict(df_feat)
    generate_abc_analysis(test_pred)