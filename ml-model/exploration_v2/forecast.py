"""
forecast.py — production daily demand forecasting for cafe supply planning.

Best configuration from exploration_v2:
  - XGBoost Poisson (count:poisson) — MAE 0.78
  - Frozen rolling/lag features (anchored on last actuals)
  - Updated recency daily (Days_Since_Last_Sale, Sales_Last_7D)
  - Weekend recency reset (Fri/Sat/Sun: act like item sold recently)
  - Newsvendor buffer: z × error_std per item, driven by cost ratio

Usage:
  python forecast.py [--days 7] [--margin 3 --spoilage 1]
"""
import os, sys, warnings, argparse, pickle
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import xgboost as xgb

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / ".." / "data" / "processed" / "sales_forecasting"
TABLES_DIR = ROOT / "tables"
MODELS_DIR = ROOT / "models"
DAILY_CSV = os.path.join(DATA_DIR, "daily_item_sales.csv")
MODELS_DIR = str(MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
MAX_WINDOW = 60

# --- Feature definition ---
TEMPORAL = ["DOW","Is_Weekend","Month","Year","WeekOfYear","DayOfMonth",
            "Quarter","MonthStart","MonthEnd","Is_Holiday_Season",
            "WeekOfMonth","DaysFromStart","DOW_Sin","DOW_Cos","Month_Sin","Month_Cos"]
LIFECYCLE = ["Days_Since_First_Sale","Item_Rank","Item_Rank_Pct"]
RECENCY = ["Days_Since_Last_Sale","Sales_Last_7D"]
ROLLING = ["Roll_Mean_7","Roll_Mean_14","Roll_Mean_28","Roll_Std_7",
           "EWMA_7","EWMA_28","Trend_7_28","WoW_Change"]
LAGS = ["Lag_1","Lag_7","Lag_14","Lag_28"]
CROSS = ["Day_Total_Qty","Day_Total_Items_Sold","Day_Total_Beverage",
         "Day_Total_Food","Day_Total_Qty_7D"]
CAT = ["Is_Beverage","Is_Food"]

# --- hus_db connection ---
HUS_DB_URL = os.getenv("HUS_DB_URL", "postgresql://user:password@localhost:5432/hus_db")


def pull_fresh_sales():
    """Pull latest sales from hus_db since last CSV date."""
    try:
        import psycopg2
    except ImportError:
        return None

    csv = pd.read_csv(DAILY_CSV)
    csv["Date_Only"] = pd.to_datetime(csv["Date_Only"])
    last_csv = csv["Date_Only"].max().date()
    cafe_items = set(csv["Item"].str.strip().unique())

    try:
        conn = psycopg2.connect(HUS_DB_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT DATE(o.created_at), oi.product_name_snapshot,
                   oi.variant_name_snapshot, SUM(oi.quantity)
            FROM order_items oi JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'PAID' AND o.created_at > %s
            GROUP BY 1,2,3 ORDER BY 1
        """, (last_csv.strftime("%Y-%m-%d"),))
        rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception:
        return None

    if not rows:
        return None

    FALLBACK = {"Kopi Susu Husgendam": "Kopi Susu Husgendam Ice",
                 "Cappucino": "Cappucino Ice"}
    SKIP = ["Add ","Filter","FIlter","V60","Harum Jasmine Tea",
            "Cookies Redvelvet","Lotus Cheesecake","Strawberry Cheesecake","Kopi Susu Bersemi"]

    matched = []
    for dt, pname, vname, qty in rows:
        pname = (pname or "").strip()
        vname = (vname or "").strip() if vname else ""
        combined = f"{pname} {vname}".strip() if vname else pname
        if any(combined.startswith(p) for p in SKIP):
            continue
        if combined in cafe_items: item = combined
        elif pname in cafe_items: item = pname
        elif pname in FALLBACK and FALLBACK[pname] in cafe_items: item = FALLBACK[pname]
        else: continue
        matched.append({"Date_Only": dt, "Item": item, "Quantity": int(qty)})

    if not matched:
        return None

    new = pd.DataFrame(matched)
    new["Date_Only"] = pd.to_datetime(new["Date_Only"])
    cat_map = csv[["Item","Category"]].drop_duplicates().set_index("Item")["Category"].to_dict()
    new["Category"] = new["Item"].map(cat_map)
    combined = pd.concat([csv, new], ignore_index=True).drop_duplicates(
        subset=["Date_Only","Item"], keep="last"
    ).sort_values(["Date_Only","Item"]).reset_index(drop=True)
    print(f"  hus_db: +{len(new)} rows → {len(combined)} total")
    return combined


def build_features(full):
    """Build all features from raw sales data (frozen-friendly)."""
    items = sorted(full["Item"].unique())
    dates = pd.date_range(full["Date_Only"].min(), full["Date_Only"].max())
    grid = pd.DataFrame([(d,i) for d in dates for i in items], columns=["Date_Only","Item"])
    full = grid.merge(full, on=["Date_Only","Item"], how="left")
    full["Quantity"] = full["Quantity"].fillna(0).astype(float)
    full["Category"] = full.groupby("Item")["Category"].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else (
            x.dropna().iloc[0] if x.dropna().shape[0] > 0 else "unknown"))
    full["Is_Sale"] = (full["Quantity"] > 0).astype(int)
    d = full["Date_Only"]

    full["DOW"] = d.dt.dayofweek
    full["Is_Weekend"] = (full["DOW"] >= 5).astype(int)
    full["Month"] = d.dt.month; full["Year"] = d.dt.year
    full["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    full["DayOfMonth"] = d.dt.day; full["Quarter"] = d.dt.quarter
    full["MonthStart"] = (full["DayOfMonth"]<=7).astype(int)
    full["MonthEnd"] = (full["DayOfMonth"]>=25).astype(int)
    full["Is_Holiday_Season"] = full["Month"].isin([12,1]).astype(int)
    full["WeekOfMonth"] = ((full["DayOfMonth"]-1)//7+1).astype(int)
    full["DaysFromStart"] = (d - d.min()).dt.days
    full["DOW_Sin"] = np.sin(2*np.pi*full["DOW"]/7)
    full["DOW_Cos"] = np.cos(2*np.pi*full["DOW"]/7)
    full["Month_Sin"] = np.sin(2*np.pi*full["Month"]/12)
    full["Month_Cos"] = np.cos(2*np.pi*full["Month"]/12)

    # Lifecycle
    fd = {}
    for item in items:
        s = full[(full["Item"]==item)&(full["Quantity"]>0)]
        if len(s) > 0: fd[item] = s["Date_Only"].min()
    full["Days_Since_First_Sale"] = full.apply(
        lambda r: (r["Date_Only"]-fd.get(r["Item"],r["Date_Only"])).days, axis=1)
    ranks = full.groupby("Item")["Quantity"].sum().rank(ascending=False)
    full["Item_Rank"] = full["Item"].map(ranks)
    full["Item_Rank_Pct"] = full["Item_Rank"]/ranks.max()

    # Item dummies
    dummies = pd.get_dummies(full["Item"], prefix="Item")
    for c in dummies.columns: full[c] = dummies[c].astype(int)
    full["Is_Beverage"] = (full["Category"]=="beverage").astype(int)
    full["Is_Food"] = (full["Category"]=="food").astype(int)

    # Recency + rolling + lags (per item, shift(1) no leakage)
    full = full.sort_values(["Item","Date_Only"]).reset_index(drop=True)
    for item in items:
        m = full["Item"]==item; idxs = full[m].index; q = full.loc[m,"Quantity"].values
        n = len(q)
        # Recency
        days_since = np.full(n, 999, dtype=int)
        s7d = np.zeros(n, dtype=int); last = None
        for i in range(n):
            if q[i] > 0: last = i
            if last is not None: days_since[i] = min(i-last, 999)
            s7d[i] = int(sum(1 for j in range(max(0,i-7),i) if q[j]>0))
        full.loc[m,"Days_Since_Last_Sale"] = days_since
        full.loc[m,"Sales_Last_7D"] = s7d
        # Rolling & lags
        qs = pd.Series(q); shifted = qs.shift(1)
        full.loc[m,"Lag_1"] = qs.shift(1).fillna(0).values
        full.loc[m,"Lag_7"] = qs.shift(7).fillna(0).values
        full.loc[m,"Lag_14"] = qs.shift(14).fillna(0).values
        full.loc[m,"Lag_28"] = qs.shift(28).fillna(0).values
        full.loc[m,"Roll_Mean_7"] = shifted.rolling(7,min_periods=1).mean().fillna(0).values
        full.loc[m,"Roll_Mean_14"] = shifted.rolling(14,min_periods=1).mean().fillna(0).values
        full.loc[m,"Roll_Mean_28"] = shifted.rolling(28,min_periods=1).mean().fillna(0).values
        full.loc[m,"Roll_Std_7"] = shifted.rolling(7,min_periods=2).std().fillna(0).values
        full.loc[m,"EWMA_7"] = shifted.ewm(span=7,min_periods=1).mean().fillna(0).values
        full.loc[m,"EWMA_28"] = shifted.ewm(span=28,min_periods=1).mean().fillna(0).values
        r7 = full.loc[m,"Roll_Mean_7"].values; r28 = full.loc[m,"Roll_Mean_28"].values
        full.loc[m,"Trend_7_28"] = np.clip((r7-r28)/(r28+0.1),-5,5)
        full.loc[m,"WoW_Change"] = np.clip(
            (pd.Series(r7)-pd.Series(r7).shift(7))/(pd.Series(r7).shift(7)+0.1),-5,5).fillna(0).values

    # Cross-item (shifted 1 day)
    daily = full.groupby("Date_Only").agg(
        Total_Qty=("Quantity","sum"), Total_Items_Sold=("Is_Sale","sum")).shift(1).fillna(0)
    daily["Total_Beverage"] = full[full["Category"]=="beverage"].groupby(
        "Date_Only")["Quantity"].sum().shift(1).fillna(0)
    daily["Total_Food"] = full[full["Category"]=="food"].groupby(
        "Date_Only")["Quantity"].sum().shift(1).fillna(0)
    daily["Total_Qty_7D"] = daily["Total_Qty"].rolling(7,min_periods=1).mean()
    for col in ["Total_Qty","Total_Items_Sold","Total_Beverage","Total_Food","Total_Qty_7D"]:
        full[f"Day_{col}"] = full["Date_Only"].map(daily[col]).fillna(0)

    return full


def get_feature_cols(full):
    item_d = [c for c in full.columns if c.startswith("Item_")
              and c not in ("Item_Rank","Item_Rank_Pct") and ".1" not in c]
    feats = TEMPORAL + LIFECYCLE + RECENCY + ROLLING + LAGS + CROSS + CAT + item_d
    return [c for c in feats if c in full.columns]


class Forecaster:
    """Recursive multi-day forecaster with frozen rolling + updated recency."""

    def __init__(self, items, meta):
        self.items = items
        self.meta = meta
        self.frozen_qty = {}   # never updated — for rolling/lag
        self.qty_history = {}  # updated daily — for recency

    def load_history(self, full, last_date):
        for item in self.items:
            grp = full[(full["Item"]==item)&(full["Date_Only"]<=last_date)]
            grp = grp.sort_values("Date_Only").tail(MAX_WINDOW)
            self.frozen_qty[item] = grp["Quantity"].values.astype(float)
            self.qty_history[item] = self.frozen_qty[item].copy()

    def update_recency(self, preds):
        for i, item in enumerate(self.items):
            self.qty_history[item] = np.append(self.qty_history[item], preds[i])
            if len(self.qty_history[item]) > MAX_WINDOW:
                self.qty_history[item] = self.qty_history[item][-MAX_WINDOW:]

    def _recency(self, qty):
        n = len(qty); d = 999
        for i in range(n-1,-1,-1):
            if qty[i] > 0: d = n-1-i; break
        return float(min(d,999)), int(sum(1 for q in qty[-7:] if q>0))

    def _rolling(self, qty):
        n = len(qty)
        if n == 0: return {f:0.0 for f in ["Lag_1","Lag_7","Lag_14","Lag_28",
            "Roll_Mean_7","Roll_Mean_14","Roll_Mean_28","Roll_Std_7",
            "EWMA_7","EWMA_28","Trend_7_28","WoW_Change"]}
        s = pd.Series(qty); sh = s.shift(1)
        return {
            "Lag_1": float(s.iloc[-1]) if n>=1 else 0, "Lag_7": float(s.iloc[-7]) if n>=7 else 0,
            "Lag_14": float(s.iloc[-14]) if n>=14 else 0, "Lag_28": float(s.iloc[-28]) if n>=28 else 0,
            "Roll_Mean_7": float(sh.rolling(7,min_periods=1).mean().iloc[-1]),
            "Roll_Mean_14": float(sh.rolling(14,min_periods=1).mean().iloc[-1]),
            "Roll_Mean_28": float(sh.rolling(28,min_periods=1).mean().iloc[-1]),
            "Roll_Std_7": float(sh.rolling(7,min_periods=2).std().fillna(0).iloc[-1]),
            "EWMA_7": float(sh.ewm(span=7,min_periods=1).mean().iloc[-1]),
            "EWMA_28": float(sh.ewm(span=28,min_periods=1).mean().iloc[-1]),
            "Trend_7_28": np.clip((float(sh.rolling(7,min_periods=1).mean().iloc[-1]) -
                                    float(sh.rolling(28,min_periods=1).mean().iloc[-1])) /
                                   (float(sh.rolling(28,min_periods=1).mean().iloc[-1])+0.1), -5, 5),
            "WoW_Change": 0.0,
        }

    def build(self, date, prev_day=None):
        rows = []; ts = pd.Timestamp(date)
        for item in self.items:
            m = self.meta[item]
            qty_r = list(self.qty_history.get(item,[0.0]))   # updated — for recency
            qty_f = list(self.frozen_qty.get(item,[0.0]))     # frozen — for rolling
            dsl, s7d = self._recency(np.array(qty_r))
            dow = ts.dayofweek
            # Weekend reset: don't penalize items absent on the slow anchor day
            if dow >= 4: dsl = 0.0; s7d = max(s7d, 1)
            roll = self._rolling(np.array(qty_f))
            pdv = prev_day or {}
            row = {
                "DOW":dow, "Is_Weekend":1 if dow>=5 else 0,
                "Month":ts.month, "Year":ts.year, "WeekOfYear":ts.isocalendar()[1],
                "DayOfMonth":ts.day, "Quarter":(ts.month-1)//3+1,
                "MonthStart":1 if ts.day<=7 else 0, "MonthEnd":1 if ts.day>=25 else 0,
                "Is_Holiday_Season":1 if ts.month in(12,1) else 0,
                "WeekOfMonth":(ts.day-1)//7+1,
                "DaysFromStart":(ts-pd.Timestamp("2022-01-01")).days,
                "DOW_Sin":np.sin(2*np.pi*dow/7), "DOW_Cos":np.cos(2*np.pi*dow/7),
                "Month_Sin":np.sin(2*np.pi*ts.month/12), "Month_Cos":np.cos(2*np.pi*ts.month/12),
                "Days_Since_First_Sale":(ts-m["first_sale_date"]).days,
                "Item_Rank":m["rank"], "Item_Rank_Pct":m["rank_pct"],
                "Days_Since_Last_Sale":dsl, "Sales_Last_7D":s7d, **roll,
                "Day_Total_Qty":pdv.get("total_qty",0),
                "Day_Total_Items_Sold":float(pdv.get("total_items",0)),
                "Day_Total_Beverage":pdv.get("total_bev",0),
                "Day_Total_Food":pdv.get("total_food",0),
                "Day_Total_Qty_7D":pdv.get("total_qty_7d",0),
                "Is_Beverage":1 if m["category"]=="beverage" else 0,
                "Is_Food":1 if m["category"]=="food" else 0,
            }
            for it in self.items: row[f"Item_{it}"] = 1 if it==item else 0
            rows.append(row)
        return pd.DataFrame(rows)


def compute_buffer(full, items, margin_lost=3, spoilage_cost=1):
    """Per-item newsvendor buffer from MODEL backtest error std (v2_per_item_errors.csv).
    Falls back to naive lag-1 proxy if no backtest data available."""
    error_csv = os.path.join(TABLES_DIR, "v2_per_item_errors.csv")
    cr = margin_lost / (margin_lost + spoilage_cost) if (margin_lost + spoilage_cost) > 0 else 0.5
    z = max(0, min(2.5, (cr - 0.5) * 3.2))  # approx inverse normal

    buffers = {}
    if os.path.exists(error_csv):
        err_df = pd.read_csv(error_csv)
        for _, r in err_df.iterrows():
            item = r["Item"]
            buffers[item] = {
                "error_std": round(r["Error_Std"], 3),
                "buffer": round(z * r["Error_Std"], 2),
                "cr": round(cr, 2), "z": round(z, 3),
            }
        # Any item not in backtest gets overall fallback
        overall_std = err_df["Error_Std"].mean()
        for item in items:
            if item not in buffers:
                buffers[item] = {
                    "error_std": round(overall_std, 3),
                    "buffer": round(z * overall_std, 2),
                    "cr": round(cr, 2), "z": round(z, 3),
                }
    else:
        # Fallback: naive lag-1 proxy (worse, but better than nothing)
        print("    (no backtest data — using lag-1 proxy; run 06_metrics.py first)")
        TW = 84; cutoff = full["Date_Only"].max() - timedelta(days=TW)
        recent = full[full["Date_Only"] > cutoff]
        for item in items:
            it_data = recent[recent["Item"]==item].sort_values("Date_Only")
            actuals = it_data["Quantity"].values
            if len(actuals) < 14:
                buffers[item] = {"error_std": 0.0, "buffer": 0.0, "cr": cr, "z": z}
                continue
            naive = np.roll(actuals, 1); naive[0] = 0
            error_std = np.std((naive - actuals)[1:])
            buffers[item] = {
                "error_std": round(error_std, 3),
                "buffer": round(z * error_std, 2),
                "cr": round(cr, 2), "z": round(z, 3),
            }
    return buffers


def main():
    parser = argparse.ArgumentParser(description="Cafe daily demand forecast")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--margin", type=float, default=3,
                       help="Margin lost per stockout cup (default 3)")
    parser.add_argument("--spoilage", type=float, default=1,
                       help="Spoilage cost per unsold cup (default 1)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 55)
    print("CAFE SUPPLY FORECAST")
    print("=" * 55)

    # 1. Pull fresh data
    print("\n[1] Loading data...")
    sales = pull_fresh_sales()
    if sales is None:
        sales = pd.read_csv(DAILY_CSV)
        sales["Date_Only"] = pd.to_datetime(sales["Date_Only"])
    last_date = sales["Date_Only"].max()
    print(f"    {len(sales)} rows, {sales['Item'].nunique()} items, last: {last_date.date()}")

    # 2. Build features
    print("[2] Building features...")
    full = build_features(sales)
    feature_cols = get_feature_cols(full)
    print(f"    {len(feature_cols)} features, {len(full)} rows")

    # 3. Train
    print("[3] Training model...")
    X, y = full[feature_cols].fillna(0), full["Quantity"]
    model = xgb.XGBRegressor(
        objective="count:poisson", n_estimators=200, max_depth=4,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=0.5, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X, y)

    # Save model
    with open(os.path.join(MODELS_DIR, "forecast_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 4. Setup forecaster
    items = sorted(full["Item"].unique())
    meta = {}
    for item in items:
        grp = full[full["Item"]==item]; s = grp[grp["Quantity"]>0]
        meta[item] = {
            "first_sale_date": s["Date_Only"].min() if len(s)>0 else grp["Date_Only"].min(),
            "category": grp["Category"].mode().iloc[0] if not grp["Category"].mode().empty else "?",
            "rank": float(grp["Item_Rank"].iloc[0]),
            "rank_pct": float(grp["Item_Rank_Pct"].iloc[0]),
        }

    fc = Forecaster(items, meta)
    fc.load_history(full, last_date)

    # 5. Initial cross-item from last day
    ld = full[full["Date_Only"]==last_date]
    cross = {"total_qty": float(ld["Quantity"].sum()),
             "total_items": int((ld["Quantity"]>0).sum()),
             "total_bev": float(ld[ld["Category"]=="beverage"]["Quantity"].sum()),
             "total_food": float(ld[ld["Category"]=="food"]["Quantity"].sum())}
    r7 = full[full["Date_Only"]>last_date-timedelta(days=7)]
    dt = r7.groupby("Date_Only")["Quantity"].sum().values
    cross["total_qty_7d"] = float(np.mean(dt)) if len(dt)>0 else cross["total_qty"]
    totals = list(dt)

    start = (last_date + timedelta(days=1)) if not args.start_date else \
            datetime.strptime(args.start_date, "%Y-%m-%d")

    # 6. Forecast
    print(f"[4] Forecasting {args.days} days from {start.date()}...")
    all_rows = []
    for d in range(args.days):
        date = start + timedelta(days=d)
        Xf = fc.build(date, prev_day=cross)
        preds = np.maximum(model.predict(Xf[feature_cols].fillna(0)), 0)
        fc.update_recency(preds)

        for i, item in enumerate(items):
            all_rows.append({"Date": date.date(), "Item": item,
                             "Predicted": round(preds[i], 2)})

        # Update cross-item only
        tq = float(np.sum(preds)); totals.append(tq)
        if len(totals) > 7: totals = totals[-7:]
        cross = {"total_qty": tq, "total_items": int(np.sum(preds>0.5)),
                 "total_bev": float(np.sum(preds[[i for i,it in enumerate(items)
                    if meta[it]["category"]=="beverage"]])),
                 "total_food": tq - cross["total_bev"],
                 "total_qty_7d": float(np.mean(totals))}

    fc_df = pd.DataFrame(all_rows)

    # 7. Buffer
    print("[5] Computing newsvendor buffers...")
    cr = args.margin / (args.margin + args.spoilage)
    print(f"    Margin={args.margin}, Spoilage={args.spoilage} → Critical Ratio={cr:.0%}")

    buf = compute_buffer(full, items, args.margin, args.spoilage)
    fc_df["Error_Std"] = fc_df["Item"].map(lambda i: buf[i]["error_std"])
    fc_df["Buffer"] = fc_df["Item"].map(lambda i: buf[i]["buffer"])
    fc_df["Supply"] = np.round(fc_df["Predicted"] + fc_df["Buffer"], 1)

    # 8. Save
    out = args.output or os.path.join(TABLES_DIR, "forecast.csv")
    fc_df.to_csv(out, index=False)
    print(f"\n    Saved: {out}")

    # 9. Summary
    daily = fc_df.groupby("Date")
    pred_tot = daily["Predicted"].sum()
    supply_tot = daily["Supply"].sum()
    dn = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    print(f"\n{'='*55}")
    print("DAILY TOTALS")
    print(f"{'='*55}")
    for d in sorted(fc_df["Date"].unique()):
        dow = dn[pd.Timestamp(d).dayofweek]
        print(f"  {d} ({dow}): pred={pred_tot[d]:.0f}  supply={supply_tot[d]:.0f}")
    print(f"  {'─'*40}")
    print(f"  Weekly:    pred={pred_tot.sum():.0f}  supply={supply_tot.sum():.0f}")

    top = fc_df.groupby("Item")["Supply"].sum().nlargest(10)
    print(f"\nTOP 10 ITEMS (supply, 7-day):")
    for item, s in top.items():
        p = fc_df[fc_df["Item"]==item]["Predicted"].sum()
        print(f"  {item:<35} pred={p:>5.0f}  supply={s:>5.0f}")


if __name__ == "__main__":
    main()
