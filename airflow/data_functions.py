"""
estructura planteada:

get_data: obtiene los datos de la fuente
process_data: procesa los datos obtenidos (limpieza, transformación, etc.)
holdout: divide los datos en conjuntos de entrenamiento, validación y prueba
feature_engineering: realiza la ingeniería de características
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import itertools
from sklearn.utils import resample
from sklearn.base import TransformerMixin, BaseEstimator
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os
import joblib

# Obtiene los datos de la fuente
def get_data():
     # nota: cambiar paths en un futuro (probablemente según fecha)
    transacciones = pd.read_parquet('data/transacciones.parquet')
    clientes = pd.read_parquet('data/clientes.parquet')
    productos = pd.read_parquet('data/productos.parquet')
    data = {
        'transacciones': transacciones,
        'clientes': clientes,
        'productos': productos
    }
    return data

# ----------------------------------------------------------------------------

# Procesamiento de datos:

def process_clients(clients):
    coordinates = clients[['X', 'Y']]
    dbscan = DBSCAN(eps=0.01, min_samples=10)
    dbscan.fit(coordinates)
    clients['custom_zone'] = dbscan.labels_
    clients.drop_duplicates(inplace=True)
    clients.dropna(inplace=True)
    return clients

def process_products(products):
    products.drop_duplicates(inplace=True)
    products.dropna(inplace=True)
    return products

def process_transactions(transactions):
    transactions.drop_duplicates(inplace=True)
    transactions.dropna(inplace=True)
    transactions = transactions[transactions['items'] > 0]
    transactions['items'] = transactions['items'].astype('int64')
    transactions['week'] = transactions['purchase_date'].dt.isocalendar().week
    transactions.loc[transactions['purchase_date'] > '2024-12-29', 'week'] = 53
    return transactions

# Crea combinaciones semanales de productos, clientes y semanas
def create_weekly_combinations(transactions):
    product_ids = transactions['product_id'].unique()
    customer_ids = transactions['customer_id'].unique()
    weeks = transactions['week'].unique()
    return pd.DataFrame(list(itertools.product(product_ids, customer_ids, weeks)),
                        columns=['product_id', 'customer_id', 'week'])

# Enriquecer las combinaciones con transacciones
# para obtener etiquetas y características adicionales
def enrich_combinations_with_transactions(combinations, transactions):
    df = combinations.merge(transactions, on=['product_id', 'customer_id', 'week'], how='left')
    df['label'] = df['order_id'].notna().astype(int)
    df = df.sort_values(['customer_id','product_id','week'])
    df['items_last_week'] = (
        df.groupby(['customer_id','product_id'])['items'].shift(1).fillna(0)
    )
    df['items_roll4_mean'] = (
        df.groupby(['customer_id','product_id'])['items']
        .rolling(window=4, min_periods=1).mean()
        .reset_index(level=[0,1], drop=True)
    )
    return df

# Combina los datos de transacciones con clientes y productos
def merge_all_data(df_combined, clients, products):
    tx = df_combined.merge(
        clients, on="customer_id", how="left",
        validate="many_to_one", indicator="client_join"
    )
    tx = tx.merge(
        products, on="product_id", how="left",
        validate="many_to_one", indicator="product_join"
    )
    return tx.drop(columns=["client_join", "product_join"])

# Finaliza el dataset con las columnas necesarias y tipos de datos adecuados
def finalize_dataset(tx):
    keep = [
        'purchase_date','customer_id','product_id','week',
        'items_last_week','items_roll4_mean',
        'customer_type','Y','X','num_deliver_per_week','custom_zone',
        'brand','category','sub_category','segment','package','size',
        'label'
    ]
    df = tx[keep].copy()
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df[['customer_type','brand','category','sub_category','segment','package']] = \
        df[['customer_type','brand','category','sub_category','segment','package']].astype('category')
    return df

# Función principal para procesar los datos
def process_data():
    data = get_data()
    clients = process_clients(data['clientes'])
    products = process_products(data['productos'])
    transactions = process_transactions(data['transacciones'])

    combinations = create_weekly_combinations(transactions)
    df_combined = enrich_combinations_with_transactions(combinations, transactions)
    tx = merge_all_data(df_combined, clients, products)
    df_processed = finalize_dataset(tx)
    # guardar el DataFrame procesado como parquet
    os.makedirs("data_processed", exist_ok=True)
    df_processed.to_parquet("data_processed/df_processed.parquet", index=False)

# ----------------------------------------------------------------------------

# Holdout:

def temporal_undersample(df, ratio=4, time_col='week', label_col='label', random_state=42):
    """Aplica undersampling estratificado por semana, manteniendo todos los positivos y una proporción limitada de negativos."""
    parts = []
    for week, grp in df.groupby(time_col):
        pos = grp[grp[label_col] == 1]
        neg = grp[grp[label_col] == 0]
        n_neg = min(len(neg), len(pos) * ratio)
        neg_sub = resample(neg, replace=False, n_samples=n_neg, random_state=random_state)
        parts.append(pd.concat([pos, neg_sub]))
    return pd.concat(parts).sample(frac=1, random_state=random_state)

# Divide los datos en conjuntos de entrenamiento, validación y prueba
def holdout():
    df = pd.read_parquet("data_processed/df_processed.parquet")
    df_final_ord = df.sort_values("week")
    train_cut = df_final_ord["week"].quantile(0.70)
    val_cut   = df_final_ord["week"].quantile(0.85)
    train_df = df_final_ord[df_final_ord["week"] <= train_cut]
    val_df   = df_final_ord[(df_final_ord["week"] > train_cut) & (df_final_ord["week"] <= val_cut)]
    test_df  = df_final_ord[df_final_ord["week"] > val_cut]

    # Aplicar undersampling temporal al conjunto de entrenamiento
    train_df = temporal_undersample(train_df, ratio=4)

    # guardar los DataFrames de entrenamiento, validación y prueba
    os.makedirs("data_holdout", exist_ok=True)
    train_df.to_parquet("data_holdout/train_df.parquet", index=False)
    val_df.to_parquet("data_holdout/val_df.parquet", index=False)
    test_df.to_parquet("data_holdout/test_df.parquet", index=False)

# ----------------------------------------------------------------------------

# Feature engineering:

class StaticFeatureMerger(BaseEstimator, TransformerMixin):
    def __init__(self, df, on, how="left"):
        self.df = df
        self.on = on
        self.how = how
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.merge(self.df, on=self.on, how=self.how)
    
# Función que recibe una semana y devuelve la fecha de algun dia al azar de esa semana
def semana_a_fecha_random(week, year=2024):
    # Clip entre 1 y 52 para evitar week 53 inválida
    w = min(max(week, 1), 52)
    monday = datetime.fromisocalendar(year, w, 1)
    return monday + timedelta(days=random.randint(0, 6))

def extract_date_features(X):
    X = X.copy()
    def pick_date(row):
        if pd.notna(row["purchase_date"]):
            return row["purchase_date"]
        return semana_a_fecha_random(int(row["week"]))
    X["date"]  = X.apply(pick_date, axis=1)
    X["month"] = X["date"].dt.month.astype("category")
    X["day"]   = X["date"].dt.day.astype("category")
    return X.drop(columns="purchase_date")


# Función principal de ingeniería de características  
def feature_engineering():
    transactions = process_transactions(pd.read_parquet("data/transacciones.parquet"))
    client_trans = (
        transactions
            .groupby("customer_id")["purchase_date"]
            .count()
            .reset_index(name="total_transactions")
    )

    # Promedios semanales por cliente
    weekly_agg = (
        transactions
        .groupby(["customer_id","week"])
        .agg(
            items_per_week    = ("items","sum"),
            products_per_week = ("product_id","count")
        )
        .groupby("customer_id")
        .mean()
        .reset_index()
        .rename(columns={
            "items_per_week":    "avg_items_per_week",
            "products_per_week": "avg_products_per_week"
        })
    )

    # Periodo medio de recompra por producto (en días)
    product_buyback = (
        transactions
        .sort_values(["product_id","purchase_date"])
        .groupby("product_id")["purchase_date"]
        .apply(lambda x: x.diff().mean().days)
        .reset_index(name="avg_time_between_sales")
    )

    date_transformer = FunctionTransformer(extract_date_features)

    # Listas de columnas
    numerical_cols = [
        "Y", "X",
        "num_deliver_per_week",
        'items_last_week','items_roll4_mean',
        "size",
        "total_transactions",
        "avg_products_per_week",
        "avg_items_per_week",
        "avg_time_between_sales"
    ]
    categorical_cols = [
        "custom_zone",
        "customer_type",
        "brand",
        "category",
        "sub_category",
        "segment",
        "package",
        "week",
        "month",
        "day"
    ]

    # Pipeline para numéricas: imputar + escalar
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler()),
    ])

    # Pipeline para categóricas: imputar + one-hot
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                sparse_output=False))
    ])

    # ColumnTransformer que une ambas
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ], remainder="drop").set_output(transform="pandas")

    # Pipeline final
    features_pipeline = Pipeline([
        # merge de features precomputadas
        ("merge_trans_count", StaticFeatureMerger(client_trans,    on="customer_id")),
        ("merge_weekly",      StaticFeatureMerger(weekly_agg,      on="customer_id")),
        ("merge_buyback",     StaticFeatureMerger(product_buyback, on="product_id")),
        # extracción de fecha → month, day
        ("date_feats",        date_transformer),
        # preprocesamiento numérico y categórico
        ("preprocessing",     preprocessor),
    ])

    # Cargar los datos de holdout
    train_df = pd.read_parquet("data_holdout/train_df.parquet")
    val_df   = pd.read_parquet("data_holdout/val_df.parquet")
    test_df  = pd.read_parquet("data_holdout/test_df.parquet")
    # Separar características y etiquetas
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_val   = val_df.drop(columns=["label"])
    y_val   = val_df["label"]
    X_test  = test_df.drop(columns=["label"])
    y_test  = test_df["label"]

    # Ajustar y transformar los datos
    features_pipeline.fit(X=X_train, y=y_train)
    X_train_tr = features_pipeline.transform(X_train)
    X_val_tr   = features_pipeline.transform(X_val)
    X_test_tr  = features_pipeline.transform(X_test)

    # Make directory if it doesn't exist
    os.makedirs("data_transformed", exist_ok=True)

    # Save as parquet (or use .to_csv("filename.csv") if preferred)
    X_train_tr.to_parquet("data_transformed/X_train.parquet")
    y_train.to_frame().to_parquet("data_transformed/y_train.parquet")

    X_val_tr.to_parquet("data_transformed/X_val.parquet")
    y_val.to_frame().to_parquet("data_transformed/y_val.parquet")

    X_test_tr.to_parquet("data_transformed/X_test.parquet")
    y_test.to_frame().to_parquet("data_transformed/y_test.parquet")

    # Optional: save the pipeline using joblib
    joblib.dump(features_pipeline, "data_transformed/features_pipeline.pkl")