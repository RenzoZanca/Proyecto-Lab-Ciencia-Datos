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
import gc  # Add garbage collection for memory management

SHARED_DATA_DIR = "/shared-data"
HISTORICAL_DIR = os.path.join(SHARED_DATA_DIR, "transacciones_historicas")

def get_data(**kwargs):
    """
    Obtiene los datos de la fuente según la fecha de ejecución y actualiza
    las transacciones históricas con los nuevos registros sin duplicar.
    """
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    data_dir = os.path.join(base_path, 'data')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(HISTORICAL_DIR, exist_ok=True)

    try:
        nuevas_transacciones = pd.read_parquet(os.path.join(data_dir, 'transacciones.parquet'))
        clientes = pd.read_parquet(os.path.join(data_dir, 'clientes.parquet'))
        productos = pd.read_parquet(os.path.join(data_dir, 'productos.parquet'))

        print(f"Datos nuevos cargados para fecha: {execution_date}")
        print(f"   - Nuevas Transacciones: {len(nuevas_transacciones)} registros")
        print(f"   - Clientes: {len(clientes)} registros")
        print(f"   - Productos: {len(productos)} registros")

        # Cargar y actualizar transacciones históricas
        historico_path = os.path.join(HISTORICAL_DIR, "transacciones.parquet")
        if os.path.exists(historico_path):
            historicas = pd.read_parquet(historico_path)
            transacciones = pd.concat([historicas, nuevas_transacciones], ignore_index=True)
            transacciones = transacciones.drop_duplicates()
            print(f"   - Histórico actualizado: {len(transacciones)} registros totales")
        else:
            transacciones = nuevas_transacciones
            print(f"   - Creando histórico con {len(transacciones)} registros")

        transacciones.to_parquet(historico_path, index=False)

    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}")
        print(f"Para esta demo, copie los archivos base a: {data_dir}/")
        raise

    return {
        'transacciones': transacciones,
        'clientes': clientes,
        'productos': productos
    }

# ----------------------------------------------------------------------------

# Procesamiento de datos:

def process_clients(clients):
    clients = clients.copy()
    
    # Manejar valores NaN en coordenadas ANTES del clustering
    clients['X'] = clients['X'].fillna(clients['X'].median())
    clients['Y'] = clients['Y'].fillna(clients['Y'].median())
    
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
    transactions = transactions.copy()
    transactions.drop_duplicates(inplace=True)
    transactions.dropna(inplace=True)
    transactions = transactions[transactions['items'] > 0]
    transactions['items'] = transactions['items'].astype('int64')
    
    # Manejar tanto datos crudos (con purchase_date) como procesados (con week)
    if 'purchase_date' in transactions.columns and 'week' not in transactions.columns:
        # Datos crudos - convertir purchase_date a week
        transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
        transactions['week'] = transactions['purchase_date'].dt.isocalendar().week
        transactions.loc[transactions['purchase_date'] > '2024-12-29', 'week'] = 53
    elif 'week' not in transactions.columns:
        # Si no hay ni purchase_date ni week, es un error
        raise ValueError("Las transacciones deben tener 'purchase_date' o 'week'")
    
    # Si ya tiene 'week', no hacer nada (datos ya procesados)
    return transactions

# Crea combinaciones semanales de productos, clientes y semanas
def create_weekly_combinations(transactions):
    # Instead of creating ALL possible combinations (which creates ~79M rows),
    # only create combinations for customer-product pairs that actually exist
    customer_products = transactions[['customer_id', 'product_id']].drop_duplicates()
    weeks = transactions['week'].unique()
    
    # Create combinations only for existing customer-product pairs
    combinations = []
    for week in weeks:
        week_combinations = customer_products.copy()
        week_combinations['week'] = week
        combinations.append(week_combinations)
    
    return pd.concat(combinations, ignore_index=True)

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
    # Columnas base requeridas
    keep = [
        'customer_id','product_id','week',
        'items_last_week','items_roll4_mean',
        'customer_type','Y','X','num_deliver_per_week','custom_zone',
        'brand','category','sub_category','segment','package','size',
        'label'
    ]
    
    # Agregar purchase_date solo si existe
    if 'purchase_date' in tx.columns:
        keep.insert(0, 'purchase_date')
    
    df = tx[keep].copy()
    
    # Procesar purchase_date solo si existe
    if 'purchase_date' in df.columns:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    df[['customer_type','brand','category','sub_category','segment','package']] = \
        df[['customer_type','brand','category','sub_category','segment','package']].astype('category')
    return df

# Función principal para procesar los datos
def process_data(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)

    print("🔄 Iniciando procesamiento de datos...")
    
    # Get and process data step by step to manage memory
    data = get_data(**kwargs)
    print(f"✅ Datos cargados. Procesando clientes...")
    
    clients = process_clients(data['clientes'])
    print(f"✅ {len(clients)} clientes procesados. Liberando memoria...")
    del data['clientes']
    gc.collect()
    
    print(f"🔄 Procesando productos...")
    products = process_products(data['productos'])
    print(f"✅ {len(products)} productos procesados. Liberando memoria...")
    del data['productos']
    gc.collect()
    
    print(f"🔄 Procesando transacciones...")
    transactions = process_transactions(data['transacciones'])
    print(f"✅ {len(transactions)} transacciones procesadas. Liberando memoria...")
    del data['transacciones'], data
    gc.collect()

    print(f"🔄 Creando combinaciones semanales...")
    combinations = create_weekly_combinations(transactions)
    print(f"✅ {len(combinations):,} combinaciones creadas (optimizado). Enriqueciendo...")
    
    df_combined = enrich_combinations_with_transactions(combinations, transactions)
    print(f"✅ Combinaciones enriquecidas. Liberando memoria...")
    del combinations
    gc.collect()
    
    print(f"🔄 Combinando todos los datos...")
    tx = merge_all_data(df_combined, clients, products)
    print(f"✅ Datos combinados. Liberando memoria...")
    del df_combined, clients, products
    gc.collect()
    
    print(f"🔄 Finalizando dataset...")
    df_processed = finalize_dataset(tx)
    print(f"✅ Dataset final: {len(df_processed):,} registros")
    del tx
    gc.collect()
    
    # guardar el DataFrame procesado como parquet
    print(f"💾 Guardando datos procesados...")
    os.makedirs(os.path.join(base_path, "data_processed"), exist_ok=True)
    df_processed.to_parquet(os.path.join(base_path, "data_processed/df_processed.parquet"), index=False)
    print(f"✅ Procesamiento completado exitosamente!")
    
    # Final cleanup
    del df_processed
    gc.collect()

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
def holdout(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)

    df = pd.read_parquet(os.path.join(base_path, "data_processed/df_processed.parquet"))
    df_final_ord = df.sort_values("week")
    
    # Ajustar proporciones para compensar el undersampling
    # Usamos más datos para train inicialmente
    train_cut = df_final_ord["week"].quantile(0.80)  # 80% para train (luego se reduce con undersampling)
    val_cut   = df_final_ord["week"].quantile(0.90)  # 10% para val
    train_df = df_final_ord[df_final_ord["week"] <= train_cut]
    val_df   = df_final_ord[(df_final_ord["week"] > train_cut) & (df_final_ord["week"] <= val_cut)]
    test_df  = df_final_ord[df_final_ord["week"] > val_cut]  # 10% para test

    print(f"Tamaños antes del undersampling:")
    print(f"   - Train: {len(train_df):,} ({len(train_df)/len(df_final_ord)*100:.1f}%)")
    print(f"   - Val:   {len(val_df):,} ({len(val_df)/len(df_final_ord)*100:.1f}%)")
    print(f"   - Test:  {len(test_df):,} ({len(test_df)/len(df_final_ord)*100:.1f}%)")

    # Aplicar undersampling temporal al conjunto de entrenamiento
    train_df_original_size = len(train_df)
    train_df = temporal_undersample(train_df, ratio=2)  # Ratio más conservador
    
    print(f" Tamaños después del undersampling:")
    print(f"   - Train: {len(train_df):,} ({len(train_df)/len(df_final_ord)*100:.1f}%) - Reducido en {(1-len(train_df)/train_df_original_size)*100:.1f}%")
    print(f"   - Val:   {len(val_df):,} ({len(val_df)/len(df_final_ord)*100:.1f}%)")
    print(f"   - Test:  {len(test_df):,} ({len(test_df)/len(df_final_ord)*100:.1f}%)")
    
    # Verificar balances
    print(f" Distribución de clases:")
    print(f"   - Train: {train_df['label'].value_counts().to_dict()}")
    print(f"   - Val:   {val_df['label'].value_counts().to_dict()}")
    print(f"   - Test:  {test_df['label'].value_counts().to_dict()}")

    # guardar los DataFrames de entrenamiento, validación y prueba
    os.makedirs(os.path.join(base_path, "data_holdout"), exist_ok=True)
    train_df.to_parquet(os.path.join(base_path, "data_holdout/train_df.parquet"), index=False)
    val_df.to_parquet(os.path.join(base_path, "data_holdout/val_df.parquet"), index=False)
    test_df.to_parquet(os.path.join(base_path, "data_holdout/test_df.parquet"), index=False)

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
        # Usar purchase_date si existe y no es nulo, sino generar fecha desde week
        if "purchase_date" in row.index and pd.notna(row["purchase_date"]):
            return row["purchase_date"]
        return semana_a_fecha_random(int(row["week"]))
    
    X["date"]  = X.apply(pick_date, axis=1)
    X["month"] = X["date"].dt.month.astype("category")
    X["day"]   = X["date"].dt.day.astype("category")
    
    # Eliminar purchase_date solo si existe
    if "purchase_date" in X.columns:
        X = X.drop(columns="purchase_date")
    
    return X


# Función principal de ingeniería de características  
def feature_engineering(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)

    
    # Usar solo datos de entrenamiento para calcular estadísticas
    train_df = pd.read_parquet(os.path.join(base_path, "data_holdout/train_df.parquet"))
    
    #  Las estadísticas se calculan SOLO con datos de entrenamiento
    # Esto evita data leakage y garantiza reproducibilidad en producción
    print(" Calculando estadísticas SOLO con datos de entrenamiento...")
    
    # Contar transacciones por cliente - usar columna disponible
    count_col = 'purchase_date' if 'purchase_date' in train_df.columns else 'week'
    client_trans = (
        train_df
            .groupby("customer_id")[count_col]
            .count()
            .reset_index(name="total_transactions")
    )

    # Promedios semanales por cliente (solo entrenamiento)
    
    weekly_agg = (
        train_df
        .groupby(["customer_id","week"])
        .agg(
            items_per_week    = ("items_last_week","sum"),  
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

    # Periodo medio de recompra por producto (solo entrenamiento)
    if 'purchase_date' in train_df.columns:
        product_buyback = (
            train_df
            .sort_values(["product_id","purchase_date"])
            .groupby("product_id")["purchase_date"]
            .apply(lambda x: x.diff().mean().days)
            .reset_index(name="avg_time_between_sales")
        )
    else:
        # Si no hay purchase_date, usar un valor aproximado basado en weeks
        product_buyback = (
            train_df
            .sort_values(["product_id","week"])
            .groupby("product_id")["week"]
            .apply(lambda x: x.diff().mean() * 7 if len(x) > 1 else 7)  # weeks * 7 days
            .reset_index(name="avg_time_between_sales")
        )
    
    print(f"Estadísticas calculadas:")
    print(f"   - Clientes únicos: {len(client_trans)}")
    print(f"   - Promedios semanales: {len(weekly_agg)}")
    print(f"   - Productos con historial: {len(product_buyback)}")
    
    # Guardar estas estadísticas para usarlas en predicción
    stats_dir = os.path.join(base_path, "feature_stats")
    os.makedirs(stats_dir, exist_ok=True)
    client_trans.to_parquet(os.path.join(stats_dir, "client_stats.parquet"), index=False)
    weekly_agg.to_parquet(os.path.join(stats_dir, "weekly_stats.parquet"), index=False)
    product_buyback.to_parquet(os.path.join(stats_dir, "product_stats.parquet"), index=False)

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

    # Cargar los datos de holdout no necesitamos recalcular train_df)
    val_df   = pd.read_parquet(os.path.join(base_path, "data_holdout/val_df.parquet"))
    test_df  = pd.read_parquet(os.path.join(base_path, "data_holdout/test_df.parquet"))
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

   
    os.makedirs(os.path.join(base_path, "data_transformed"), exist_ok=True)

   
    X_train_tr.to_parquet(os.path.join(base_path, "data_transformed/X_train.parquet"))
    y_train.to_frame().to_parquet(os.path.join(base_path, "data_transformed/y_train.parquet"))

    X_val_tr.to_parquet(os.path.join(base_path, "data_transformed/X_val.parquet"))
    y_val.to_frame().to_parquet(os.path.join(base_path, "data_transformed/y_val.parquet"))

    X_test_tr.to_parquet(os.path.join(base_path, "data_transformed/X_test.parquet"))
    y_test.to_frame().to_parquet(os.path.join(base_path, "data_transformed/y_test.parquet"))

   
    joblib.dump(features_pipeline, os.path.join(base_path, "data_transformed/features_pipeline.pkl"))