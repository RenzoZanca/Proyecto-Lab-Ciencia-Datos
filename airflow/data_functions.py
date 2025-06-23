"""
estructura planteada:

get_data: obtiene los datos de la fuente
process_data: procesa los datos obtenidos (limpieza, transformación, etc.)
feature_engineering: realiza la ingeniería de características
export_data: exporta los datos procesados a un archivo
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import itertools

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

# funciones auxiliares para el procesamiento de datos
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

def create_weekly_combinations(transactions):
    product_ids = transactions['product_id'].unique()
    customer_ids = transactions['customer_id'].unique()
    weeks = transactions['week'].unique()
    return pd.DataFrame(list(itertools.product(product_ids, customer_ids, weeks)),
                        columns=['product_id', 'customer_id', 'week'])

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
def process_data(data):
    clients = process_clients(data['clientes'])
    products = process_products(data['productos'])
    transactions = process_transactions(data['transacciones'])

    combinations = create_weekly_combinations(transactions)
    df_combined = enrich_combinations_with_transactions(combinations, transactions)
    tx = merge_all_data(df_combined, clients, products)
    return finalize_dataset(tx)

    
def feature_engineering(data):
    pass

def export_data(data, path):
    pass




