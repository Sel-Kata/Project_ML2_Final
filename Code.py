import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from minisom import MiniSom
import umap
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 1. Data Preprocessing & Feature Engineering
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['customer_gender'] = df['customer_gender'].astype('category')
    df['customer_name'] = df['customer_name'].astype('string')

    df_clean = df.copy()

    # Convert birthdate to Age
    current_year = datetime.now().year
    df_clean['customer_birthdate'] = pd.to_datetime(df_clean['customer_birthdate'])
    df_clean['age'] = current_year - df_clean['customer_birthdate'].dt.year

    # Fill age missing values with median
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

    # Presence of loyalty card (1 - yes, 0 - no)
    df_clean['has_loyalty_card'] = df_clean['loyalty_card_number'].notna().astype(int)

    # Drop columns not needed for clustering
    # customer_id can be kept in index
    df_clean = df_clean.set_index('customer_id')
    cols_to_drop = ['customer_name', 'loyalty_card_number', 'customer_birthdate']
    df_clean = df_clean.drop(columns=cols_to_drop)

    return df_clean

def handle_outliers_and_sanity(df_clean):
    numeric_cols_for_outliers = [
        'kids_home', 'teens_home', 'number_complaints', 'typical_hour',
        'distinct_stores_visited', 'lifetime_spend_groceries', 
        'lifetime_spend_electronics', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products', 'percentage_of_products_bought_promotion',
        'age'
    ]
    non_negative_cols = [
        'kids_home', 'teens_home', 'number_complaints', 'distinct_stores_visited',
        'lifetime_spend_groceries', 'lifetime_spend_electronics', 
        'lifetime_spend_vegetables', 'lifetime_spend_nonalcohol_drinks', 
        'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat', 
        'lifetime_spend_fish', 'lifetime_spend_hygiene', 
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products', 
        'percentage_of_products_bought_promotion', 'typical_hour'
    ]

    # Sanity Checks
    for col in non_negative_cols:
        # Clip values less than 0 to 0
        df_clean[col] = df_clean[col].clip(lower=0)
        
    # Percentage of promotional purchases cannot exceed 100% 
    df_clean['percentage_of_products_bought_promotion'] = df_clean['percentage_of_products_bought_promotion'].clip(upper=100)
    df_clean['typical_hour'] = df_clean['typical_hour'].clip(upper=23)
    df_clean['age'] = df_clean['age'].clip(lower=14, upper=100)

    # Remove outliers
    for col in numeric_cols_for_outliers:
        upper_limit = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=upper_limit)

    df_clean = pd.get_dummies(df_clean, columns=['customer_gender'], drop_first=True, dtype=int)
    return df_clean

def scale_and_impute_data(df_clean):
    # Scales data and imputes missing values using KNN.
    features = df_clean.columns
    customer_indexes = df_clean.index

    scaler = StandardScaler()
    df_scaled_matrix = scaler.fit_transform(df_clean)

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    df_imputed_matrix = imputer.fit_transform(df_scaled_matrix)
    
    df_final = pd.DataFrame(df_imputed_matrix, columns=features, index=customer_indexes)

    print("Number of missing values after processing:", df_final.isnull().sum().sum())
    print("Dimensions of the final dataset:", df_final.shape)
    
    return df_final

# 2. Exploratory Data Analysis (EDA)
def plot_distributions(df_clean):
    sns.set_theme(style="whitegrid")
    
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64', 'int32']).columns

    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df_clean, x=col, kde=True, ax=axes[i], bins=30, color='steelblue')
        axes[i].set_title(f'Dist: {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')

    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df_clean, threshold=0.85):
    corr_matrix = df_clean.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # If absolute correlation > threshold, consider features as duplicates
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop_corr = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]

    print(f"Correlation Threshold: {threshold}")
    print(f"Columns to drop (highly correlated): {cols_to_drop_corr}")

# 3. Clustering Models
def evaluate_optimal_clusters(df_final):
    k_range = range(2, 15)
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_final)
        inertia.append(kmeans.inertia_)
        
        # Calculate Silhouette Score (-1, 1 - higher is better)
        labels = kmeans.labels_
        score = silhouette_score(df_final, labels)
        silhouette_scores.append(score)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Elbow Method
    axes[0].plot(k_range, inertia, marker='o', color='steelblue', linewidth=2)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('k', fontsize=12)
    axes[0].set_ylabel('WCSS', fontsize=12)
    axes[0].set_xticks(k_range)

    # Silhouette Score
    axes[1].plot(k_range, silhouette_scores, marker='s', color='seagreen', linewidth=2)
    axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('k', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_xticks(k_range)

    plt.tight_layout()
    plt.show()

def run_kmeans(df_final, df_clean, n_clusters):
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(df_final)

    df_analysis = df_clean.copy()
    df_analysis['cluster'] = cluster_labels

    print(f"Customer distribution across K-Means clusters (k={n_clusters}):")
    print(df_analysis['cluster'].value_counts())
    print("-" * 50)

    cols_to_analyze = [
        'age', 'kids_home', 'teens_home', 'number_complaints', 'typical_hour',
        'distinct_stores_visited', 'lifetime_spend_groceries', 
        'lifetime_spend_electronics', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products', 'percentage_of_products_bought_promotion',
        'has_loyalty_card', 'cluster'
    ]

    # Safely select only columns that actually exist in df_analysis
    valid_cols = [col for col in cols_to_analyze if col in df_analysis.columns]

    cluster_profiles = df_analysis[valid_cols].groupby('cluster').mean().T
    cluster_profiles = cluster_profiles.round(2)

    print("\n--- K-Means Cluster Profiles ---")
    print(cluster_profiles)
    
    return cluster_labels, df_analysis

def run_som(df_final, df_clean, grid_x, grid_y=1):
    data_matrix = df_final.values
    n_samples, n_features = data_matrix.shape

    # SOM 
    som = MiniSom(x=grid_x, y=grid_y, input_len=n_features, sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(data_matrix)
    som.train_random(data_matrix, num_iteration=10000)
    
    # Best Matching Unit 
    som_labels = np.array([som.winner(x)[0] for x in data_matrix])
    som_silhouette = silhouette_score(data_matrix, som_labels)

    print(f"\nSOM Silhouette Score ({grid_x*grid_y} clusters): {som_silhouette:.4f}")
    print("\nCustomer distribution (SOM):")
    print(pd.Series(som_labels).value_counts().sort_index().values)

    df_som_analysis = df_clean.copy()
    df_som_analysis['cluster_som'] = som_labels

    cols_to_analyze_som = [
        'age', 'kids_home', 'teens_home', 'number_complaints', 'typical_hour',
        'distinct_stores_visited', 'lifetime_spend_groceries', 
        'lifetime_spend_electronics', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products', 'percentage_of_products_bought_promotion',
        'has_loyalty_card', 'cluster_som'
    ]

    valid_cols_som = [col for col in cols_to_analyze_som if col in df_som_analysis.columns]
    som_profiles = df_som_analysis[valid_cols_som].groupby('cluster_som').mean().T
    som_profiles = som_profiles.round(2)

    print("\n--- SOM Cluster Profiles ---")
    print(som_profiles)

    return som_labels, df_som_analysis

# 4. Dimensionality Reduction and Visualizations
def plot_pca_clusters(df_final, som_labels):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Distribution of people across clusters
    som_counts = pd.Series(som_labels).value_counts().sort_index()
    sns.barplot(x=som_counts.index, y=som_counts.values, ax=axes[0], palette="coolwarm")
    axes[0].set_title('Customer Distribution across SOM Clusters', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('SOM Cluster', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)

    for i, v in enumerate(som_counts.values):
        axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')

    # PCA
    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(df_final)

    df_viz = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'])
    df_viz['Cluster_SOM'] = som_labels

    sns.scatterplot(
        data=df_viz, x='PCA1', y='PCA2', 
        hue='Cluster_SOM', palette='coolwarm', alpha=0.6, s=30, ax=axes[1]
    )

    axes[1].set_title('SOM Clusters 2D Projection (PCA)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    axes[1].set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    axes[1].legend(title='SOM Cluster', loc='best')

    plt.tight_layout()
    plt.show()

def plot_umap_clusters(df_final, som_labels, n_neighbors, min_dist, alpha, s, title):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    data_umap = reducer.fit_transform(df_final)

    df_viz_umap = pd.DataFrame(data_umap, columns=['UMAP1', 'UMAP2'])
    df_viz_umap['Cluster_SOM'] = som_labels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_viz_umap, x='UMAP1', y='UMAP2', 
        hue='Cluster_SOM', palette='coolwarm', alpha=alpha, s=s  
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.legend(title='SOM Cluster', loc='best')
    plt.tight_layout()
    plt.show()

def compare_clustering_models(df_final, labels_dict):
    """
    Compares different clustering models using internal evaluation metrics.
    
    Parameters:
    df_final: The scaled and imputed dataframe.
    labels_dict: Dictionary with model names as keys and their cluster labels as values.
                 Example: {'K-Means': kmeans_labels, 'SOM': som_labels}
    """
    print("\n Clustering Models Comparison ")
    
    results = []
    
    for model_name, labels in labels_dict.items():
        # Calculate metrics
        sil_score = silhouette_score(df_final, labels)
        ch_score = calinski_harabasz_score(df_final, labels)
        
        results.append({
            'Model': model_name,
            'Silhouette (Higher is better)': sil_score,
            'Calinski-Harabasz (Higher is better)': ch_score
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.round(4).to_string(index=False))
    
    #Visualizing the comparison
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    #Silhouette Plot 
    sns.barplot(data=results_df, x='Model', y='Silhouette (Higher is better)', ax=axes[0], palette='viridis')
    axes[0].set_title('Silhouette Score \n(Higher = Better Separation)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score')
    
    #Calinski-Harabasz Plot
    sns.barplot(data=results_df, x='Model', y='Calinski-Harabasz (Higher is better)', ax=axes[1], palette='coolwarm')
    axes[1].set_title('Calinski-Harabasz Index \n(Higher = Better Density)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score')
    
    plt.tight_layout()
    plt.show()

# 5. Recommendation System
def generate_recommendations(basket_filepath, df_som_analysis, n_clusters):
    print("\n--- Starting Recommendation System ---")
    df_basket = pd.read_csv(basket_filepath)

    if isinstance(df_basket['list_of_goods'].iloc[0], str):
        df_basket['list_of_goods'] = df_basket['list_of_goods'].apply(ast.literal_eval)

    df_merged = df_basket[['customer_id', 'invoice_id', 'list_of_goods']].merge(
        df_som_analysis[['cluster_som']], 
        on='customer_id', 
        how='inner'
    )

    def get_recommendations_for_cluster(cluster_id, min_sup=0.02, min_conf=0.2):
        print(f"\n--- Analyzing Cluster {cluster_id} ---")
        
        cluster_baskets = df_merged[df_merged['cluster_som'] == cluster_id]['list_of_goods']
        
        if len(cluster_baskets) == 0:
            print("No data for this cluster.")
            return None
        
        te = TransactionEncoder()
        te_ary = te.fit(cluster_baskets).transform(cluster_baskets)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_transactions, min_support=min_sup, use_colnames=True)
        
        if frequent_itemsets.empty:
            print("No frequent itemsets found.")
            return None

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
        rules = rules[rules['lift'] > 1.2]
        rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])

        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        print(f"Rules found: {len(rules)}")
        return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)

    cluster_rules = {}
    
    # Process dynamically based on n_clusters
    for i in range(n_clusters):
        # min_sup=0.02: the itemset must appear in at least 2% of the transactions for this cluster
        # min_conf=0.2: if item A is bought, item B is bought in at least 20% of the cases
        rules = get_recommendations_for_cluster(cluster_id=i, min_sup=0.02, min_conf=0.2)
        
        if rules is not None:
            cluster_rules[i] = rules
            print(rules)

def generate_business_report(df_analysis, cluster_col='cluster_som'):
  
    print("\n" + "="*60)
    print(" CUSTOMER BEHAVIOR & MARKETING STRATEGIES")
    print("="*60)
    
    key_metrics = ['age', 'kids_home', 'typical_hour', 'number_complaints', 
                   'has_loyalty_card', 'distinct_stores_visited']
    
    valid_metrics = [col for col in key_metrics if col in df_analysis.columns]
    
    # mean by clasters
    cluster_summary = df_analysis.groupby(cluster_col)[valid_metrics].mean()
    
    cluster_names = {
        0: "Breakfast Lovers",
        1: "Youth / On-the-go",
        2: "Families with pets",
        3: "Healthy & Vegans ",
        4: "Tech & Athletes "
    }
    
    for cluster_id in sorted(df_analysis[cluster_col].unique()):
        print(f"\n{'='*40}")
        print(f" CLUSTER {cluster_id} - {cluster_names.get(cluster_id, 'Unknown')}")
        print(f"{'='*40}")
        
        row = cluster_summary.loc[cluster_id]
        
        # Customer Behavior
        print("\n[ Customer Behavior Analysis ]")
        print(f" • Average Age: {row['age']:.1f} years")
        print(f" • Kids at home: {row['kids_home']:.2f} (avg per customer)")
        print(f" • Typical Shopping Hour: {int(row['typical_hour'])}:00")
        print(f" • Loyalty Card Usage: {(row['has_loyalty_card'] * 100):.1f}%")
        print(f" • Average Complaints: {row['number_complaints']:.2f}")
        print(f" • Distinct Stores Visited: {row['distinct_stores_visited']:.1f}")
        
        # Targeted Marketing Strategies 
        print("\n[  Targeted Marketing Strategy ]")
        
        if cluster_id == 0:
            print(" • Cross-sell Combo: 'Buy eggs and butter — get 20% off fresh bread'.")
            print(f" • Timing Strategy: Send breakfast recipes and promo codes via email at {int(row['typical_hour'])-1}:00 (just before their typical shopping time).")
            
        elif cluster_id == 1:
            print(" • Trigger Promo: Offer a free energy drink when adding any headphones to the cart online.")
            print(f" • Push Notifications: Send mobile app alerts for quick snacks at {int(row['typical_hour'])}:00.")
            print(" • Note: This cluster has very few strict association rules, indicating chaotic/spontaneous purchase behavior.")
            
        elif cluster_id == 2:
            print(" • Bundle Offer: 'Family Basket' promo. Buy baby food + dog food, get a large pack of napkins with a 30% discount.")
            if row['number_complaints'] > df_analysis['number_complaints'].mean():
                print(" • Retention Strategy: This cluster has higher complaints. Send an apology promo code for their favorite bundle to increase loyalty.")
            else:
                print(" • Subscription Model: Send reminders to re-stock on diapers and dog food every 2-3 weeks.")
                
        elif cluster_id == 3:
            print(" • Targeted Discount: 20% off avocados when buying fresh asparagus and spinach.")
            if row['has_loyalty_card'] < 0.5:
                print(" • Acquisition Strategy: Loyalty card penetration is low. Offer the avocado discount ONLY upon app registration/loyalty card activation.")
            print(" • Communication: Send newsletters about fresh farm vegetable arrivals (do not push meat products).")
            
        elif cluster_id == 4:
            print(" • Sports Bundle: 'Post-Workout' promo code (Shampoo + Deodorant + Protein bar).")
            print(" • Upsell Strategy: When a high-ticket item (e.g., iPhone 10) is added to the cart, automatically offer a coupon for AirPods/Bluetooth headphones.")

    print("\n" + "="*60)
    print(" Next Steps / Evaluation:")
    print(" To measure the success of these strategies, we will implement A/B testing.")
    print(" The control group will not receive these targeted promos, while the test group will.")
    print(" Key metrics to track: Conversion Rate, Average Order Value (AOV), and Loyalty Card Sign-ups.")
    print("="*60)

def investigate_cluster_1_paradox(df_analysis):
    """Investigates the demographic paradox of Cluster 1 (Tech/Youth items vs High Age)."""
    print("\n" + "="*60)
    print(" INVESTIGATING THE PARADOX: CLUSTER 1 (Age 60+ buying AirPods)")
    print("="*60)
    
    cluster_1_data = df_analysis[df_analysis['cluster_som'] == 1]
    
    # 1. Check for teens/kids in the household
    print("\n1. Hypothesis: Are they buying gifts for teens/grandchildren?")
    teens_dist = cluster_1_data['teens_home'].value_counts(normalize=True) * 100
    print("Percentage of customers with teens at home:")
    for teens_count, percent in teens_dist.items():
        print(f" {int(teens_count)} teen(s): {percent:.1f}% of cluster")
        
    kids_dist = cluster_1_data['kids_home'].value_counts(normalize=True) * 100
    print("\nPercentage of customers with kids at home:")
    for kids_count, percent in kids_dist.items():
        print(f" {int(kids_count)} kid(s): {percent:.1f}% of cluster")

    # 2. Check lifetime spend patterns
    print("\n2. Hypothesis: One-time gift vs Regular consumption?")
    avg_electronics = cluster_1_data['lifetime_spend_electronics'].mean()
    avg_drinks = cluster_1_data['lifetime_spend_nonalcohol_drinks'].mean()
    print(f"Average lifetime spend on electronics: {avg_electronics:.1f}")
    print(f"Average lifetime spend on non-alcohol drinks: {avg_drinks:.1f}")
    
    # 3. Plotting distributions
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Age Distribution
    sns.histplot(cluster_1_data['age'], bins=20, kde=True, ax=axes[0], color='purple')
    axes[0].set_title('Age Distribution in Cluster 1', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Number of Customers')
    
    # Electronics Spend Distribution
    sns.histplot(cluster_1_data['lifetime_spend_electronics'], bins=20, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Lifetime Spend on Electronics Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lifetime Spend')
    axes[1].set_ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.show()

def analyze_profit_vs_stability(df_analysis):
    """
    Evaluates clusters based on Profitability (Total Spend) 
    and Stability (Loyalty, low complaints, low promo dependency).
    Draws a Customer Portfolio Matrix.
    """
    print("\n" + "="*60)
    print(" ADVANCED ANALYSIS: PROFITABILITY vs STABILITY MATRIX")
    print("="*60)
    
    # Calculate Total Lifetime Spend
    spend_cols = [c for c in df_analysis.columns if 'lifetime_spend_' in c]
    df_analysis['total_lifetime_spend'] = df_analysis[spend_cols].sum(axis=1)
    
    summary = df_analysis.groupby('cluster_som').agg(
        avg_spend=('total_lifetime_spend', 'mean'),
        loyalty_pct=('has_loyalty_card', lambda x: x.mean() * 100),
        promo_pct=('percentage_of_products_bought_promotion', 'mean'),
        avg_complaints=('number_complaints', 'mean')
    ).round(2)
    
    # Calculate a synthetic "Stability Score" (0 to 100 scale)
    summary['stability_score'] = (
        (summary['loyalty_pct'] * 1.0) - 
        (summary['promo_pct'] * 0.5) - 
        (summary['avg_complaints'] * 10)
    )
    # Normalize score to be roughly out of 100 for easier reading
    summary['stability_score'] = summary['stability_score'].apply(lambda x: max(0, min(100, x + 20))).round(1)
    
    print("\n[ Cluster Performance Table ]")
    print(summary[['avg_spend', 'stability_score', 'loyalty_pct', 'promo_pct', 'avg_complaints']].to_string())
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    scatter = sns.scatterplot(
        data=summary, 
        x='avg_spend', 
        y='stability_score', 
        hue=summary.index, 
        palette='tab10', 
        s=800, # Bubble size
        alpha=0.8,
        legend=False
    )
    
    # Add labels to the bubbles
    for i in summary.index:
        plt.text(summary.loc[i, 'avg_spend'], summary.loc[i, 'stability_score'], 
                 f"C{i}", horizontalalignment='center', verticalalignment='center', 
                 fontsize=12, color='white', fontweight='bold')
    
    # Calculate medians to draw quadrant lines
    median_spend = summary['avg_spend'].median()
    median_stability = summary['stability_score'].median()
    
    plt.axvline(median_spend, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(median_stability, color='gray', linestyle='--', alpha=0.7)
    
    plt.text(median_spend * 1.05, median_stability * 1.1, "STAR CUSTOMERS\n(High Profit, High Stability)", 
             color='green', fontweight='bold', alpha=0.6)
    plt.text(median_spend * 0.5, median_stability * 1.1, "LOYAL BUT LOW SPEND\n(Cash Cows)", 
             color='blue', fontweight='bold', alpha=0.6)
    plt.text(median_spend * 1.05, median_stability * 0.8, "AT-RISK WHALES\n(High Profit, Low Stability)", 
             color='red', fontweight='bold', alpha=0.6)
    plt.text(median_spend * 0.5, median_stability * 0.8, "LOW VALUE\n(Low Profit, Low Stability)", 
             color='orange', fontweight='bold', alpha=0.6)
    
    plt.title('Customer Portfolio Matrix: Profitability vs. Stability', fontsize=16, fontweight='bold')
    plt.xlabel('Profitability (Average Total Lifetime Spend, $)', fontsize=12)
    plt.ylabel('Stability Score (Loyalty + Low Complaints)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




info_file = "customer_info.csv"
basket_file = "customer_basket.csv"

print(" Loading and preprocessing data")
df_clean_initial = preprocess_data(info_file)
df_clean = handle_outliers_and_sanity(df_clean_initial)
df_final = scale_and_impute_data(df_clean)
"""
print("\n Generating EDA plots")
plot_distributions(df_clean)
plot_correlation_matrix(df_clean)

print("\n Evaluating optimal number of clusters")
evaluate_optimal_clusters(df_final)"""
NUM_CLUSTERS = 5

print(f"\n Running K-Means clustering with {NUM_CLUSTERS} clusters")
kmeans_labels, df_kmeans_analysis = run_kmeans(df_final, df_clean, n_clusters=NUM_CLUSTERS)

print(f"\n Running SOM clustering with {NUM_CLUSTERS} clusters")
# (1D SOM)
som_labels, df_som_analysis = run_som(df_final, df_clean, grid_x=NUM_CLUSTERS, grid_y=1)
"""
print("\n Generating Dimensionality Reduction visualisations")
plot_pca_clusters(df_final, som_labels)
        
# UMAP - Local View (n_neighbors=15, min_dist=0.1, alpha=0.6, s=15)
plot_umap_clusters(
            df_final=df_final, 
            som_labels=som_labels, 
            n_neighbors=15, 
            min_dist=0.1, 
            alpha=0.6, 
            s=15, 
            title='SOM Clusters 2D Projection (UMAP - Local View)'
        )
        
# UMAP - Global View (n_neighbors=50, min_dist=0.05, alpha=0.4, s=5)
plot_umap_clusters(
            df_final=df_final, 
            som_labels=som_labels, 
            n_neighbors=50, 
            min_dist=0.05, 
            alpha=0.4, 
            s=5, 
            title='SOM Clusters 2D Projection (UMAP - Global View)'
        )


print("\nComparing Clustering Models")
models_to_compare = {
        'K-Means': kmeans_labels,
        'SOM': som_labels
    }
compare_clustering_models(df_final, models_to_compare)

print("\n Generating Dimensionality Reduction visualisations")
plot_pca_clusters(df_final, som_labels)"""

print(f"\n Building Recommendation System based on {NUM_CLUSTERS} SOM clusters")
generate_recommendations(basket_file, df_som_analysis, n_clusters=NUM_CLUSTERS)

#generate_business_report(df_som_analysis, cluster_col='cluster_som')
#investigate_cluster_1_paradox(df_som_analysis)
analyze_profit_vs_stability(df_som_analysis)


"""
============================================================
 CUSTOMER BEHAVIOR & MARKETING STRATEGIES
============================================================

========================================
 CLUSTER 0 - Breakfast Lovers
========================================

[ Customer Behavior Analysis ]
 • Average Age: 58.3 years
 • Kids at home: 3.14 (avg per customer)
 • Typical Shopping Hour: 10:00
 • Loyalty Card Usage: 68.2%
 • Average Complaints: 0.95
 • Distinct Stores Visited: 3.3

[  Targeted Marketing Strategy ]
 • Cross-sell Combo: 'Buy eggs and butter — get 20% off fresh bread'.
 • Timing Strategy: Send breakfast recipes and promo codes via email at 9:00 (just before their typical shopping time).

========================================
 CLUSTER 1 - Youth / On-the-go
========================================

[ Customer Behavior Analysis ]
 • Average Age: 60.3 years
 • Kids at home: 0.90 (avg per customer)
 • Typical Shopping Hour: 11:00
 • Loyalty Card Usage: 73.1%
 • Average Complaints: 0.88
 • Distinct Stores Visited: 3.0

[  Targeted Marketing Strategy ]
 • Trigger Promo: Offer a free energy drink when adding any headphones to the cart online.
 • Push Notifications: Send mobile app alerts for quick snacks at 11:00.
 • Note: This cluster has very few strict association rules, indicating chaotic/spontaneous purchase behavior.

========================================
 CLUSTER 2 - Families with pets
========================================

[ Customer Behavior Analysis ]
 • Average Age: 58.2 years
 • Kids at home: 1.03 (avg per customer)
 • Typical Shopping Hour: 12:00
 • Loyalty Card Usage: 57.4%
 • Average Complaints: 0.72
 • Distinct Stores Visited: 3.5

[  Targeted Marketing Strategy ]
 • Bundle Offer: 'Family Basket' promo. Buy baby food + dog food, get a large pack of napkins with a 30% discount.
 • Subscription Model: Send reminders to re-stock on diapers and dog food every 2-3 weeks.

========================================
 CLUSTER 3 - Healthy & Vegans 
========================================

[ Customer Behavior Analysis ]
 • Average Age: 48.6 years
 • Kids at home: 0.94 (avg per customer)
 • Typical Shopping Hour: 12:00
 • Loyalty Card Usage: 55.6%
 • Average Complaints: 1.02
 • Distinct Stores Visited: 3.5

[  Targeted Marketing Strategy ]
 • Targeted Discount: 20% off avocados when buying fresh asparagus and spinach.
 • Communication: Send newsletters about fresh farm vegetable arrivals (do not push meat products).

========================================
 CLUSTER 4 - Tech & Athletes 
========================================

[ Customer Behavior Analysis ]
 • Average Age: 57.2 years
 • Kids at home: 0.27 (avg per customer)
 • Typical Shopping Hour: 17:00
 • Loyalty Card Usage: 48.4%
 • Average Complaints: 1.09
 • Distinct Stores Visited: 1.5

[  Targeted Marketing Strategy ]
 • Sports Bundle: 'Post-Workout' promo code (Shampoo + Deodorant + Protein bar).
 • Upsell Strategy: When a high-ticket item (e.g., iPhone 10) is added to the cart, automatically offer a coupon for AirPods/Bluetooth headphones.

 
 =======================================================================
BUSINESS INSIGHT: THE CLUSTER 1 PARADOX (Tech Items vs. High Age)
=======================================================================
• Initial Hypothesis: Based purely on the shopping basket (AirPods, 
  Bluetooth headphones, energy drinks), this cluster was assumed to be "Youth".
  
• Demographic Reality: The average age is ~60 years old. The distribution 
  is normal (no bimodality), meaning these are genuinely older adults.

• The Evidence: ~75% of these customers have teens or kids at home. 
  Furthermore, lifetime spend shows huge isolated spikes in electronics 
  (expensive one-time purchases) but low continuous spend on drinks.

• Final Conclusion: These are NOT teenagers. They are parents and 
  grandparents buying tech gifts for Gen Z.

• Marketing Pivot: Shift the communication strategy from "on-the-go youth" 
  messaging to "Gift Ideas for Teens" (e.g., Holidays, Back-to-School).
=======================================================================

=======================================================================
EXECUTIVE SUMMARY: CUSTOMER PORTFOLIO MATRIX
=======================================================================
By mapping our 5 clusters on a Profitability vs. Stability matrix, 
we successfully prioritized our marketing budget:

1. THE STARS (Clusters 0 & 1):
   - High revenue (~$34k-$37k) and the highest loyalty. 
   - Strategy: Retain through VIP treatment and early access, NOT deep 
     discounts, as they already buy at full price.

2. THE CASH COWS (Cluster 2):
   - Lower revenue but highly stable with the lowest complaint rate.
   - Strategy: Focus on increasing Average Order Value (AOV) via 
     product bundling and subscription models.

3. THE LOW-MARGIN DEAL HUNTERS (Clusters 3 & 4):
   - Highly promo-sensitive (up to 50% promo purchases), lowest loyalty 
     card adoption (<50%), and highest complaint rates.
   - Strategy: Minimize acquisition costs. Use automated marketing 
     (e.g., flash sales) and avoid spending premium resources here.
=======================================================================

"""
