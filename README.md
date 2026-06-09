# Report: Customer Segmentation and Marketing Strategy. Group 20

**Authors:** Vitalii Hvozd (20240647), Tymur Novikov (20240649), Ekaterina Selikhovkina (20231788)

**Repository:** https://github.com/Sel-Kata/Project_ML2_Final

## 1. Data Preprocessing & Feature Engineering

To ensure the high-quality performance of clustering algorithms (which rely on distance calculations), the raw data underwent a rigorous transformation and cleaning stage:

* **Feature Engineering:** The `customer_birthdate` attribute was converted into a continuous numerical variable, `age`. The `loyalty_card_number` identifier was binarized into `has_loyalty_card` (1/0) to prevent the ordinal magnitude of random card numbers from skewing distance metrics.
* **Hybrid Missing Value Imputation:** Missing values in the `age` column were filled with the median to ensure robustness against potential outliers. For financial and behavioral features, we utilized a **`KNNImputer` (k=5)**. Unlike global mean imputation, this approach preserves the natural variance and covariance structure of the high-dimensional space by reconstructing missing data based on the 5 nearest neighbors for each customer vector.
* **Anomaly Handling & Scaling:** We applied sanity checks (capping hours at 23, promo purchases at 100%, and age between 14 and 100). Extreme financial outliers were capped at the 99th percentile. The final feature matrix was standardized using `StandardScaler` to ensure the correct calculation of Euclidean distances.

## 2. Clustering Models & Quality Evaluation

The search for the optimal number of clusters was conducted using the Elbow Method and Silhouette Score analysis, both indicating a global optimum at **k=5**.
<img width="974" height="299" alt="Elbow method and silhouette score" src="https://github.com/user-attachments/assets/56ef8747-41de-4694-99fb-0d1cef814272" />

* **Algorithm Selection:** Three different models were trained and compared: a centroid-based model (K-Means), a topological neural network (Self-Organizing Maps, 5x1 grid), and a connectivity-based model (Hierarchical / Agglomerative clustering). SOM was kept as the final segmentation model; hierarchical clustering was used as a benchmark only.
* **Hierarchical Benchmark:** We compared `ward`, `complete` and `average` linkage (dendrograms computed on a 5,000-customer sample, with the chosen `ward` solution refit on the full dataset). `ward` produced the most balanced and cohesive clusters and agreed with the K-Means/SOM structure, confirming that the segmentation is stable across very different algorithms.
* **Quality Metrics:** The results were evaluated using internal metrics: *Silhouette Score*, *Davies-Bouldin Index*, and *Calinski-Harabasz Index*. SOM demonstrated high stability and excellent business interpretability of the formed segments.
* **Dimensionality Reduction:** To visually confirm cluster separability in a high-dimensional space (20+ features), we applied PCA and UMAP (using `n_neighbors` of 15 and 50 for local and global topological analysis, respectively).
 <img width="910" height="574" alt="PCA and UMAP projection of SOM clusters" src="https://github.com/user-attachments/assets/95d20ca6-8dfb-42a0-8bcc-6aa4175581ba" /> 


## 3. Market Basket Analysis (Recommendation System)

To uncover joint purchase patterns within each cluster, we applied the **Apriori** algorithm. The filtering parameters were carefully selected to maximize business efficiency during cross-selling:

* **`min_support = 0.02`**: An optimal threshold that allowed us to filter out statistical noise while retaining non-obvious niche patterns (preventing the output from being skewed purely toward trivial rules like "bread $\rightarrow$ milk").
* **`min_confidence = 0.2`**: A strict threshold for e-commerce, ensuring that the probability of a cross-sell trigger firing (20%+) justifies the marketing costs of an email or banner ad.
* **`lift > 1.2`**: A condition guaranteeing that the co-occurrence of items exceeds their mathematical expectation of independent selection by at least 20%, thereby eliminating false-positive correlations.

---

## 4. Formed Cluster Profiles

Integrating the results of SOM and Apriori allowed us to identify 5 distinct behavioral segments:

### **Cluster 0: "Breakfast Lovers" (Large Families)**
* *Profile:* Average age 58.3, Kids at home 3.14 (highest overall). High loyalty (68.2%). Peak shopping hour: 10:00.
* *Basket:* Butter + oatmeal $\rightarrow$ milk (Lift = 1.32); Fresh bread + honey $\rightarrow$ tea.


### **Cluster 1: "Adult Gift Buyers" (Tech Focus)**
* *Profile:* Average age 60.3. Maximum loyalty (73.1%). Peak shopping hour: 11:00.
* *Finance & Stability:* LTV = **$34,411**. Stability Index = **84.2** (Most reliable customers). Average electronics spend: **$2,464**, energy drinks: **$542**.
* *Basket (2 rules):* Energy drinks $\rightarrow$ AirPods (Lift = 2.16); Bluetooth headphones $\rightarrow$ AirPods.


### **Cluster 2: "Families with Pets"**
* *Profile:* Average age 58.2, Kids at home 1.03. Minimum complaints (0.72 - most conflict-free).
* *Finance & Stability:* LTV = **$17,481**. Stability Index = **70.1**. Promo purchases: only **14%** (willing to pay full price).
* *Basket (40 rules):* Baby food $\rightarrow$ paper napkins (Lift = 2.07); Dog food $\rightarrow$ napkins (Lift = 1.96).


### **Cluster 3: "Healthy & Vegans"**
* *Profile:* Youngest segment (48.6). Peak shopping hour: 12:00. Promo purchases: **50%** (highly price-sensitive).
* *Finance & Stability:* LTV = **$16,453** (Lowest check). Stability Index = 65.2.
* *Basket (102 rules):* Avocado + asparagus $\rightarrow$ salad (Confidence $\approx$ 40%, Lift = 3.92).


### **Cluster 4: "Tech & Athletes"**
* *Profile:* Average age 57.2. Peak shopping hour: 17:00 (drop by after work). Visited stores: **1.5** (dislike browsing offline). Loyalty: **48.4%** (lowest). Complaints: **1.09** (highest).
* *Finance & Stability:* LTV = **$24,026**. Stability Index = **57.4** (highest risk). Promo purchases: 27%.
* *Basket (1375 rules):* iPhone 10 $\rightarrow$ AirPods (Lift = 3.34); Protein bar + deodorant $\rightarrow$ shampoo (Lift = 1.74).
<img width="688" height="609" alt="Снимок экрана 2026-06-08 в 16 13 03" src="https://github.com/user-attachments/assets/5d9785f3-0b05-425e-ab14-3a69feae434f" />



---

## 5. Business Insights & Advanced Analytics

### 5.1. The Demographic Paradox of Cluster 1

Initial analysis of Cluster 1's association rules (headphones, energy drinks, AirPods) pointed towards a youth segment. However, the age distribution analysis revealed a normal distribution with a mean of $\approx$ 60 years old.
A deep dive showed that 75% of customers in this cluster have teens or kids at home. Additional variance analysis of the `lifetime_spend_electronics` metric revealed rare but extremely high spikes (one-time purchases up to $10,000+).
**Conclusion:** This cluster represents an older generation buying expensive gifts for Generation Z. This requires a radical shift in communication strategy (from "youth-oriented" to "gifts for loved ones").

### 5.2. Customer Portfolio Matrix (Profitability vs. Stability)

To optimize retention budgets, we constructed a synthetic metric, `Stability Score`, combining loyalty, promo purchase percentage, and complaint frequency. Clusters were projected onto a 2D plane (LTV vs. Stability):

* **Stars (Clusters 0 and 1):** Highest LTV (>$34k) and maximum stability. Highly independent of promo actions.
* **Cash Cows (Cluster 2):** Extremely stable, moderate LTV ($17k). Optimal for Upsell strategies.
* **At-Risk / Deal Hunters (Clusters 3 and 4):** Low stability, high share of complaints and promo-driven purchases. Require minimization of servicing costs.
  <img width="988" height="763" alt="Customer portfolio matrix" src="https://github.com/user-attachments/assets/2ca43edc-630e-4f16-a3a7-3cd2321d82e5" />


---

## 6. Final Marketing Strategies (Action Plan)

* **Cluster 0 (Stars / Breakfasts):** Implement Cross-sell combos ("Eggs + Butter = 20% off Bread"). Automate email newsletters at 09:00 (one hour before their transaction peak).
* **Cluster 1 (Stars / Gifts):** Launch targeted "Holiday Tech Gifts" campaigns. Eliminate direct monetary discounts in favor of Value-Added Services (VIP treatment, product bundling with energy drinks).
* **Cluster 2 (Cash Cows):** Transition to a subscription model, with automated reminders to restock baby food and dog food every 3 weeks with a bonus for auto-pay.
* **Cluster 3 (Deal Hunters / Vegans):** Given their low brand loyalty and high price elasticity, deep discounts (e.g., 20% off avocados) should be granted **strictly** in exchange for registering in the loyalty app (Acquisition strategy).
* **Cluster 4 (At-Risk / Tech):** Set up an automated e-commerce trigger: a coupon for AirPods when a smartphone is added to the cart. In response to their high complaint rate, implement an automated Retention strategy that sends apology promo codes for a "Sports Bundle" following negative feedback.

**Next Steps:** Launch A/B testing of the proposed strategies.
