# **The State of Retrieval-Augmented Forecasting (RAF): Architectural Paradigms, Industry Implementations, and Future Horizons**

## **1\. The Theoretical Crisis in Parametric Forecasting and the Retrieval Paradigm**

The field of Time Series Forecasting (TSF) is currently navigating a profound structural shift, moving from purely parametric methodologies—where historical patterns are compressed into fixed model weights—toward non-parametric, retrieval-augmented architectures. This transition is analogous to the Retrieval-Augmented Generation (RAG) revolution in Natural Language Processing (NLP), yet it addresses challenges specific to temporal data: non-stationarity, distribution shifts, and the inherent difficulty of "memorizing" rare, high-impact events within a finite parameter space.

### **1.1 The Memory Bottleneck in Deep Learning**

For the past decade, the dominant approach to forecasting has been parametric. Architectures ranging from Recurrent Neural Networks (RNNs) like LSTMs to modern Transformers (e.g., PatchTST, iTransformer) operate on the assumption that the laws governing future trajectories can be learned and stored within the network's weights during training. This "closed-world" assumption posits that the training data is a sufficient proxy for the future and that the model's capacity is large enough to encode all relevant temporal dynamics.1  
However, this approach faces a fundamental "Memory Bottleneck." Real-world time series data—whether in high-frequency financial markets, renewable energy grids, or retail supply chains—is vast and continuously expanding. Forcing a neural network to compress terabytes of historical data into a few gigabytes of floating-point weights results in lossy compression. The model inevitably prioritizes dominant, frequent patterns (such as daily seasonality or global trends) at the expense of rare, critical anomalies. When a model encounters a "black swan" event, such as a pandemic-induced supply shock or an extreme weather event, it often fails because the specific dynamics of that event were either seen too infrequently during training to be encoded or were overwritten by more recent, normative data—a phenomenon known as catastrophic forgetting.1

### **1.2 The Non-Stationarity Challenge**

The second failure mode of parametric models is their struggle with non-stationarity. In time series analysis, stationarity implies that the statistical properties of a process (mean, variance, autocorrelation) remain constant over time. Real-world systems are almost universally non-stationary. The distribution of electricity load in 2024 differs structurally from 2020 due to the proliferation of electric vehicles and solar panels. A model trained on 2020 data has learned the conditional probability distribution $P(Y\_{future} | X\_{past})$ for that specific regime. Applying this frozen distribution to 2024 data often leads to degradation in performance, as the underlying causal mechanisms have shifted.2  
Traditional solutions to this problem involve frequent retraining or fine-tuning. However, fine-tuning is computationally expensive and operationally brittle. It requires careful hyperparameter management to ensure the model adapts to new patterns without erasing necessary historical knowledge. In high-velocity environments like stock trading or real-time IoT monitoring, the time required to retrain a model may exceed the lifespan of the market opportunity it was meant to capture.3

### **1.3 The RAF Paradigm: Decoupling Memory from Computation**

Retrieval-Augmented Forecasting (RAF) proposes a radical restructuring of the forecasting pipeline. It decouples the "memory" (storage of historical patterns) from the "computation" (inference of future values). In an RAF system, the model is not expected to memorize history. Instead, it is granted access to an external, queryable Knowledge Base (KB) containing a vast repository of historical time series segments.  
When making a prediction, the RAF system queries this database to find historical instances that are semantically or structurally similar to the current input. It retrieves these instances—and crucially, their subsequent outcomes—and provides them to the forecasting model as "context." This shifts the inductive bias of the system: rather than relying solely on *learned abstractions*, the model can rely on *explicit precedent*. If a specific, rare volatility pattern occurred five years ago, the RAF system retrieves that exact event and uses its aftermath to inform the current prediction, bypassing the limitations of the model's weights.1  
This paradigm offers two distinct theoretical advantages:

1. **Infinite Memory Capacity:** The "memory" of the system is limited only by disk space, not parameter count. New data can be added to the Knowledge Base instantly without retraining the model, enabling near-instant adaptation to new regimes.  
2. **Explainability via Precedent:** Unlike a "black box" neural network output, an RAF forecast can be audited by examining the retrieved examples. An analyst can see that the model predicts a spike in energy demand because it retrieved three specific instances from previous heatwaves that exhibited identical load signatures.3

---

**2\. Architectural Advancements in 2024-2025**

The research landscape in 2024 and 2025 has been characterized by the rapid formalization of RAF architectures. While early attempts were ad-hoc, recent literature defines clear methodologies for how time series should be tokenized, stored, retrieved, and fused. The most significant advancements can be categorized into three architectural archetypes: **Patch-Based Augmentation**, **Foundation Model Wrappers**, and **Native In-Context Learners**.

### **2.1 RAFT: The Direct Augmentation Archetype**

Proposed by Han et al. at ICML 2025, **RAFT (Retrieval-Augmented Forecasting of Time Series)** represents the state-of-the-art in "Direct Augmentation." RAFT is designed to be model-agnostic, demonstrating that the injection of retrieved context allows even simple architectures (like MLPs) to outperform complex Transformers.1

#### **2.1.1 The Patching Mechanism**

A critical insight of RAFT is that point-wise retrieval is ineffective. Comparing a single time step $t$ to historical steps lacks semantic context. Instead, RAFT employs a "Patching" strategy. It segments the historical time series into sliding windows (patches).

* **Key ($K$):** A historical window of length $L$ (the lookback window).  
* **Value ($V$):** The subsequent window of length $F$ (the forecast horizon).

This Key-Value pairing is central to the strategy. Unlike NLP RAG, where the retrieved text is usually the text itself, in RAFT, the system retrieves the *outcome* of the pattern. The database stores the cause ($K$) linked to the effect ($V$).1

#### **2.1.2 Similarity Metrics: Pearson Correlation vs. Euclidean Distance**

RAFT deviates from standard vector search by utilizing **Pearson Correlation Coefficient (PCC)** as its primary similarity metric, rather than the Euclidean distance (L2) typically used in vector databases.

* **The Logic of PCC:** In many time series domains (e.g., finance), the *shape* of the curve matters more than the *magnitude*. A stock rising from $10 to $11 has the same trend shape as a stock rising from $100 to $110. Euclidean distance would penalize this match due to the absolute difference in values. Pearson correlation focuses on the linear relationship, effectively matching the "momentum" and "volatility" profile regardless of the price level. This allows RAFT to retrieve relevant patterns even after significant distribution shifts in the data's mean and variance.1

#### **2.1.3 The Fusion Mechanism**

Once the top-$k$ similar patches are identified, RAFT aggregates their corresponding Values (futures) into a "Prototype" forecast. This prototype is concatenated with the embedding of the current input window. A projection layer then learns to weigh the contribution of the model's internal processing against the external prototype. Empirical evaluations across ten benchmark datasets show RAFT achieving an average win ratio of 86% against contemporary baselines, largely due to its superior handling of rare, recurring motifs that standard Transformers fail to reconstruct.2

### **2.2 TS-RAG and TimeRAF: The Foundation Model Integrators**

While RAFT builds a specialized forecaster, **TS-RAG** (Ning et al., arXiv 2025\) and **TimeRAF** (Zhang et al., IEEE TKDE 2025\) focus on augmenting pre-trained **Time Series Foundation Models (TSFMs)**. TSFMs, such as TimeGPT or Lag-Llama, are trained on massive corpora of diverse time series. However, they often struggle with zero-shot generalization when applied to niche domains (e.g., specific hydrological sensors or proprietary retail data).3

#### **2.2.1 The Adaptive Retrieval Mixer (ARM)**

A major risk in RAG systems is "hallucination via retrieval"—if the current query has no good historical match, the retriever might return irrelevant noise, confusing the model. TS-RAG introduces the **Adaptive Retrieval Mixer (ARM)** to mitigate this.

* **Mechanism:** The ARM is a learnable gating module. It takes the query embedding and the retrieved context embeddings as input and outputs a confidence scalar $\\alpha$.  
* **Function:** If the retrieval distance is high (indicating a poor match), $\\alpha$ approaches 0, and the system suppresses the retrieved signal, forcing the foundation model to rely on its internal weights. If the match is strong, $\\alpha$ increases, blending the retrieved future horizon into the prediction. This dynamic gating is essential for robustness in zero-shot scenarios where the Knowledge Base might be sparse.3

#### **2.2.2 Channel Prompting and Learnable Retrievers**

**TimeRAF** advances the field by addressing multivariate time series. In complex systems (like a weather station), not all channels are equally relevant for retrieval. "Channel Prompting" treats each variate (e.g., temperature, pressure) as an independent query channel. The system retrieves history relevant to *temperature* separately from history relevant to *pressure*, then fuses these multi-modal contexts via a cross-attention mechanism. Furthermore, TimeRAF employs an end-to-end learnable retriever, optimizing the embedding space specifically for forecasting utility rather than generic similarity.9

### **2.3 Retrieval-Augmented Diffusion (RATD)**

Generative AI has also entered the forecasting domain via Diffusion Models. **RATD (Retrieval-Augmented Time series Diffusion)**, presented at NeurIPS 2024, addresses the instability of diffusion-based forecasting. Diffusion models generate forecasts by iteratively denoising random noise. Without strong guidance, this process can diverge. RATD retrieves similar historical sequences and uses them as a "condition" or anchor during the denoising steps. The retrieved reference guides the diffusion trajectory, ensuring the generated sample adheres to realistic temporal dynamics observed in the past. This approach has proven particularly effective for probabilistic forecasting, where the goal is to generate a distribution of possible futures rather than a single point estimate.11

---

**3\. Foundation Models and Native In-Context Learning**

Parallel to the development of explicit RAG architectures, the rise of **Time Series Foundation Models (TSFMs)** has introduced a form of "implicit" retrieval known as **In-Context Learning (ICL)**.

### **3.1 Chronos-2: The Universal Forecaster**

**Chronos-2**, released by Amazon Science in late 2024, treats time series forecasting as a language modeling problem. It quantizes continuous time series values into discrete tokens and processes them using a T5 encoder-decoder architecture. Unlike explicit RAF models that query a database, Chronos-2 relies on its massive context window (up to 8192 tokens) to perform retrieval.13

#### **3.1.1 Group Attention as Retrieval**

Chronos-2 introduces a mechanism called **Group Attention**. This allows the model to process multiple related time series simultaneously.

* **Scenario:** Forecasting the CPU load of Server A.  
* **Context:** The input includes the history of Server A, but also Server B and Server C (which are part of the same cluster).  
* **Mechanism:** Through Group Attention, the model can "attend" to the patterns in Server B and C to inform the forecast for Server A. If Server B exhibited a load spike 10 minutes ago, the model "retrieves" this information to predict a similar spike for Server A. This acts as a dynamic, in-memory retrieval system that exploits cross-series correlations without an external index.15

### **3.2 Chronos-Bolt: Speed and Efficiency**

Recognizing that the computational cost of Transformers can be prohibitive for real-time applications, Amazon released **Chronos-Bolt** in early 2025\. This variant utilizes a patch-based input strategy (similar to RAFT) but processes them within the transformer context.

* **Performance:** Chronos-Bolt is reported to be up to 250 times faster than the original Chronos model while maintaining comparable accuracy. This speed is achieved by reducing the sequence length through patching (tokenizing chunks of time steps rather than individual points).  
* **Application:** This speed-up is critical for high-frequency forecasting scenarios (e.g., programmatic advertising or algorithmic trading) where the latency of retrieval and inference must be kept in the millisecond range.16

### **3.3 Moirai and Any-Variate Attention**

Salesforce's **Moirai** model introduces "Any-Variate Attention," which allows the model to handle time series with arbitrary numbers of variables. While Moirai is primarily a foundation model, its architecture supports a form of retrieval by allowing users to include "variates" that serve purely as context. A user can append historical "reference" series as additional variates in the input, and Moirai's attention mechanism will learn to utilize them as predictors, effectively performing retrieval-augmented inference within the forward pass.18

---

**4\. Industry Implementation: The Pragmatics of Scale**

While academia refines the mathematics of retrieval, the industry is applying RAF to solve massive-scale engineering and business problems. The focus in industry is less on "beating benchmarks" and more on solving the "Cold Start" problem, managing data scale, and enabling semantic interaction with time series data.

### **4.1 Retail and Logistics: Solving the Cold Start Problem**

One of the most valuable applications of RAF is in handling new products or entities that lack historical data—the "Cold Start" problem.

#### **4.1.1 DoorDash: Taxonomy Retrieval and Entity Resolution**

**DoorDash** has deployed a sophisticated RAF system to manage its rapidly expanding grocery and retail catalog. When a new product (e.g., a specific brand of organic almond milk) is onboarded, it has no sales history, making traditional time series forecasting impossible.21

* **Taxonomy Mapping via RAG:** DoorDash uses RAG to map the new item's metadata (text description, image tags) to their existing "Golden Taxonomy." They generate vector embeddings for the new item and retrieve the nearest neighbors from their database of established products with known demand curves.  
* **Transfer Forecasting:** The sales patterns, seasonality, and "shelf life" characteristics of the retrieved neighbors are transferred to the new item. If the system retrieves "generic almond milk" as a neighbor, it uses that product's historical demand to initialize the forecast for the new organic brand.  
* **Entity Resolution:** They further utilize RAG to identify duplicate items across millions of SKUs (e.g., "12oz Coke" vs. "Coca-Cola 12 ounce can"). By cleaning the data entity layer using retrieval, they ensure that the time series history is aggregated correctly, preventing data fragmentation that would degrade forecast accuracy.23

#### **4.1.2 Walmart: Anomaly "Forgetting" and Supply Chain Flow**

**Walmart's** supply chain forecasting leverages the inverse of retrieval—selective forgetting. Their AI systems ingest vast amounts of "future data" (weather forecasts, local events) to predict demand. However, a key challenge is preventing one-off anomalies (e.g., panic buying before a hurricane) from corrupting the historical baseline used for future years.25

* **Mechanism:** Their system identifies these anomalous periods and effectively "masks" them in the retrieval process for year-over-year comparisons. Conversely, when a similar event (e.g., another hurricane) is forecast, the system can selectively "retrieve" the specific demand patterns from previous hurricanes to model the expected spike. This dynamic inclusion/exclusion of historical contexts represents a highly applied form of RAF.

### **4.2 Financial Services: Analog Forecasting**

The finance sector has long used "analog years" as a heuristic. RAF formalizes this into an automated, high-speed workflow.

#### **4.2.1 Pinecone and "Smart Trackers"**

Financial technology platforms leverage vector databases like **Pinecone** to implement "Pattern Retrieval."

* **Application:** A quantitative analyst wants to predict the movement of a stock that has just formed a specific technical pattern (e.g., a "Double Bottom" with decreasing volume).  
* **Vectorization:** The recent price and volume history is converted into a vector embedding.  
* **Retrieval:** The system queries the database to find the top-$k$ most similar patterns across all tracked assets over the last 20 years.  
* **Analog Forecasting:** The system retrieves the *subsequent* price movements of these $k$ historical instances. By aggregating these futures (e.g., calculating the mean return and volatility), the system generates a probabilistic forecast for the current stock. This allows traders to quantify the "success rate" of technical patterns dynamically based on empirical history rather than fixed rules.27

#### **4.2.2 Uber: Finch and Conversational Analytics**

**Uber** has integrated RAF principles into its financial intelligence tool, **Finch**. Built on top of their massive **M3** time series database, Finch allows finance teams to query data using natural language.29

* **Semantic-to-Temporal Retrieval:** When a user asks, "How does Q3 2024 profitability compare to the post-pandemic recovery period?", the system must identify (retrieve) the specific time window corresponding to "post-pandemic recovery" and the relevant profitability metrics. It then retrieves these time series segments and performs a comparative analysis.  
* **Significance:** This application highlights the role of RAG in bridging the semantic gap. It allows users to retrieve time series data based on *conceptual* descriptions ("recovery period," "high volatility") rather than just timestamps, democratizing access to complex forecasting insights.

### **4.3 Energy and Utilities: Anomaly Detection as Retrieval**

In the energy sector, forecasting and anomaly detection are two sides of the same coin. An anomaly is simply an observation that deviates significantly from the forecast.

#### **4.3.1 Vector-Based Anomaly Detection**

Companies utilizing **Milvus** and **Qdrant** have deployed RAF architectures for monitoring grid stability and IoT sensor networks.31

* **Normalcy Index:** They maintain a vector index of "normal" operating states (historical windows of voltage, frequency, and vibration that represented stable operation).  
* **Real-Time Query:** Live sensor streams are continuously vectorized and queried against this Normalcy Index.  
* **Distance-Based Classification:** The "Anomaly Score" is defined as the distance to the nearest neighbor in the index. If the closest historical match is far away, the state is flagged as anomalous.  
* **Forecasting Support:** For load forecasting, operators retrieve historical days with similar weather and calendar attributes (e.g., "A hot Tuesday following a holiday"). The load curves from these retrieved days form the basis (the "prior") for the current day's forecast, which is then refined by real-time adjustments.33

---

**5\. The Infrastructure Ecosystem: Enabling Scale**

The feasibility of RAF in 2025 is underpinned by a mature ecosystem of specialized **Vector Databases** that can handle the unique requirements of time series data (high dimensionality, rapid updates, and massive scale).

### **5.1 Vector Database Comparison for Time Series**

| Feature | Pinecone | Qdrant | Milvus |
| :---- | :---- | :---- | :---- |
| **Primary Use Case** | Financial Pattern Matching, SaaS Multi-tenancy | Anomaly Detection, Recommendation Systems | IoT Sensor Networks, Grid Monitoring |
| **Architecture** | Serverless, Pod-based | Rust-based, Container-native | Distributed, Cloud-native |
| **Key Capability** | **Namespaces:** Allows isolation of data for millions of users (e.g., separate stock portfolios), critical for B2B fintech apps.35 | **Recommendation API:** Supports positive/negative examples (e.g., "Find patterns like X but unlike Y"), essential for refining forecast retrieval.36 | **Hybrid Search:** Combines vector similarity with scalar filtering (e.g., "Find similar load curves *only* from Substation 5"), crucial for physical infrastructure.34 |
| **Throughput** | High concurrency for user-facing apps. | Extremely low latency (Rust) for real-time loops. | Massive scale (billions of vectors) for historical archives. |

### **5.2 The Convergence of SQL and Vector**

A significant trend in 2025 is the blurring of lines between traditional data warehouses and vector search. Platforms like **Snowflake** (via Cortex Search) and **Databricks** are integrating vector capabilities directly into their SQL engines.

* **Snowflake Cortex:** Enables analysts to perform RAG over operational data using SQL functions. An analyst can write a query that joins a standard sales table with a vector search function to "select similar past sales periods," streamlining the RAF workflow without needing a separate vector database.37  
* **Databricks:** Integrates Mosaic AI and vector search to allow energy companies (like Xcel) to build RAG agents that query both unstructured documents (regulatory PDFs) and structured time series logs within a unified "Data Intelligence Platform".39

---

**6\. Critical Analysis and Future Outlook**

### **6.1 The "Cheating" Debate and Data Leakage**

A recurring theoretical debate in RAF research concerns the boundary between "valid retrieval" and "data leakage." If a model retrieves a historical window that overlaps with the test set, it is effectively "cheating." SOTA architectures like RAFT and Chronos enforce strict temporal masking to prevent this. However, in industrial applications (like anomaly detection), retrieving the "future" of a past event is the entire point. The definition of "leakage" depends heavily on whether the goal is *generalization evaluation* (academic) or *accurate prediction* (industry).

### **6.2 The "Agentic" Future**

The next frontier, already visible in early 2025, is **Agentic RAF**. Current systems typically perform a single retrieval step. Agentic systems employ a "Master Agent" that orchestrates multiple retrieval sub-agents.

* **Scenario:** Forecasting Sales.  
* **Agent 1:** Retrieves similar historical sales patterns.  
* **Agent 2:** Retrieves relevant macroeconomic news (text) from that period.  
* **Agent 3:** Retrieves competitor pricing history.  
* **Synthesis:** The Master Agent synthesizes these multi-modal inputs into a coherent forecast. This approach, pioneered in research concepts like "Agentic RAG" and Uber's internal tools, promises to move forecasting from "pattern matching" to "reasoned analysis".29

### **6.3 Conclusion**

As of 2025, Retrieval-Augmented Forecasting has matured from a niche experiment to a dominant architectural paradigm. It has successfully addressed the "Memory Bottleneck" of deep learning by externalizing historical knowledge. The "Most Performant" systems today—**RAFT** in academia and **Chronos-2** in the foundation model space—share a common philosophy: they do not try to *learn* the entire universe of temporal dynamics. Instead, they learn how to *search* for the relevant dynamics in an infinite, updateable memory. With the infrastructure of vector databases now capable of handling billions of time series points, and major industry players like Uber and Walmart validating the approach at scale, RAF is poised to become the standard operating procedure for high-stakes, non-stationary forecasting problems.

---

**Table 1: Comparison of SOTA Retrieval-Augmented Forecasting Architectures (2025)**

| Architecture | Core Mechanism | Retrieval Target (Key/Value) | Primary Use Case | Performance Highlight |
| :---- | :---- | :---- | :---- | :---- |
| **RAFT** (Han et al.) | **Direct Augmentation** | Key: Past Patch / Value: **Future Patch** | General TSF, Rare Event Handling | **86% win rate** vs. baselines; excels at random-walk patterns.1 |
| **TS-RAG** (Ning et al.) | **Foundation Model Wrapper** | Key: Embedding / Value: Horizon | Zero-Shot Domain Adaptation | **Adaptive Retrieval Mixer (ARM)** prevents hallucination by gating retrieval.3 |
| **TimeRAF** (Zhang et al.) | **Learnable Retrieval** | Key: Channel / Value: Context | Multivariate / High-Dim Data | **Channel Prompting** allows independent retrieval per variable.8 |
| **Chronos-2** (Amazon) | **In-Context Learning** | Key: Token Sequence / Value: Next Token | Universal / Zero-Shot | **Group Attention** enables implicit retrieval from batch context.15 |
| **RATD** (Liu et al.) | **Guided Diffusion** | Key: Sequence / Value: Condition | Probabilistic / Generative TSF | Uses retrieval to stabilize the reverse diffusion process.11 |

### **Table 2: Industry Implementation Matrix**

| Company | Problem Space | Solution Mechanism | Infrastructure | Impact |
| :---- | :---- | :---- | :---- | :---- |
| **DoorDash** | **Cold Start** (New Products) | RAG maps new item metadata to existing "Golden Taxonomy" demand curves. | Internal Knowledge Graph | Enables forecasting for millions of new SKUs with zero history.21 |
| **Uber** | **Financial Analytics** | "Finch" agent maps natural language to M3 time series queries. | M3 DB, Orbit | Democratized access to complex financial forecasting for non-technical teams.29 |
| **Walmart** | **Supply Chain** | "Forgetting" mechanism masks anomaly periods (e.g., hurricanes) from retrieval. | Proprietary AI / Cloud | Prevents one-off events from skewing baseline inventory forecasts.26 |
| **Pinecone** | **Stock Trading** | "Analog Forecasting" retrieves similar chart patterns to predict future probability. | Pinecone Vector DB | Allows traders to quantify the success rate of technical patterns dynamically.27 |

### **Table 3: Retrieval Metrics Effectiveness Analysis**

| Metric | Best For | Weakness | Used By |
| :---- | :---- | :---- | :---- |
| **Pearson Correlation** | **Shape Matching** (e.g., Stock trends). Ignores amplitude differences. | Misses scale-dependent patterns (e.g., if absolute load causes grid failure). | **RAFT** 1 |
| **Euclidean (L2)** | **Latent Space Matching**. Works well with learned embeddings. | Sensitive to outliers and amplitude shifts. | **TS-RAG**, **TimeRAF** 42 |
| **DTW (Dynamic Time Warping)** | **Phase-Shifted Matching**. Handles temporal misalignment (e.g., a delayed reaction). | Computationally expensive ($O(N^2)$). | Specialized financial tools |
| **Learned Embeddings** | **Semantic Matching**. Maps disparate data (text \+ time) to shared space. | Requires massive training data; "black box" similarity. | **TimeGPT**, **Chronos** 43 |

### **Citations**

* **RAFT (Han et al.):** 1  
* **TS-RAG (Ning et al.):** 3  
* **TimeRAF (Zhang et al.):** 8  
* **RATD:** 11  
* **Chronos/TimeGPT:** 13  
* **Industry (Uber, DoorDash, Walmart):** 21  
* **Vector DBs (Pinecone, Qdrant, Milvus):** 27

#### **Works cited**

1. Retrieval Augmented Time Series Forecasting \- OpenReview, accessed on December 4, 2025, [https://openreview.net/forum?id=GUDnecJdJU](https://openreview.net/forum?id=GUDnecJdJU)  
2. Retrieval Augmented Time Series Forecasting \- arXiv, accessed on December 4, 2025, [https://arxiv.org/html/2505.04163v1](https://arxiv.org/html/2505.04163v1)  
3. TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster \- arXiv, accessed on December 4, 2025, [https://arxiv.org/html/2503.07649v4](https://arxiv.org/html/2503.07649v4)  
4. Retrieval Augmented Time Series Forecasting \- OpenReview, accessed on December 4, 2025, [https://openreview.net/pdf?id=GUDnecJdJU](https://openreview.net/pdf?id=GUDnecJdJU)  
5. TS RAG Retrieval Augment | PDF | Time Series | Forecasting \- Scribd, accessed on December 4, 2025, [https://www.scribd.com/document/915293857/13271-TS-RAG-Retrieval-Augment](https://www.scribd.com/document/915293857/13271-TS-RAG-Retrieval-Augment)  
6. Retrieval Augmented Time Series Forecasting \- Proceedings of Machine Learning Research, accessed on December 4, 2025, [https://proceedings.mlr.press/v267/han25d.html](https://proceedings.mlr.press/v267/han25d.html)  
7. TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster \- OpenReview, accessed on December 4, 2025, [https://openreview.net/pdf?id=TJuUelhGQr](https://openreview.net/pdf?id=TJuUelhGQr)  
8. TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot ..., accessed on December 4, 2025, [https://www.computer.org/csdl/journal/tk/2025/09/11031238/27uvv8498Uo](https://www.computer.org/csdl/journal/tk/2025/09/11031238/27uvv8498Uo)  
9. (PDF) TimeRAF: Retrieval-Augmented Foundation model for Zero-shot Time Series Forecasting \- ResearchGate, accessed on December 4, 2025, [https://www.researchgate.net/publication/387540687\_TimeRAF\_Retrieval-Augmented\_Foundation\_model\_for\_Zero-shot\_Time\_Series\_Forecasting](https://www.researchgate.net/publication/387540687_TimeRAF_Retrieval-Augmented_Foundation_model_for_Zero-shot_Time_Series_Forecasting)  
10. TimeRAF: Retrieval-Augmented Foundation model for Zero-shot Time Series Forecasting, accessed on December 4, 2025, [https://arxiv.org/html/2412.20810v1](https://arxiv.org/html/2412.20810v1)  
11. Retrieval-Augmented Diffusion Models for Time Series Forecasting \- NIPS papers, accessed on December 4, 2025, [https://papers.nips.cc/paper\_files/paper/2024/hash/053ee34c0971568bfa5c773015c10502-Abstract-Conference.html](https://papers.nips.cc/paper_files/paper/2024/hash/053ee34c0971568bfa5c773015c10502-Abstract-Conference.html)  
12. Retrieval-Augmented Diffusion Models for Time Series Forecasting \- NIPS papers, accessed on December 4, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf)  
13. amazon/chronos-2 \- Hugging Face, accessed on December 4, 2025, [https://huggingface.co/amazon/chronos-2](https://huggingface.co/amazon/chronos-2)  
14. Introducing Chronos-2: From univariate to universal forecasting \- Amazon Science, accessed on December 4, 2025, [https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)  
15. Chronos-2: From Univariate to Universal Forecasting \- arXiv, accessed on December 4, 2025, [https://arxiv.org/html/2510.15821v1](https://arxiv.org/html/2510.15821v1)  
16. Complete Amazon Chronos Guide for Production Time Series Forecasting \- Galileo AI, accessed on December 4, 2025, [https://galileo.ai/blog/amazon-chronos-ai-time-series-forecasting-guide](https://galileo.ai/blog/amazon-chronos-ai-time-series-forecasting-guide)  
17. amazon-science/chronos-forecasting: Chronos: Pretrained Models for Time Series Forecasting \- GitHub, accessed on December 4, 2025, [https://github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)  
18. MOIRAI: Salesforce's Foundation Model for Time-Series Forecasting, accessed on December 4, 2025, [https://towardsdatascience.com/moirai-salesforces-foundation-model-for-time-series-forecasting-4eff6c34093d/](https://towardsdatascience.com/moirai-salesforces-foundation-model-for-time-series-forecasting-4eff6c34093d/)  
19. Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts, accessed on December 4, 2025, [https://arxiv.org/html/2410.10469v1](https://arxiv.org/html/2410.10469v1)  
20. Moirai: A Time Series Foundation Model for Universal Forecasting \- Salesforce, accessed on December 4, 2025, [https://www.salesforce.com/blog/moirai/](https://www.salesforce.com/blog/moirai/)  
21. Using LLMs to infer grocery preferences from DoorDash restaurant orders, accessed on December 4, 2025, [https://careersatdoordash.com/blog/doordash-llms-for-grocery-preferences-from-restaurant-orders/](https://careersatdoordash.com/blog/doordash-llms-for-grocery-preferences-from-restaurant-orders/)  
22. How DoorDash labels Millions of Items with Large Language Models \- Quastor, accessed on December 4, 2025, [https://blog.quastor.org/p/doordash-labels-millions-items-large-language-models](https://blog.quastor.org/p/doordash-labels-millions-items-large-language-models)  
23. Unleashing the power of large language models at DoorDash for a seamless shopping adventure, accessed on December 4, 2025, [https://careersatdoordash.com/blog/unleashing-the-power-of-large-language-models-at-doordash-for-a-seamless-shopping-adventure/](https://careersatdoordash.com/blog/unleashing-the-power-of-large-language-models-at-doordash-for-a-seamless-shopping-adventure/)  
24. How DoorDash leverages LLMs for better search retrieval, accessed on December 4, 2025, [https://careersatdoordash.com/blog/how-doordash-leverages-llms-for-better-search-retrieval/](https://careersatdoordash.com/blog/how-doordash-leverages-llms-for-better-search-retrieval/)  
25. Walmart's U.S. Supply Chain Playbook Goes Global — and It's Reinventing Retail at Scale, accessed on December 4, 2025, [https://corporate.walmart.com/news/2025/07/17/walmarts-us-supply-chain-playbook-goes-global-and-its-reinventing-retail-at-scale](https://corporate.walmart.com/news/2025/07/17/walmarts-us-supply-chain-playbook-goes-global-and-its-reinventing-retail-at-scale)  
26. Decking the aisles with data: How Walmart's AI-powered inventory system brightens the holidays, accessed on December 4, 2025, [https://tech.walmart.com/content/walmart-global-tech/en\_us/blog/post/walmarts-ai-powered-inventory-system-brightens-the-holidays.html](https://tech.walmart.com/content/walmart-global-tech/en_us/blog/post/walmarts-ai-powered-inventory-system-brightens-the-holidays.html)  
27. Time Series Similarity Search \- Colab, accessed on December 4, 2025, [https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/analytics-and-ml/time-series/time-series-stocks-pattern-example.ipynb](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/analytics-and-ml/time-series/time-series-stocks-pattern-example.ipynb)  
28. Time Series Analysis Through Vectorization | Pinecone, accessed on December 4, 2025, [https://www.pinecone.io/learn/time-series-vectors/](https://www.pinecone.io/learn/time-series-vectors/)  
29. Unlocking Financial Insights with Finch: Uber's Conversational AI Data Agent | Uber Blog, accessed on December 4, 2025, [https://www.uber.com/blog/unlocking-financial-insights-with-finch/](https://www.uber.com/blog/unlocking-financial-insights-with-finch/)  
30. The Billion Data Point Challenge: Building a Query Engine for High Cardinality Time Series Data | Uber Blog, accessed on December 4, 2025, [https://www.uber.com/blog/billion-data-point-challenge/](https://www.uber.com/blog/billion-data-point-challenge/)  
31. Big Data Mining of Energy Time Series for Behavioral Analytics and Energy Consumption Forecasting \- MDPI, accessed on December 4, 2025, [https://www.mdpi.com/1996-1073/11/2/452](https://www.mdpi.com/1996-1073/11/2/452)  
32. Vector Database Use Cases \- Qdrant, accessed on December 4, 2025, [https://qdrant.tech/use-cases/](https://qdrant.tech/use-cases/)  
33. How can similarity search detect abnormal sensor readings in real-time? \- Milvus, accessed on December 4, 2025, [https://milvus.io/ai-quick-reference/how-can-similarity-search-detect-abnormal-sensor-readings-in-realtime](https://milvus.io/ai-quick-reference/how-can-similarity-search-detect-abnormal-sensor-readings-in-realtime)  
34. What is Milvus? \- Tessell, accessed on December 4, 2025, [https://www.tessell.com/blogs/what-is-milvus](https://www.tessell.com/blogs/what-is-milvus)  
35. Reimagining the vector database to enable knowledgeable AI \- Pinecone, accessed on December 4, 2025, [https://www.pinecone.io/blog/serverless-architecture/](https://www.pinecone.io/blog/serverless-architecture/)  
36. RAG Use Case: Advanced Vector Search for AI Applications \- Qdrant, accessed on December 4, 2025, [https://qdrant.tech/rag/](https://qdrant.tech/rag/)  
37. Cortex Search | Snowflake Documentation, accessed on December 4, 2025, [https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview)  
38. Streamlining Support Case Analysis with Snowflake Cortex, accessed on December 4, 2025, [https://www.snowflake.com/en/developers/guides/streamlining-support-case-analysis-with-snowflake-cortex/](https://www.snowflake.com/en/developers/guides/streamlining-support-case-analysis-with-snowflake-cortex/)  
39. Xcel Energy: Developing a RAG-based Chatbot on Databricks, accessed on December 4, 2025, [https://www.databricks.com/blog/xcel-energy-rag](https://www.databricks.com/blog/xcel-energy-rag)  
40. Data Intelligence in Action: 100+ Data and AI Use Cases from Databricks Customers, accessed on December 4, 2025, [https://www.databricks.com/blog/data-intelligence-action-100-data-and-ai-use-cases-databricks-customers](https://www.databricks.com/blog/data-intelligence-action-100-data-and-ai-use-cases-databricks-customers)  
41. Agentic Retrieval-Augmented Generation for Time Series Analysis \- arXiv, accessed on December 4, 2025, [https://arxiv.org/html/2408.14484v1](https://arxiv.org/html/2408.14484v1)  
42. Actions · kutaytire/Retrieval-Augmented-Time-Series-Forecasting \- GitHub, accessed on December 4, 2025, [https://github.com/kutaytire/Retrieval-Augmented-Time-Series-Forecasting/actions](https://github.com/kutaytire/Retrieval-Augmented-Time-Series-Forecasting/actions)  
43. TimeGPT: Revolutionizing Time Series Forecasting \- Analytics Vidhya, accessed on December 4, 2025, [https://www.analyticsvidhya.com/blog/2024/02/timegpt-revolutionizing-time-series-forecasting/](https://www.analyticsvidhya.com/blog/2024/02/timegpt-revolutionizing-time-series-forecasting/)  
44. \[2505.04163\] Retrieval Augmented Time Series Forecasting \- arXiv, accessed on December 4, 2025, [https://arxiv.org/abs/2505.04163](https://arxiv.org/abs/2505.04163)  
45. TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster \- arXiv, accessed on December 4, 2025, [https://arxiv.org/html/2503.07649v1](https://arxiv.org/html/2503.07649v1)  
46. UConn-DSIS/TS-RAG: Official code and models for Time Series RAG \- GitHub, accessed on December 4, 2025, [https://github.com/UConn-DSIS/TS-RAG](https://github.com/UConn-DSIS/TS-RAG)  
47. TimeRAG: Boosting LLM Time Series Forecasting via Retrieval-Augmented Generation, accessed on December 4, 2025, [https://www.researchgate.net/publication/390537701\_TimeRAG\_Boosting\_LLM\_Time\_Series\_Forecasting\_via\_Retrieval-Augmented\_Generation](https://www.researchgate.net/publication/390537701_TimeRAG_Boosting_LLM_Time_Series_Forecasting_via_Retrieval-Augmented_Generation)  
48. TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting | Request PDF \- ResearchGate, accessed on December 4, 2025, [https://www.researchgate.net/publication/392634340\_TimeRAF\_Retrieval-Augmented\_Foundation\_Model\_for\_Zero-Shot\_Time\_Series\_Forecasting](https://www.researchgate.net/publication/392634340_TimeRAF_Retrieval-Augmented_Foundation_Model_for_Zero-Shot_Time_Series_Forecasting)  
49. \[2411.08249\] Retrieval Augmented Time Series Forecasting \- arXiv, accessed on December 4, 2025, [https://arxiv.org/abs/2411.08249](https://arxiv.org/abs/2411.08249)  
50. \[2403.07815\] Chronos: Learning the Language of Time Series \- arXiv, accessed on December 4, 2025, [https://arxiv.org/abs/2403.07815](https://arxiv.org/abs/2403.07815)  
51. Quickstart Guide \- TimeGPT Foundational model for time series forecasting and anomaly detection \- Nixtla, accessed on December 4, 2025, [https://www.nixtla.io/docs/forecasting/timegpt\_quickstart](https://www.nixtla.io/docs/forecasting/timegpt_quickstart)  
52. Distributed Forecasting with Spark, Dask & Ray \- TimeGPT Foundational model for time series forecasting and anomaly detection \- Nixtla, accessed on December 4, 2025, [https://www.nixtla.io/docs/forecasting/forecasting-at-scale/computing\_at\_scale](https://www.nixtla.io/docs/forecasting/forecasting-at-scale/computing_at_scale)  
53. Using Machine Learning to Ensure the Capacity Safety of Individual Microservices \- Uber, accessed on December 4, 2025, [https://www.uber.com/blog/machine-learning-capacity-safety/](https://www.uber.com/blog/machine-learning-capacity-safety/)
