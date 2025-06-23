## 🛡️ Real-Time Cyber Threat Detection Using a Mixture of Experts Model (TTP-based) 🛡️

This repository contains the codebase for my final thesis project in Mathematical Engineering in Data Science at UPF.

The project introduces a modular Mixture of Experts (MoE) deep learning model, where each expert specializes in recognizing a specific cyber Tactic, Technique, or Procedure (TTP). It operates on real system log data collected from university endpoints and labeled using Splunk alerts. The goal is to build a more scalable, interpretable, and modular detection framework for real-world cyber threats.


## 📂 Repository Structure

```
.
├── create_dataset/        
|    ├── create_dataset.py 
|    |        ├── get_alerts             # Step 1 : To create a df of Alert_Name/User/Last_Ocurrence     
|    |        ├── fetch_logs_by_time     # Step 2 : Create a dataset of a whole day 
|    |        ├── label_logs_with_alerts # Step 3 : Take Step 1 + Step 2 and take the matchs + time windows labeling
|    |        ├── + auxliar functions    # Extra : To connect with elastic, for some format, filtering, etc...
|    |        └── parquet_files/  
|    |             ├── non_label_data/   # To store non_label_data   (Not public data)
|    |             ├── label_data/       # To store label data       (Not public data)
|    |             └── clean_data.py     # Apply preprocessing steps    
|    ├── merge_datasets.py
|    |     ├── merge_parquet_files       # Step 4 : Merge datasets (reference to label_data )
|    |     └── undersample_benign_logs   # Step 5 : Apply undersampling to result dataset (To balance the dataset)
|    └── q_db.py                         # External queries that I need for Step 1-3
|   
├── model/   
|    └── moe_v3/
|            ├── data_loader.py               # Data Loader for each expert
|            ├── full_dataset.py              # Data Loader for the gate network
|            ├── inference_dataset.py         # Data Loader
|            ├── experts.py                   # Experts logic
|            ├── gn.py                        # Gate network logic
|            ├── moe.py                       # MoE structure logic
|            ├── split_dataset_normal.py      # Step 6: Split data (train/test/validation) (balance) 
|            ├── split_dataset_realistic.py   # Split data (train/test/validation) realistic scenario (unbalance)
|            ├── train_experts.py             # Step 7: pretrain the experts indivudually 
|            ├── train_gn.py                  # Step 8: pretrain the gn with pretrained experts
|            ├── unseen_data.py               # Step 9: Evaluate the whole moe with unseen data (test df)
|            └── data/
|                  ├── expert_{i}.pt          # pretrained experts          (not public)
|                  ├── gating_network.pt      # pretrained gating network   (not public)
|                  ├── df_train.parquet       # (not public)
|                  ├── df_validation.parquet  # (not public)
|                  ├── df_test.parquet        # (not public)
|                  ├── expert_outputs.npy     # (not public)
|                  ├── gn_outputs.npy         # (not public)
|                  └── final_preds.npy        # (not public)
├── requirements.txt                          # Python dependencies
└── README.md 
```

## 📊 Dataset: Real-World Labeled TTP Logs
Logs were collected from various endpoints across the UPC network and stored in Elasticsearch. Then:

- Splunk alerts were used to label malicious behavior.
- For each alert, I scanned a X-minute window after the alert's Last_Ocurrence.
- I matched key fields like subject.account_name with the alert's user to label related logs.

## 🧠 MoE - Solution design
**Experts** are simple Multi-Layer Perceptrons (MLPs) trained as binary classifiers. Each expert has its own dedicated dataloader, allowing the model to specialize individual experts for specific cyberattacks (TTPs). The first step of the training process involves training all experts in parallel and monitoring their individual losses.

The **Gating Network** acts as a router and plays a central role in the architecture. Given a specific input, it determines which experts should be activated. It outputs a vector of scores or probabilities, which are used to decide expert activation.

There are several strategies to select experts based on these scores:

- A common approach is to activate the **top-1 expert** (i.e., the one with the highest probability).
-Another strategy is to select the **top-K experts**, which is often used in NLP domains. This allows multiple experts to contribute to the final decision, improving **robustness** and averaging out individual prediction errors.

While top-K gating is popular in large-scale language models, it has not been extensively explored in cybersecurity applications. In this project, I experimented with different gating strategies and ultimately adopted a **threshold-based activation** approach. This method activates all experts whose gating score exceeds a tunable threshold.

Threshold gating offers several benefits:

- It makes the model more input-sensitive and computationally efficient, as the number of active experts varies depending on the input.
- It provides greater interpretability, since expert activation is directly tied to a known threshold.
- It allows for fine control over the computational budget, enabling faster inference when fewer experts are activated.

This design choice proved effective for my use case, but the gating logic can be easily modified or extended depending on the application or experimental needs.

## Experiments

### **First Experiment:** Baseline Inference with Imbalanced Dataset
In this experiment, I performed inference using dataset_1, which contains 72.2% benign logs and 27.8% malicious logs (distributed evenly across 4 attacks, ~7% each).

- 📈 Result: The model achieved an average F1-score of 72%.

At first glance, this result seemed promising. However, upon closer inspection and debugging, I realized that the model was predicting only the benign class for nearly all inputs.

The root cause was an imbalance in the training data used for the gating network (70% benign / 30% malicious), which led the gate to consistently route inputs to the benign expert. As a result, only one expert was being activated, introducing a strong bias in the model's behavior.

While the F1-score appears acceptable, it is misleading—the model fails to perform its primary task: detecting cyberattacks. Simply put, the model "cheats" by defaulting to the majority class, which is not acceptable in a cybersecurity context where the goal is to catch threats, not just optimize a global metric.

This highlighted the need to rethink the gating strategy and address class imbalance, ensuring that the model learns to activate the appropriate expert(s) for malicious behavior.

### **Second Experiment: Training the Gate Network with a Balanced Dataset**
To address the expert selection bias observed in the first experiment, I trained the gate network using a balanced dataset. I applied a hard undersampling strategy to reduce the majority (benign) class, resulting in a dataset where all classes were equally represented: 20% benign and 20% per attack.

In this experiment, I also evaluated the model on a balanced test set. This setup ensured that:
- The gate network was exposed equally to all classes during training.
- All experts had an equal opportunity to learn and be activated.
- The test performance was not artificially inflated by class imbalance.

✅ Outcome:

- The model produced balanced predictions.
- All experts were being used appropriately.
- Most importantly, the model was now able to correctly detect attacks, which was the primary objective.

🔁 Reflection:
While this setup validated that the architecture was capable of learning meaningful distinctions between classes, it also introduced an artificial scenario:
> In real-world cybersecurity applications, benign events are far more frequent than attacks.

Therefore, although training with a balanced dataset is a reasonable strategy to promote fair learning, a more realistic evaluation approach would be to test the model on an imbalanced dataset that reflects true operational conditions. This would provide a better understanding of the model's effectiveness and robustness in practice, especially in detecting rare but critical attack events.

### **Third Experiment:**
In this final experiment, I maintained the strategy of training the model on a balanced dataset to ensure fair learning across all classes. However, the key difference was in the evaluation phase:
The model was tested on a **realistic, imbalanced dataset**, where benign logs dominate—mimicking an actual production environment. This setup is more aligned with real-world cybersecurity scenarios, where attacks are rare but critical to detect.

🎯 **Objective**:
To validate whether the model, trained in a balanced way, could still perform accurately under **real-world conditions—correctly** identifying rare attacks while handling a flood of benign data.

✅ **Results**:
- The model achieved strong overall performance under this imbalanced test setup.
- It successfully detected most attacks, indicating that the training strategy generalized well.
- The main source of error was the presence of false positives, where some benign logs were incorrectly labeled as malicious (specifically, misclassified as class 4—the benign expert).

⚠️ **Interpretation**:
While false positives are not ideal, in the context of cybersecurity, this is often a preferable trade-off. It is far more acceptable to raise a false alarm than to miss a true attack, which could have serious consequences.
This experiment demonstrated that the model is not only capable of learning from balanced data but is also **robust** and **reliable** when faced with imbalanced, real-world data distributions.


## Authors and acknowledgment
**Author**: Marcel Fernández Serrano
- UPF Supervisor: Euan McGill
- i2cat Supervisor : Nil Ortiz
