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

📊 Dataset: Real-World Labeled TTP Logs
Logs were collected from various endpoints across the UPC network and stored in Elasticsearch. Then:

- Splunk alerts were used to label malicious behavior.
- For each alert, we scanned a 10-minute window after the alert's Last_Ocurrence.
- We matched key fields like subject.account_name with the alert's user to label related logs.




```
cd existing_repo
git remote add origin https://gitlab.i2cat.net/areas/cybersecurity/tfm/tfg-marcel.git
git branch -M main
git push -uf origin main
```

## Authors and acknowledgment
**Author **: Marcel Fernández Serrano
- UPF Supervisor: Euan McGill
- i2cat Supervisor : Nil Ortiz
## License

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
