# Data Access — CGMacros Dataset

The CGMacros dataset is **not included in this repository**. All raw and processed files are excluded via `.gitignore` in compliance with PhysioNet Terms of Use and GDPR regulations.

Access requires completing the following steps in order.

---

## Step 1 — Create a PhysioNet account

1. Go to [https://physionet.org](https://physionet.org)
2. Click **Sign Up** and complete the registration form
3. Verify your email address

---

## Step 2 — Complete the CITI Program certification

PhysioNet requires proof of human subjects research ethics training before granting access to restricted datasets.

1. Go to [https://about.citiprogram.org](https://about.citiprogram.org)
2. Create a CITI account — **use the same email address as your PhysioNet account**
3. For institutional affiliation, select **"Independent Learner"** if your institution is not listed
4. Enroll in the course: **"Data or Specimens Only Research"**
5. Complete all required modules and pass the final assessment
6. Download your completion certificate (PDF)

---

## Step 3 — Link your CITI certification to PhysioNet

1. Log in to your PhysioNet account
2. Navigate to **Settings → CITI Program Certification**
3. Upload the PDF certificate obtained in Step 2
4. Wait for validation — this typically takes a few hours to a few days

---

## Step 4 — Request access to CGMacros

1. Go to the dataset page:
   [https://physionet.org/content/cgmacros/1.0.0/](https://physionet.org/content/cgmacros/1.0.0/)
2. Click **"Request Access"**
3. Read and sign the **Data Use Agreement (DUA)**
4. Once approved, download the dataset files

---

## Step 5 — Set up the local data directory

Once downloaded, place the raw files as follows:

```
data/
├── raw/
│   ├── bio.csv
│   ├── CGMacros-001.xlsx
│   ├── CGMacros-002.xlsx
│   └── ...
└── processed/                  ← generated automatically by the pipeline
    ├── bio_with_group.csv
    ├── meal_windows_dataset.csv
    └── column_description_meal_window.csv
```

Then run the pipeline to generate the processed files:

```bash
python src/build_patient_table.py
python src/meal_window_builder.py
```

---

## Important notices

- **Do not share or redistribute** the dataset — this is a strict condition of the PhysioNet Data Use Agreement.
- **Do not commit data files** to this repository. The `.gitignore` is configured to exclude `data/raw/` and `data/processed/*.csv`. Verify with `git status` before every push.
- Any publication or public use derived from this dataset must cite the original source:

> Bent, B., Henriquez, C., & Dunn, J. (2021). *CGMacros* (version 1.0.0). PhysioNet. [https://doi.org/10.13026/ceh6-4x45](https://doi.org/10.13026/ceh6-4x45)
