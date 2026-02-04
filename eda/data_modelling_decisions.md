# Data Modeling Decisions: Diabetes Readmission Prediction

## 1. Unit of Analysis: Individual Hospital Encounters

**Decision**: Each row (hospital encounter) is treated as an independent prediction instance.

**Rationale**: 
The clinical question we're addressing is: "For this specific hospital admission, what is the risk that this patient will be readmitted within 30 days?" This is an encounter-level prediction problem, not a patient-level one. Healthcare providers need to assess readmission risk at the point of discharge for each individual hospitalization to determine appropriate follow-up care, discharge planning, and intervention strategies. The same patient might have different risk profiles across different encounters depending on the diagnoses, treatments, and circumstances of each admission. Therefore, modeling at the encounter level provides actionable, timely predictions that align with clinical decision-making workflows.

---

## 2. Train/Test Split Must Be Done by Patient ID

**Decision**: Split data by unique patient IDs, ensuring all encounters for a given patient appear exclusively in either the training set or test set, never both.

**Rationale**: 
This dataset contains multiple encounters per patient over a 10-year period. If we perform a random row-level split, the same patient could appear in both training and test sets with different encounters. This creates **data leakage** because the model can learn patient-specific patterns during training (e.g., a particular patient's tendency to be readmitted, their specific combination of demographics and comorbidities) and then exploit that knowledge when making predictions on that same patient's encounters in the test set. This artificially inflates performance metrics and produces overly optimistic results that won't generalize to truly unseen patients in production. By splitting on patient ID, we ensure the model is evaluated on its ability to predict outcomes for patients it has never seen before, which reflects real-world deployment conditions.

---

## 3. Historical Features Require Verification to Prevent Future Leakage

**Decision**: Variables like `number_inpatient`, `number_outpatient`, and `number_emergency` should only be used if we can confirm they represent counts **up to and including** the current encounter, not total counts across all time.

**Rationale**: 
Without temporal information (admission dates) in this dataset, we cannot independently verify the temporal ordering of encounters for each patient. If these count variables include future visits that occurred after the current encounter, using them as features would constitute **future leakage**—using information that wouldn't be available at prediction time. The model would be learning from the future to predict the future. Before including these features, we must carefully review the data documentation to confirm they represent historical counts only. If documentation is unclear or unavailable, the safest approach is to exclude these features entirely, relying only on encounter-specific information that is definitively known at the time of the current admission.

---

## 4. Discharge Disposition Cannot Be Used as a Feature

**Decision**: Exclude `discharge_disposition_id` from the feature set used for modeling.

**Rationale**: 
The discharge disposition describes where the patient went after leaving the hospital (home, transferred to another facility, etc.). This information is only known **after** the discharge has occurred. However, readmission risk predictions need to be made **before or at the time of discharge** to enable interventions like enhanced discharge planning, follow-up appointment scheduling, or home health services. At the point when a clinical decision needs to be made, the discharge disposition is not yet available—it's part of the outcome, not a predictor. Using it as a feature would create **temporal leakage**: the model would perform well in historical analysis but fail catastrophically in production because the required input wouldn't exist at prediction time. Note that we still use discharge disposition for **data filtering** (removing deceased patients, etc.), which is valid preprocessing, but it must not be included as a model input.

---

## 5. Deceased Patients Must Be Removed from the Dataset

**Decision**: Exclude all encounters where the patient died (discharge dispositions indicating expired, hospice, or deceased status).

**Rationale**: 
The target variable is whether a patient is readmitted within 30 days. Deceased patients have **zero probability** of being readmitted—it is structurally impossible. Including them in the dataset creates several problems:

- **Artificial performance inflation**: They are trivially easy to classify as "not readmitted," inflating accuracy metrics without providing any useful predictive signal
- **Misaligned learning objective**: The model would learn to predict death rather than readmission risk among living patients
- **Clinical irrelevance**: The population of interest for readmission prediction is patients who are discharged alive with the opportunity to return

We are interested in identifying high-risk patients among those who can actually be readmitted, so deceased patients represent out-of-scope cases that should be excluded during data cleaning.

---

## 6. Newborn Cases Must Be Removed from the Dataset

**Decision**: Exclude encounters with newborn-related admission sources (codes 11-14, 23-24) or newborn admission type (code 4).

**Rationale**: 
This is a diabetes readmission prediction model for an adult population. Newborn admissions represent a fundamentally different population with:

- **Different physiology and risk factors**: Neonatal care has nothing to do with chronic diabetes management, diabetic medications, A1C levels, or adult comorbidities
- **Irrelevant outcomes**: A newborn's readmission is driven by entirely different clinical factors (prematurity, birth complications, maternal health) unrelated to the diabetes management patterns we're trying to model
- **Scope mismatch**: Even if the mother has gestational diabetes, the newborn's encounter is not informative for predicting adult diabetic readmissions

Including newborns would introduce noise and confound the model with irrelevant patterns. These cases fall outside the target population and clinical use case, so they should be filtered out during preprocessing.

---

## 7. "Still in System" Cases Are Ambiguous and Can Be Removed

**Decision**: Exclude encounters with "STILL_IN_SYSTEM" discharge disposition (codes 9, 12, 15, 16, 17).

**Rationale**: 
These discharge codes represent ambiguous situations:

- **Code 9**: "Admitted as inpatient to this hospital" - unclear if this is a transfer within the same facility or ongoing care
- **Code 12**: "Still patient or expected to return for outpatient services" - suggests the episode of care hasn't truly ended
- **Codes 15-17**: Internal transfers or outpatient referrals - blur the line between discharge and continuation of care

The ambiguity creates problems:

- **Unclear discharge timing**: When does the 30-day readmission window actually start if the patient is still receiving care?
- **Invalid readmission calculations**: The target variable may be incorrectly computed if the "discharge" isn't a true end of the encounter
- **Data quality concerns**: These may represent administrative coding errors or incomplete records

Given that these cases represent only **112 out of 101,766 encounters (0.11%)**, the data loss from excluding them is negligible, while the potential for introducing noise and measurement errors is significant. The conservative approach is to remove these ambiguous cases and work with clean, well-defined discharge events where the 30-day readmission window is unambiguous.