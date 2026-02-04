discharge_disposition_mapping = {
    # HOME - Discharged to home (with or without home health services)
    1: 'HOME',  # Discharged to home
    6: 'HOME',  # Discharged/transferred to home with home health service
    8: 'HOME',  # Discharged/transferred to home under care of Home IV provider
    
    # DECEASED - Patient died or entered hospice care
    11: 'DECEASED',  # Expired
    13: 'DECEASED',  # Hospice / home
    14: 'DECEASED',  # Hospice / medical facility
    19: 'DECEASED',  # Expired at home. Medicaid only, hospice.
    20: 'DECEASED',  # Expired in a medical facility. Medicaid only, hospice.
    21: 'DECEASED',  # Expired, place unknown. Medicaid only, hospice.
    
    # TRANSFERRED_ACUTE_CARE - Transferred to another acute care hospital
    2: 'TRANSFERRED_ACUTE_CARE',  # Discharged/transferred to another short term hospital
    5: 'TRANSFERRED_ACUTE_CARE',  # Discharged/transferred to another type of inpatient care institution
    27: 'TRANSFERRED_ACUTE_CARE',  # Discharged/transferred to a federal health care facility
    29: 'TRANSFERRED_ACUTE_CARE',  # Discharged/transferred to a Critical Access Hospital (CAH)
    30: 'TRANSFERRED_ACUTE_CARE',  # Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere
    
    # TRANSFERRED_POST_ACUTE - Transferred to skilled nursing, rehab, or long-term care
    3: 'TRANSFERRED_POST_ACUTE',  # Discharged/transferred to SNF (Skilled Nursing Facility)
    4: 'TRANSFERRED_POST_ACUTE',  # Discharged/transferred to ICF (Intermediate Care Facility)
    22: 'TRANSFERRED_POST_ACUTE',  # Discharged/transferred to another rehab fac including rehab units of a hospital
    23: 'TRANSFERRED_POST_ACUTE',  # Discharged/transferred to a long term care hospital
    24: 'TRANSFERRED_POST_ACUTE',  # Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare
    
    # PSYCHIATRIC - Transferred to psychiatric care
    28: 'PSYCHIATRIC',  # Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital
    
    # LEFT_AMA - Patient left against medical advice
    7: 'LEFT_AMA',  # Left AMA (Against Medical Advice)
    
    # STILL_IN_SYSTEM - Patient still receiving care or administrative transfer within facility
    9: 'STILL_IN_SYSTEM',  # Admitted as an inpatient to this hospital
    12: 'STILL_IN_SYSTEM',  # Still patient or expected to return for outpatient services
    15: 'STILL_IN_SYSTEM',  # Discharged/transferred within this institution to Medicare approved swing bed
    16: 'STILL_IN_SYSTEM',  # Discharged/transferred/referred another institution for outpatient services
    17: 'STILL_IN_SYSTEM',  # Discharged/transferred/referred to this institution for outpatient services
    
    # NEWBORN_SPECIAL - Newborn transferred for specialized care
    10: 'NEWBORN_SPECIAL',  # Neonate discharged to another hospital for neonatal aftercare
    
    # UNKNOWN - Missing, null, or unmapped values
    18: 'UNKNOWN',  # NULL
    25: 'UNKNOWN',  # Not Mapped
    26: 'UNKNOWN'   # Unknown/Invalid
}


admission_source_mapping = {
    # REFERRAL - Planned admission via physician, clinic, or HMO referral
    1: 'REFERRAL',  # Physician Referral
    2: 'REFERRAL',  # Clinic Referral
    3: 'REFERRAL',  # HMO Referral
    
    # EMERGENCY - Unplanned emergency admission
    7: 'EMERGENCY',  # Emergency Room
    8: 'EMERGENCY',  # Court/Law Enforcement
    
    # TRANSFER - Transferred from another healthcare facility
    4: 'TRANSFER',  # Transfer from a hospital
    5: 'TRANSFER',  # Transfer from a Skilled Nursing Facility (SNF)
    6: 'TRANSFER',  # Transfer from another health care facility
    10: 'TRANSFER',  # Transfer from critical access hospital
    18: 'TRANSFER',  # Transfer From Another Home Health Agency
    22: 'TRANSFER',  # Transfer from hospital inpt/same fac reslt in a sep claim
    25: 'TRANSFER',  # Transfer from Ambulatory Surgery Center
    26: 'TRANSFER',  # Transfer from Hospice
    
    # NEWBORN - Obstetric/newborn admissions
    11: 'NEWBORN',  # Normal Delivery
    12: 'NEWBORN',  # Premature Delivery
    13: 'NEWBORN',  # Sick Baby
    14: 'NEWBORN',  # Extramural Birth
    23: 'NEWBORN',  # Born inside this hospital
    24: 'NEWBORN',  # Born outside this hospital
    
    # HOME_HEALTH - Readmission from home health agency
    19: 'HOME_HEALTH',  # Readmission to Same Home Health Agency
    
    # UNKNOWN - Missing, null, or unmapped values
    9: 'UNKNOWN',   # Not Available
    15: 'UNKNOWN',  # Not Available
    17: 'UNKNOWN',  # NULL
    20: 'UNKNOWN',  # Not Mapped
    21: 'UNKNOWN'   # Unknown/Invalid
}

admission_type_mapping = {
    # EMERGENCY - Immediate, unplanned admission for urgent medical need
    1: 'EMERGENCY',  # Emergency
    7: 'EMERGENCY',  # Trauma Center
    
    # URGENT - Needs admission soon but not immediately life-threatening
    2: 'URGENT',  # Urgent
    
    # ELECTIVE - Scheduled/planned admission for non-emergency procedure
    3: 'ELECTIVE',  # Elective
    
    # NEWBORN - Newborn/obstetric admission
    4: 'NEWBORN',  # Newborn
    
    # UNKNOWN - Missing, null, or unmapped values
    5: 'UNKNOWN',  # Not Available
    6: 'UNKNOWN',  # NULL
    8: 'UNKNOWN'   # Not Mapped
}

payer_mapping = {
    'MC': 'Government',
    'MD': 'Government',
    'CH': 'Government',
    'OG': 'Government',
    'SI': 'Government',
    'MP': 'Government',
    
    'HM': 'Private',
    'BC': 'Private',
    'CP': 'Private',
    'CM': 'Private',
    'PO': 'Private',
    'DM': 'Private',
    
    'SP': 'Self-Pay',
    'WC': 'Workers-Comp',
    
    '?': 'Unknown',
    'UN': 'Unknown',
    'OT': 'Unknown',
    'FR': 'Unknown'
}