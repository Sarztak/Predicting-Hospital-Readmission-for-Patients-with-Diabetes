import pandas as pd
def categorize_diagnosis(diag_code):
    """
    Categorize ICD-9 diagnosis codes into broad categories
    """
    if pd.isna(diag_code) or diag_code == '?':
        return 'Missing'
    
    # Convert to string and get first 3 characters
    diag_str = str(diag_code)
    
    # Remove decimal points for comparison
    if '.' in diag_str:
        diag_num = diag_str.split('.')[0]
    else:
        diag_num = diag_str
    
    # Handle E and V codes
    if diag_str.startswith('E'):
        return 'Injury'
    if diag_str.startswith('V'):
        return 'Other'
    
    # Convert to float for numeric comparisons
    try:
        diag_float = float(diag_num)
    except:
        return 'Other'
    
    # Circulatory: 390-459, 785
    if (390 <= diag_float <= 459) or diag_float == 785:
        return 'Circulatory'
    
    # Respiratory: 460-519, 786
    elif (460 <= diag_float <= 519) or diag_float == 786:
        return 'Respiratory'
    
    # Digestive: 520-579, 787
    elif (520 <= diag_float <= 579) or diag_float == 787:
        return 'Digestive'
    
    # Diabetes: 250.xx
    elif 250 <= diag_float < 251:
        return 'Diabetes'
    
    # Injury: 800-999
    elif 800 <= diag_float <= 999:
        return 'Injury'
    
    # Musculoskeletal: 710-739
    elif 710 <= diag_float <= 739:
        return 'Musculoskeletal'
    
    # Genitourinary: 580-629, 788
    elif (580 <= diag_float <= 629) or diag_float == 788:
        return 'Genitourinary'
    
    # Neoplasms: 140-239
    elif 140 <= diag_float <= 239:
        return 'Neoplasms'
    
    else:
        return 'Other'
