# Define prediction features and target variable
REGRESSION_FEATURES = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size', 'remote_ratio']    # List of features for prediction

# Define mappings for user interface values to dataset codes
EXPERIENCE_LEVEL_MAPPING = [
    "Entry Level",    # For newcomers to the field
    "Mid Level",      # For those with some experience
    "Senior Level",   # For experienced professionals
    "Executive"       # For leadership positions
]

EMPLOYMENT_TYPE_MAPPING = [
    "Full-Time",      # Standard full-time employment
    "Part-Time",      # Part-time positions
    "Contract",       # Fixed-term contract work
    "Freelance"       # Independent contractor work
]

COMPANY_SIZE_MAPPING = [
    "Small",            # Small enterprises
    "Medium",           # Mid-sized organisations
    "Large"             # Large corporations
]



# Mapping entries for Classification prediction
ymapping=1 # we can set the year as 2022, which has 1 as value, because it is the closest to today
exlemapping=[['Entry Level',0],['Mid Level',0.19],['Senior Level',0.56],['Executive',1.0]]
emtymapping=[['Contract',1.0],['Freelance',0.10],['Full-Time',0.52],['Part-Time',0]]
jtmapping=[['3D Computer Vision Researcher',0],['AI Scientist',0.152],['Analytics Engineer',0.424],\
            ['Applied Data Scientist',0.426],['Applied Machine Learning Scientist',0.341],['BI Data Analyst',0.173],\
            ['Big Data Architect',0.235],['Big Data Engineer',0.116],['Business Data Analyst',0.178],\
            ['Cloud Data Engineer',0.298],['Computer Vision Engineer',0.097],['Computer Vision Software Engineer',0.250],\
            ['Data Analyst',0.211],['Data Analytics Engineer',0.148],['Data Analytics Lead',1.000],\
            ['Data Analytics Manager',0.304],['Data Architect',0.431],['Data Engineer',0.261],['Data Engineering Manager',0.294],\
            ['Data Science Consultant',0.160],['Data Science Engineer',0.176],['Data Science Manager',0.382],\
            ['Data Scientist',0.245],['Data Specialist',0.399],['Director of Data Engineering',0.378],\
            ['Director of Data Science',0.474],['ETL Developer',0.123],['Finance Data Analyst',0.141],\
            ['Financial Data Analyst',0.674],['Head of Data',0.387],['Head of Data Science',0.353],['Head of Machine Learning',0.184],\
            ['Lead Data Analyst',0.217],['Lead Data Engineer',0.336],['Lead Data Scientist',0.274],['Lead Machine Learning Engineer',0.206],\
            ['ML Engineer',0.280],['Machine Learning Developer',0.201],['Machine Learning Engineer',0.240],\
            ['Machine Learning Infrastructure Engineer',0.239],['Machine Learning Manager',0.279],['Machine Learning Scientist',0.383],\
            ['Marketing Data Analyst',0.208],['NLP Engineer',0.079],['Principal Data Analyst',0.293],['Principal Data Engineer',0.808],\
            ['Principal Data Scientist',0.525],['Product Data Analyst',0.019],['Research Scientist',0.259],['Staff Data Scientist',0.249]]
ermapping=[['AE',0.4896],['AR',0.2857],['AT',0.3711],['AU',0.5308],['BE',0.4168],['BG',0.3877],['BO',0.3622],['BR',0.2583],['CA',0.4754],\
            ['CH',0.6038],['CL',0.1838],['CN',0.2006],['CO',0.0910],['CZ',0.3367],['DE',0.4149],['DK',0.1696],['DZ',0.4896],\
            ['EE',0.1478],['ES',0.2734],['FR',0.2851],['GB',0.3952],['GR',0.2675],['HK',0.3164],['HN',0.0816],['HR',0.2123],\
            ['HU',0.1632],['IE',0.3441],['IN',0.1700],['IQ',0.4897],['IR',0],['IT',0.2938],['JE',0.4898],['JP',0.5078],['KE',0.0268],\
            ['LU',0.2811],['MD',0.0714],['MT',0.1243],['MX',0.0723],['MY',1.0],['NG',0.1326],['NL',0.2905],['NZ',0.6173],['PH',0.2130],\
            ['PK',0.1197],['PL',0.2662],['PR',0.7959],['PT',0.1982],['RO',0.2419],['RS',0.1098],['RU',0.5191],['SG',0.5111],\
            ['SI',0.3052],['TN',0.1422],['TR',0.0821],['UA',0.0479],['US',0.7453],['VN',0.1367]]
rrmapping=[['0',0.62],['50',0.00],['100',1.00]]
clmapping=[['AE',0.6254],['AS',0.0915],['AT',0.4489],['AU',0.6778],['BE',0.5322],['BR',0.0951],['CA',0.6262],['CH',0.3916],['CL',0.2347],\
            ['CN',0.4408],['CO',0.1162],['CZ',0.3057],['DE',0.5052],['DK',0.3282],['DZ',0.6255],['EE',0.1887],['ES',0.3196],['FR',0.3906],\
            ['GB',0.5058],['GR',0.3128],['HN',0.1042],['HR',0.2711],['HU',0.2067],['IE',0.4393],['IL',0.7495],['IN',0.1601],['IQ',0.6256],\
            ['IR',0.00],['IT',0.2108],['JP',0.7174],['KE',0.0343],['LU',0.2602],['MD',0.0912],['MT',0.1587],['MX',0.1832],['MY',0.2345],\
            ['NG',0.1693],['NL',0.3318],['NZ',0.7882],['PK',0.0608],['PL',0.4044],['PT',0.2853],['RO',0.3648],['RU',1.00],['SG',0.5556],\
            ['SI',0.3897],['TR',0.1048],['UA',0.0612],['US',0.9139],['VN',0.01]]
csmapping=[['Large',1.00],['Medium',0.91],['Small',0.00]]



# Function to retrieve value based on label
def get_value_by_label(label, couples_list):
    for couple in couples_list:
        if couple[0] == label:
            return couple[1]
    return None  # Return None if label is not found
