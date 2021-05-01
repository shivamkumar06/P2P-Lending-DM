import pandas as pd
from sklearn.preprocessing import LabelEncoder

Loan_file=pd.read_csv("loan.csv")

print('The shape is {}'.format(Loan_file.shape))
print('Memory : {} Mb'.format(int(Loan_file.memory_usage(deep=False).sum() / 1000000)))

check_null = Loan_file.isnull().sum(axis=0).sort_values(ascending=False)/float(len(Loan_file))
check_null[check_null>0.6]

Loan_file.drop(check_null[check_null>0.5].index, axis=1, inplace=True)
Loan_file.dropna(axis=0, thresh=30, inplace=True)

delete_me = ['policy_code','pymnt_plan', 'url', 'member_id', 'application_type', 'acc_now_delinq','emp_title', 'zip_code','title']
Loan_file.drop(delete_me , axis=1, inplace=True)

# strip months from 'term' and make it an int
Loan_file['term'] = Loan_file['term'].str.split(' ').str[1]

# extract numbers from emp_length and fill missing values with the median
Loan_file['emp_length'] = Loan_file['emp_length'].str.extract('(\d+)').astype(float)
Loan_file['emp_length'] = Loan_file['emp_length'].fillna(Loan_file.emp_length.median())

col_dates = Loan_file.dtypes[Loan_file.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    Loan_file[d] = Loan_file[d].dt.to_period('M')

Loan_file['amt_difference'] = 'same'
Loan_file.loc[(Loan_file['funded_amnt'] - Loan_file['funded_amnt_inv']) > 0,'amt_difference'] = 'low'

# Make categorical

Loan_file['delinq_2yrs_cat'] = 'no'
Loan_file.loc[Loan_file['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'

Loan_file['inq_last_6mths_cat'] = 'no'
Loan_file.loc[Loan_file['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'

Loan_file['pub_rec_cat'] = 'no'
Loan_file.loc[Loan_file['pub_rec']> 0,'pub_rec_cat'] = 'yes'


# Create new metric
Loan_file['acc_ratio'] = Loan_file.open_acc / Loan_file.total_acc

features = ['loan_amnt', 'amt_difference', 'term',
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat',
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',
            'loan_status'
           ]

X_clean = Loan_file.loc[Loan_file.loan_status != 'Current', features]
#print(X_clean.head())

mask = (X_clean.loan_status == 'Charged Off')
X_clean['target'] = 0
X_clean.loc[mask,'target'] = 1

cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status','loan_status']

# Drop any residual missing value (only 24)
X_clean.dropna(axis=0, how = 'any', inplace = True)

#Categorical datas are numerically encoded in the same column
enc=LabelEncoder()
for i in cat_features:
    X_clean[i]=enc.fit_transform(X_clean[i])

X_clean.to_csv('clean_final.csv')



