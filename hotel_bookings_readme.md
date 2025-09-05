# Hotel Bookings Data Cleaning & Preprocessing Challenge

This project focuses on cleaning and preprocessing a hotel bookings dataset to prepare it for machine learning analysis. The dataset contains various booking-related features that require careful handling of missing values, outliers, and categorical variables.

## Dataset Overview

The hotel bookings dataset contains information about hotel reservations including:
- Guest demographics (adults, children, babies)
- Booking details (lead time, stay duration, room type)
- Financial information (ADR - Average Daily Rate)
- Categorical features (meal type, market segment, distribution channel)
- Booking status and dates

## Key Data Quality Issues Identified

### Missing Values
- **Company**: >90% missing values - dropped entirely
- **Agent**: Some missing values - filled with 0
- **Country**: Some missing values - filled with mode
- **Children**: Some missing values - filled with median

### Outliers Detected
- **Lead Time**: Values >600 days (capped at 600)
- **Weekend Nights**: Values >10 nights (capped at 10)
- **Week Nights**: Values >20 nights (capped at 20)
- **Adults**: Values >10 guests (capped at 10)
- **Children**: Values >4 children (capped at 3)
- **ADR**: Values >$1000 (capped at $1000)
- **Babies**: Values >2 babies (capped at 2)

### Data Quality Issues
- **Duplicates**: ~25% of the dataset consisted of duplicate records
- **Categorical Variables**: Multiple categories requiring encoding

## Data Preprocessing Steps

### 1. Data Loading and Exploration
```python
df = pd.read_csv('hotel_bookings.csv')
df.info()
df.describe()
```

### 2. Missing Value Analysis
- Created heatmap visualization to identify missing value patterns
- Calculated missing value percentages for each column

### 3. Outlier Detection and Treatment
- Used boxplots to identify outliers in numerical columns
- Applied capping strategy to limit extreme values to reasonable thresholds

### 4. Duplicate Removal
```python
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
```

### 5. Missing Value Imputation
- **Agent**: Filled with 0 (assuming no agent)
- **Country**: Filled with mode (most common country)
- **Children**: Filled with median value

### 6. Feature Engineering
Created new meaningful features:
- **total_guests**: Sum of adults, children, and babies
- **total_nights**: Sum of weekend and weeknight stays
- **is_family**: Binary indicator for bookings with children/babies

### 7. Categorical Variable Encoding
- Applied one-hot encoding to categorical variables:
  - meal, market_segment, distribution_channel
  - deposit_type, customer_type
- Reduced country categories to top 10 + "other"

### 8. Data Cleaning
- Dropped irrelevant columns (reservation_status, reservation_status_date)
- Converted date columns to proper datetime format

### 9. Train-Test Split
```python
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
```

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **scikit-learn**: Train-test splitting

## File Structure

```
├── gtc_ml_project_1_data_cleaning_&_preprocessing_challenge.py
├── hotel_bookings.csv
└── README.md
```

## Key Findings

1. **High Duplicate Rate**: 25% of records were duplicates, significantly reducing dataset size after cleaning
2. **Extensive Outliers**: Most numerical columns contained outliers requiring capping strategies
3. **Missing Data Patterns**: Company field was largely unusable due to >90% missing values
4. **Feature Engineering Opportunities**: Created meaningful derived features from existing data

## Next Steps

This cleaned dataset is now ready for:
- Exploratory Data Analysis (EDA)
- Feature selection and importance analysis
- Machine learning model development
- Predictive modeling for booking cancellations or customer behavior

## Usage

1. Ensure the `hotel_bookings.csv` file is in your working directory
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the preprocessing script:
   ```python
   python gtc_ml_project_1_data_cleaning_&_preprocessing_challenge.py
   ```

## Data Quality Metrics

- **Original Dataset Size**: [Original row count]
- **After Duplicate Removal**: ~75% of original size
- **Missing Values**: Reduced from multiple columns to 0% missing
- **Outliers**: Capped to reasonable business limits
- **Feature Count**: Expanded through one-hot encoding and feature engineering

---

**Note**: This preprocessing pipeline ensures the data is clean, consistent, and ready for machine learning applications while preserving the integrity and business meaning of the original dataset.