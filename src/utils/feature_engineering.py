import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_risk_indicators(self, df):
        """Create new risk indicator features"""
        df = df.copy()
        
        # 1. Velocity-based risk indicators
        df['velocity_risk_score'] = (
            df['velocity_6h'] * 0.4 + 
            df['velocity_24h'] * 0.3 + 
            df['velocity_4w'] * 0.3
        )
        
        # 2. Address stability indicator
        df['address_stability'] = np.where(
            df['prev_address_months_count'] == -1, 0,
            df['current_address_months_count'] / (df['prev_address_months_count'] + 1)
        )
        
        # 3. Credit limit utilization risk
        df['credit_limit_risk'] = df['proposed_credit_limit'] / (df['income'] + 1)
        
        # 4. Email-phone consistency
        df['contact_consistency'] = df['phone_home_valid'] * df['phone_mobile_valid']
        
        # 5. Banking relationship depth
        df['banking_depth'] = df['bank_months_count'] * df['bank_branch_count_8w']
        
        # 6. Device risk score
        df['device_risk_score'] = (
            df['device_distinct_emails_8w'] * 0.6 + 
            df['device_fraud_count'] * 0.4
        )
        
        # 7. Age-income consistency
        df['age_income_consistency'] = df['customer_age'] / (df['income'] + 1)
        
        # 8. Session behavior risk
        df['session_risk'] = np.where(df['session_length_in_minutes'] > 30, 1, 0)
        
        # 9. Foreign request risk
        df['foreign_risk'] = df['foreign_request'] * df['zip_count_4w']
        
        # 10. Multiple cards risk
        df['multiple_cards_risk'] = df['has_other_cards'] * df['credit_risk_score']
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col + '_encoded'] = le.transform(df[col])
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Income vs Credit Limit interaction
        df['income_credit_ratio'] = df['income'] / (df['proposed_credit_limit'] + 1)
        
        # Age vs Velocity interaction
        df['age_velocity_interaction'] = df['customer_age'] * df['velocity_risk_score']
        
        # Email similarity vs device risk
        df['email_device_risk'] = df['name_email_similarity'] * df['device_risk_score']
        
        # Banking depth vs credit risk
        df['banking_credit_risk'] = df['banking_depth'] / (df['credit_risk_score'] + 1)
        
        # Address stability vs foreign request
        df['address_foreign_risk'] = df['address_stability'] * df['foreign_request']
        
        return df
    
    def scale_numerical_features(self, X_train, X_test=None):
        """Scale numerical features"""
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle infinite values
        X_train = X_train.copy()
        if X_test is not None:
            X_test = X_test.copy()
            
        # Replace inf and -inf with NaN, then fill with median
        for col in numerical_cols:
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
            if X_test is not None:
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with median
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            if X_test is not None:
                X_test[col] = X_test[col].fillna(median_val)
        
        if X_test is not None:
            X_train_scaled = self.scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled = self.scaler.transform(X_test[numerical_cols])
            
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_cols, index=X_test.index)
            
            # Combine scaled numerical with categorical
            X_train_final = X_train.drop(numerical_cols, axis=1)
            X_test_final = X_test.drop(numerical_cols, axis=1)
            
            X_train_final = pd.concat([X_train_final, X_train_scaled], axis=1)
            X_test_final = pd.concat([X_test_final, X_test_scaled], axis=1)
            
            return X_train_final, X_test_final
        else:
            X_scaled = self.scaler.transform(X_train[numerical_cols])
            X_scaled = pd.DataFrame(X_scaled, columns=numerical_cols, index=X_train.index)
            
            X_final = X_train.drop(numerical_cols, axis=1)
            X_final = pd.concat([X_final, X_scaled], axis=1)
            
            return X_final
    
    def fit_transform(self, df):
        """Complete feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Step 1: Create risk indicators
        print("Creating risk indicators...")
        df = self.create_risk_indicators(df)
        
        # Step 2: Encode categorical features
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)
        
        # Step 3: Create interaction features
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Step 4: Separate features and target
        y = df['fraud_bool']
        X = df.drop(['fraud_bool'], axis=1)
        
        # Step 5: Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Step 6: Scale numerical features
        print("Scaling numerical features...")
        X_train_scaled, X_test_scaled = self.scale_numerical_features(X_train, X_test)
        
        # Store feature names
        self.feature_names = X_train_scaled.columns.tolist()
        
        print(f"Feature engineering complete!")
        print(f"Original features: {df.shape[1] - 1}")
        print(f"Engineered features: {len(self.feature_names)}")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def transform(self, df):
        """Transform new data using fitted transformers"""
        # Step 1: Create risk indicators
        df = self.create_risk_indicators(df)
        
        # Step 2: Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Step 3: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 4: Separate features and target (if present)
        if 'fraud_bool' in df.columns:
            y = df['fraud_bool']
            X = df.drop(['fraud_bool'], axis=1)
        else:
            X = df
            y = None
        
        # Step 5: Scale numerical features
        X_scaled = self.scale_numerical_features(X)
        
        return X_scaled, y
    
    def save_transformers(self, filepath='feature_engineer.pkl'):
        """Save the fitted transformers"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }, f)
        print(f"Transformers saved to {filepath}")
    
    def load_transformers(self, filepath='feature_engineer.pkl'):
        """Load fitted transformers"""
        with open(filepath, 'rb') as f:
            transformers = pickle.load(f)
            self.scaler = transformers['scaler']
            self.label_encoders = transformers['label_encoders']
            self.feature_names = transformers['feature_names']
        print(f"Transformers loaded from {filepath}")

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('Base.csv')
    
    # Initialize feature engineer
    feature_engineer = FraudFeatureEngineer()
    
    # Fit and transform
    X_train, X_test, y_train, y_test = feature_engineer.fit_transform(df)
    
    # Save the processed data
    print("Saving processed data...")
    X_train.to_csv('X_train_engineered.csv', index=False)
    X_test.to_csv('X_test_engineered.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    # Save the feature engineer
    feature_engineer.save_transformers()
    
    print("Feature engineering complete! Files saved:")
    print("- X_train_engineered.csv")
    print("- X_test_engineered.csv") 
    print("- y_train.csv")
    print("- y_test.csv")
    print("- feature_engineer.pkl")