import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModels:
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        
    def load_data(self):
        """Load the engineered data"""
        print("Loading engineered data...")
        X_train = pd.read_csv('X_train_engineered.csv')
        X_test = pd.read_csv('X_test_engineered.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        y_test = pd.read_csv('y_test.csv').values.ravel()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training fraud rate: {y_train.mean():.4f}")
        print(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("\nTraining Logistic Regression...")
        
        # Handle class imbalance with class weights
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=0.1  # Regularization
        )
        
        lr_model.fit(X_train, y_train)
        self.models['Logistic_Regression'] = lr_model
        
        print("Logistic Regression training completed!")
        return lr_model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['Random_Forest'] = rf_model
        
        print("Random Forest training completed!")
        return rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        print("XGBoost training completed!")
        return xgb_model
    
    def train_neural_network(self, X_train, y_train):
        """Train Neural Network model"""
        print("\nTraining Neural Network...")
        
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn_model.fit(X_train, y_train)
        self.models['Neural_Network'] = nn_model
        
        print("Neural Network training completed!")
        return nn_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store performance
        self.model_performance[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        
        return self.model_performance[model_name]
    
    def cross_validate_model(self, model, X_train, y_train, model_name):
        """Perform cross-validation"""
        print(f"\nCross-validating {model_name}...")
        
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        print(f"{model_name} Cross-Validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def train_all_models(self):
        """Train all four models"""
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Select only the encoded and numerical features (exclude original categorical columns)
        feature_columns = [col for col in X_train.columns 
                          if col.endswith('_encoded') or 
                          col in ['income', 'name_email_similarity', 'prev_address_months_count',
                                 'current_address_months_count', 'customer_age', 'days_since_request',
                                 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h',
                                 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
                                 'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
                                 'bank_months_count', 'has_other_cards', 'proposed_credit_limit', 'foreign_request',
                                 'session_length_in_minutes', 'keep_alive_session', 'device_distinct_emails_8w',
                                 'device_fraud_count', 'month', 'velocity_risk_score', 'address_stability',
                                 'credit_limit_risk', 'contact_consistency', 'banking_depth', 'device_risk_score',
                                 'age_income_consistency', 'session_risk', 'foreign_risk', 'multiple_cards_risk',
                                 'income_credit_ratio', 'age_velocity_interaction', 'email_device_risk',
                                 'banking_credit_risk', 'address_foreign_risk']]
        
        X_train_processed = X_train[feature_columns]
        X_test_processed = X_test[feature_columns]
        
        print(f"Training with {len(feature_columns)} features: {feature_columns[:10]}...")
        
        # Train models
        models_to_train = [
            ("Logistic Regression", self.train_logistic_regression),
            ("Random Forest", self.train_random_forest),
            ("XGBoost", self.train_xgboost),
            ("Neural Network", self.train_neural_network)
        ]
        
        for model_name, train_function in models_to_train:
            try:
                model = train_function(X_train_processed, y_train)
                self.evaluate_model(model, X_test_processed, y_test, model_name)
                self.cross_validate_model(model, X_train_processed, y_train, model_name)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        return self.models, self.model_performance
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name.upper()} - Top 10 Feature Importances:")
            print(importance_df.head(10))
            
            return importance_df
        else:
            print(f"\n{model_name.upper()} - Feature importance not available")
            return None
    
    def save_models(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved: {filename}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        performance_df = pd.DataFrame(self.model_performance).T
        
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*60)
        print(f"\nDataset Information:")
        print(f"Training samples: {len(self.X_train):,}")
        print(f"Test samples: {len(self.X_test):,}")
        print(f"Features: {self.X_train_processed.shape[1]}")
        print(f"Fraud rate in training: {self.y_train.mean():.3f}")
        print(f"Fraud rate in test: {self.y_test.mean():.3f}")
        
        print(f"\nModel Performance Summary:")
        # Only include numeric columns for display
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        display_df = performance_df[numeric_cols].round(4)
        print(display_df)
        
        # Save performance report
        performance_df.to_csv('model_performance_report.csv')
        print(f"\nPerformance report saved to: model_performance_report.csv")
        
        return performance_df

# Main execution
if __name__ == "__main__":
    print("Starting Fraud Detection Model Training...")
    
    # Initialize and train models
    fraud_models = FraudDetectionModels()
    models, performance = fraud_models.train_all_models()
    
    # Generate performance report
    performance_df = fraud_models.generate_performance_report()
    
    # Save models
    fraud_models.save_models()
    
    # Get feature importance for tree-based models
    X_train = pd.read_csv('X_train_engineered.csv')
    feature_names = X_train.columns.tolist()
    
    fraud_models.get_feature_importance('Random_Forest', feature_names)
    fraud_models.get_feature_importance('XGBoost', feature_names)
    
    print("\nModel training and evaluation completed!")
    print("Models saved as .pkl files")
    print("Performance metrics saved in model_performance dictionary")