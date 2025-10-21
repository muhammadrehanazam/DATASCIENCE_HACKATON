import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class FraudEnsembleModel:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.results = {}
        
    def train_models(self, X_train, y_train):
        """Train individual models for ensemble"""
        print("Training individual models for ensemble...")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.neural_network import MLPClassifier
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            class_weight='balanced', 
            random_state=42, 
            max_iter=1000
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # XGBoost
        print("Training XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Neural Network
        print("Training Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=42,
            max_iter=500
        )
        nn_model.fit(X_train, y_train)
        self.models['neural_network'] = nn_model
        
        print(f"Successfully trained {len(self.models)} models")
    
    def load_data(self):
        """Load the engineered data"""
        print("Loading engineered data...")
        
        X_train = pd.read_csv('X_train_engineered.csv')
        X_test = pd.read_csv('X_test_engineered.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        y_test = pd.read_csv('y_test.csv').values.ravel()
        
        # Select only the encoded and numerical features (same as individual models)
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
        
        print(f"Training data shape: {X_train_processed.shape}")
        print(f"Test data shape: {X_test_processed.shape}")
        print(f"Training fraud rate: {y_train.mean():.4f}")
        print(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def create_voting_ensemble(self, X_train, y_train, voting='soft'):
        """Create voting ensemble from loaded models"""
        print(f"\nCreating {voting} voting ensemble...")
        
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        self.ensemble_model.fit(X_train, y_train)
        print("✓ Ensemble model trained successfully")
        
        return self.ensemble_model
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble model performance"""
        print("\nEvaluating ensemble model...")
        
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nENSEMBLE MODEL RESULTS:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")
        
        return self.results
    
    def compare_with_individual_models(self, X_test, y_test):
        """Compare ensemble performance with individual models"""
        print("\n" + "="*60)
        print("COMPARISON: ENSEMBLE vs INDIVIDUAL MODELS")
        print("="*60)
        
        comparison_results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            comparison_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
        
        # Add ensemble results
        comparison_results['ensemble'] = self.results
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        print("\nPerformance Comparison:")
        print(comparison_df.round(4))
        
        # Find best model
        best_model = comparison_df['auc'].idxmax()
        print(f"\nBest Model (by AUC): {best_model}")
        print(f"Best AUC Score: {comparison_df.loc[best_model, 'auc']:.4f}")
        
        # Save comparison
        comparison_df.to_csv('ensemble_vs_individual_comparison.csv')
        print(f"\nComparison saved to: ensemble_vs_individual_comparison.csv")
        
        return comparison_df
    
    def cross_validate_ensemble(self, X_train, y_train, cv=5):
        """Perform cross-validation on ensemble model"""
        print(f"\nCross-validating ensemble model ({cv}-fold)...")
        
        cv_scores = cross_val_score(
            self.ensemble_model, X_train, y_train, 
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        print(f"Cross-Validation AUC Scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_ensemble_model(self, filename='fraud_ensemble_model.pkl'):
        """Save the ensemble model"""
        with open(filename, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        print(f"\nEnsemble model saved to: {filename}")
    
    def run_complete_ensemble_pipeline(self):
        """Run the complete ensemble pipeline"""
        print("Starting Complete Ensemble Model Pipeline...")
        print("="*60)
        
        try:
            # Load data and train models
            X_train, X_test, y_train, y_test = self.load_data()
            self.train_models(X_train, y_train)
            
            # Create ensemble
            self.create_voting_ensemble(X_train, y_train, voting='soft')
            
            # Evaluate ensemble
            self.evaluate_ensemble(X_test, y_test)
            
            # Cross-validate
            self.cross_validate_ensemble(X_train, y_train)
            
            # Compare with individual models
            self.compare_with_individual_models(X_test, y_test)
            
            # Save ensemble
            self.save_ensemble_model()
            
            print("\n" + "="*60)
            print("ENSEMBLE MODEL PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return self.results
            
        except Exception as e:
            print(f"Error in ensemble pipeline: {str(e)}")
            return None

if __name__ == "__main__":
    ensemble = FraudEnsembleModel()
    results = ensemble.run_complete_ensemble_pipeline()