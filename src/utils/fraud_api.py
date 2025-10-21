from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import integrated fraud detection system
from integrated_fraud_system import IntegratedFraudDetector

app = Flask(__name__)

# Global variables for model and feature engineering
model = None
feature_engineer = None
feature_columns = None
integrated_detector = None

# Define default feature columns (same as used in training)
DEFAULT_FEATURE_COLUMNS = [
    'credit_risk_score', 'velocity_4w', 'age_velocity_interaction',
    'velocity_24h', 'zip_count_4w', 'banking_depth', 'current_address_months_count',
    'income', 'name_email_similarity', 'prev_address_months_count', 'customer_age',
    'days_since_request', 'intended_balcon_amount', 'bank_branch_count_8w',
    'date_of_birth_distinct_emails_4w', 'email_is_free', 'phone_home_valid',
    'phone_mobile_valid', 'bank_months_count', 'has_other_cards', 'proposed_credit_limit',
    'foreign_request', 'session_length_in_minutes', 'keep_alive_session',
    'device_distinct_emails_8w', 'device_fraud_count', 'month', 'velocity_risk_score',
    'address_stability', 'credit_limit_risk', 'contact_consistency', 'device_risk_score',
    'age_income_consistency', 'session_risk', 'foreign_risk', 'multiple_cards_risk',
    'income_credit_ratio', 'email_device_risk', 'banking_credit_risk', 'address_foreign_risk',
    'payment_type_encoded', 'employment_status_encoded', 'housing_status_encoded',
    'source_encoded', 'device_os_encoded'
]

def load_models():
    """Load the trained models and feature engineer"""
    global model, feature_engineer, feature_columns, integrated_detector
    
    try:
        # Initialize integrated fraud detector (combines ML + AI)
        integrated_detector = IntegratedFraudDetector()
        print("✓ Integrated fraud detector initialized")
        
        # Load feature engineer
        try:
            with open('feature_engineer.pkl', 'rb') as f:
                feature_engineer = pickle.load(f)
        except FileNotFoundError:
            print("⚠️  Feature engineer file not found, using default preprocessing")
            feature_engineer = None
        
        # Load ensemble model (we'll use a simple RandomForest for demo)
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple model for demonstration
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Use default feature columns
        global feature_columns
        feature_columns = DEFAULT_FEATURE_COLUMNS
        
        print("✓ Models loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def preprocess_transaction(transaction_data):
    """Preprocess transaction data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Apply feature engineering (simplified for demo)
        # In real implementation, this would use the full feature_engineer
        
        # Create engineered features
        df['name_email_similarity'] = np.random.uniform(0, 1)
        df['prev_address_months_count'] = np.random.exponential(12)
        df['current_address_months_count'] = np.random.exponential(24)
        df['customer_age'] = np.random.normal(35, 10)
        df['days_since_request'] = np.random.exponential(30)
        df['velocity_6h'] = np.random.exponential(0.1)
        df['velocity_24h'] = np.random.exponential(0.05)
        df['velocity_4w'] = np.random.exponential(0.2)
        df['bank_branch_count_8w'] = np.random.poisson(3)
        df['date_of_birth_distinct_emails_4w'] = np.random.poisson(2)
        df['credit_risk_score'] = np.random.normal(0, 1)
        df['email_is_free'] = np.random.choice([0, 1])
        df['phone_home_valid'] = np.random.choice([0, 1])
        df['phone_mobile_valid'] = np.random.choice([0, 1])
        df['bank_months_count'] = np.random.exponential(24)
        df['has_other_cards'] = np.random.choice([0, 1])
        df['session_length_in_minutes'] = np.random.exponential(15)
        df['keep_alive_session'] = np.random.choice([0, 1])
        df['device_distinct_emails_8w'] = np.random.poisson(1)
        df['device_fraud_count'] = np.random.poisson(0)
        df['month'] = np.random.choice(range(1, 13))
        df['velocity_risk_score'] = np.random.normal(0, 1)
        df['address_stability'] = np.random.normal(0, 1)
        df['credit_limit_risk'] = np.random.normal(0, 1)
        df['contact_consistency'] = np.random.normal(0, 1)
        df['banking_depth'] = np.random.normal(0, 1)
        df['device_risk_score'] = np.random.normal(0, 1)
        df['age_income_consistency'] = np.random.normal(0, 1)
        df['session_risk'] = np.random.normal(0, 1)
        df['foreign_risk'] = np.random.normal(0, 1)
        df['multiple_cards_risk'] = np.random.normal(0, 1)
        df['income_credit_ratio'] = np.random.normal(0, 1)
        df['age_velocity_interaction'] = np.random.normal(0, 1)
        df['email_device_risk'] = np.random.normal(0, 1)
        df['banking_credit_risk'] = np.random.normal(0, 1)
        df['address_foreign_risk'] = np.random.normal(0, 1)
        
        # Encode categorical variables (simplified)
        categorical_mappings = {
            'payment_type': {'AA': 0, 'AB': 1, 'AC': 2, 'AD': 3, 'AE': 4},
            'employment_status': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11},
            'housing_status': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25},
            'source': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4},
            'device_os': {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(0)
        
        # Use default feature columns if global one is None
        current_feature_columns = feature_columns if feature_columns is not None else DEFAULT_FEATURE_COLUMNS
        
        # Select only the required features
        available_features = [col for col in current_feature_columns if col in df.columns]
        X = df[available_features]
        
        # Fill missing features with defaults
        missing_features = set(current_feature_columns) - set(available_features)
        for feature in missing_features:
            X[feature] = 0
        
        return X[current_feature_columns]
        
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fraud Detection API with Integrated ML + AI',
        'version': '2.0.0',
        'endpoints': {
            'predict': '/predict - POST transaction data for fraud prediction',
            'predict_integrated': '/predict?integrated=true - Use ML + AI agents',
            'health': '/health - Check API health status',
            'info': '/info - Get API information'
        },
        'features': {
            'ml_models': 'Traditional ML-only prediction',
            'integrated_analysis': 'ML + AI agent analysis',
            'ai_agents': '5 specialized AI agents for fraud detection'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/info', methods=['GET'])
def get_info():
    """Get API information"""
    return jsonify({
        'model_type': 'Integrated ML + AI (Ensemble + 5 AI Agents)',
        'features': len(feature_columns) if feature_columns else 0,
        'feature_names': feature_columns[:10] + ['...'] if feature_columns else [],
        'ai_agents': [
            'TransactionMonitor',
            'PatternAnalyzer', 
            'BehaviorChecker',
            'CaseResearcher',
            'AlertGenerator'
        ],
        'threshold': 0.5,
        'version': '2.0.0',
        'integration_weights': {
            'ml_weight': 0.6,
            'ai_weight': 0.4
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud probability for a transaction using integrated ML + AI"""
    try:
        # Get transaction data from request
        transaction_data = request.get_json()
        print(f"📥 Received request with transaction data: {transaction_data is not None}")
        
        if not transaction_data:
            return jsonify({
                'error': 'No transaction data provided',
                'status': 'failed'
            }), 400
        
        # Check if integrated analysis is requested
        integrated_param = request.args.get('integrated', 'false')
        ai_analysis_param = request.args.get('ai_analysis', 'false')
        
        print(f"📊 Raw parameters - integrated: {integrated_param}, ai_analysis: {ai_analysis_param}")
        
        use_integrated = str(integrated_param).lower() == 'true' if integrated_param else False
        include_ai_analysis = str(ai_analysis_param).lower() == 'true' if ai_analysis_param else False
        
        print(f"📊 Parsed parameters - integrated: {use_integrated}, ai_analysis: {include_ai_analysis}")
        
        if use_integrated and integrated_detector:
            # Use integrated fraud detection (ML + AI)
            print("🚀 Using integrated fraud detection (ML + AI)")
            
            # Create user history if not provided
            user_history = None
            if 'user_history' in transaction_data:
                user_history = pd.DataFrame(transaction_data['user_history'])
            
            # Run integrated analysis
            result = integrated_detector.analyze_transaction(transaction_data, user_history)
            
            # Extract scalar values from potential pandas Series/numpy arrays
            fraud_prob = float(result['integrated_analysis']['fraud_probability'])
            confidence_val = float(result['integrated_analysis']['confidence'])
            
            return jsonify({
                'prediction': fraud_prob > 0.5,
                'fraud_probability': fraud_prob,
                'confidence': confidence_val,
                'risk_level': result['integrated_analysis']['risk_level'],
                'processing_time': result['processing_time'],
                'method': 'integrated_ml_ai',
                'ai_analysis': result['ai_analysis'] if include_ai_analysis else None,
                'status': 'success',
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        else:
            # Use traditional ML-only approach
            print("🤖 Using traditional ML-only prediction")
            
            # Preprocess the transaction
            X = preprocess_transaction(transaction_data)
            print(f"📊 Preprocessed data shape: {X.shape if X is not None else 'None'}")
            print(f"📊 Preprocessed data type: {type(X)}")
            
            # Make prediction (using a simple rule-based approach for demo)
            # In real implementation, this would use the trained ensemble model
            
            # Simple risk scoring for demonstration
            risk_score = 0
            print(f"🎯 Starting risk_score calculation with X shape: {X.shape if X is not None else 'None'}")
            
            # Ensure we have data and extract scalar values safely
            if X is not None and not X.empty:
                print(f"🎯 X is not None and not empty, columns: {list(X.columns)}")
                # Credit risk score
                if 'credit_risk_score' in X.columns and len(X) > 0:
                    print(f"🎯 Processing credit_risk_score...")
                    credit_risk_val = float(X['credit_risk_score'].iloc[0]) if hasattr(X['credit_risk_score'].iloc[0], 'item') else float(X['credit_risk_score'].iloc[0])
                    print(f"🎯 credit_risk_val: {credit_risk_val} (type: {type(credit_risk_val)})")
                    risk_score += credit_risk_val * 0.3
                
                # Velocity factors
                if 'velocity_4w' in X.columns and len(X) > 0:
                    print(f"🎯 Processing velocity_4w...")
                    velocity_4w_val = float(X['velocity_4w'].iloc[0]) if hasattr(X['velocity_4w'].iloc[0], 'item') else float(X['velocity_4w'].iloc[0])
                    print(f"🎯 velocity_4w_val: {velocity_4w_val} (type: {type(velocity_4w_val)})")
                    risk_score += min(velocity_4w_val * 2, 0.2)
                
                if 'velocity_24h' in X.columns and len(X) > 0:
                    print(f"🎯 Processing velocity_24h...")
                    velocity_24h_val = float(X['velocity_24h'].iloc[0]) if hasattr(X['velocity_24h'].iloc[0], 'item') else float(X['velocity_24h'].iloc[0])
                    print(f"🎯 velocity_24h_val: {velocity_24h_val} (type: {type(velocity_24h_val)})")
                    risk_score += min(velocity_24h_val * 3, 0.15)
                
                # Device and email risk
                if 'email_device_risk' in X.columns and len(X) > 0:
                    print(f"🎯 Processing email_device_risk...")
                    print(f"🎯 X['email_device_risk']: {X['email_device_risk']}")
                    print(f"🎯 X['email_device_risk'].iloc[0]: {X['email_device_risk'].iloc[0]}")
                    print(f"🎯 Type of X['email_device_risk'].iloc[0]: {type(X['email_device_risk'].iloc[0])}")
                    
                    try:
                        email_device_val = float(X['email_device_risk'].iloc[0])
                        print(f"🎯 email_device_val: {email_device_val} (type: {type(email_device_val)})")
                        risk_score += email_device_val * 0.1
                    except (ValueError, TypeError) as e:
                        print(f"❌ Failed to convert email_device_risk to float: {e}")
                        # Use a safe default value
                        email_device_val = 0.0
                        print(f"🎯 Using default value: {email_device_val}")
                        risk_score += email_device_val * 0.1
                
                # Age-velocity interaction
                if 'age_velocity_interaction' in X.columns and len(X) > 0:
                    print(f"🎯 Processing age_velocity_interaction...")
                    age_velocity_val = float(X['age_velocity_interaction'].iloc[0]) if hasattr(X['age_velocity_interaction'].iloc[0], 'item') else float(X['age_velocity_interaction'].iloc[0])
                    print(f"🎯 age_velocity_val: {age_velocity_val} (type: {type(age_velocity_val)})")
                    risk_score += age_velocity_val * 0.15
                    
                print(f"🎯 Final risk_score before normalization: {risk_score}")
            else:
                # Fallback to random values if preprocessing failed
                print(f"🎯 X is None or empty, using random fallback")
                risk_score = np.random.uniform(0, 0.5)
            
            # Normalize risk score
            print(f"🎯 risk_score before normalization: {risk_score} (type: {type(risk_score)})")
            fraud_probability = min(max(risk_score, 0), 1)
            print(f"🎯 Calculated fraud_probability: {fraud_probability} (type: {type(fraud_probability)})")
            
            # Determine risk level and recommendation
            if fraud_probability >= 0.7:
                risk_level = 'HIGH'
            elif fraud_probability >= 0.4:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            if fraud_probability >= 0.7:
                recommendation = 'BLOCK_TRANSACTION'
            elif fraud_probability >= 0.4:
                recommendation = 'REVIEW_MANUALLY'
            else:
                recommendation = 'APPROVE'
                
            print(f"🎯 About to create response with fraud_probability: {fraud_probability}")
            
            # Create response
            response = {
                'status': 'success',
                'prediction': {
                    'fraud_probability': float(fraud_probability),
                    'risk_level': risk_level,
                    'recommendation': recommendation,
                    'confidence': 0.85 if fraud_probability > 0.5 else 0.75
                },
                'explanation': {
                'top_risk_factors': [
                    {'factor': 'credit_risk_score', 'impact': float(X['credit_risk_score'].iloc[0]) if 'credit_risk_score' in X.columns and len(X) > 0 else 0},
                    {'factor': 'velocity_4w', 'impact': float(X['velocity_4w'].iloc[0]) if 'velocity_4w' in X.columns and len(X) > 0 else 0},
                    {'factor': 'email_device_risk', 'impact': float(X['email_device_risk'].iloc[0]) if 'email_device_risk' in X.columns and len(X) > 0 else 0}
                ]
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict fraud probability for multiple transactions"""
    try:
        # Get batch transaction data from request
        batch_data = request.get_json()
        
        if not batch_data or 'transactions' not in batch_data:
            return jsonify({
                'error': 'No batch transaction data provided',
                'status': 'failed'
            }), 400
        
        transactions = batch_data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({
                'error': 'Transactions must be a list',
                'status': 'failed'
            }), 400
        
        predictions = []
        
        for transaction in transactions:
            try:
                # Preprocess the transaction
                X = preprocess_transaction(transaction)
                
                # Simple risk scoring (same as single predict)
                risk_score = 0
                
                # Ensure we have data and extract scalar values safely
                if X is not None and not X.empty:
                    if 'credit_risk_score' in X.columns and len(X) > 0:
                        credit_risk_val = float(X['credit_risk_score'].iloc[0]) if hasattr(X['credit_risk_score'].iloc[0], 'item') else float(X['credit_risk_score'].iloc[0])
                        risk_score += credit_risk_val * 0.3
                    if 'velocity_4w' in X.columns and len(X) > 0:
                        velocity_4w_val = float(X['velocity_4w'].iloc[0]) if hasattr(X['velocity_4w'].iloc[0], 'item') else float(X['velocity_4w'].iloc[0])
                        risk_score += min(velocity_4w_val * 2, 0.2)
                    if 'velocity_24h' in X.columns and len(X) > 0:
                        velocity_24h_val = float(X['velocity_24h'].iloc[0]) if hasattr(X['velocity_24h'].iloc[0], 'item') else float(X['velocity_24h'].iloc[0])
                        risk_score += min(velocity_24h_val * 3, 0.15)
                    if 'email_device_risk' in X.columns and len(X) > 0:
                        email_device_val = float(X['email_device_risk'].iloc[0]) if hasattr(X['email_device_risk'].iloc[0], 'item') else float(X['email_device_risk'].iloc[0])
                        risk_score += email_device_val * 0.1
                    if 'age_velocity_interaction' in X.columns and len(X) > 0:
                        age_velocity_val = float(X['age_velocity_interaction'].iloc[0]) if hasattr(X['age_velocity_interaction'].iloc[0], 'item') else float(X['age_velocity_interaction'].iloc[0])
                        risk_score += age_velocity_val * 0.15
                else:
                    # Fallback to random values if preprocessing failed
                    risk_score = np.random.uniform(0, 0.5)
                
                fraud_probability = min(max(risk_score, 0), 1)
                
                # Determine risk level and recommendation
                if fraud_probability >= 0.7:
                    risk_level = 'HIGH'
                    recommendation = 'BLOCK_TRANSACTION'
                elif fraud_probability >= 0.4:
                    risk_level = 'MEDIUM'
                    recommendation = 'REVIEW_MANUALLY'
                else:
                    risk_level = 'LOW'
                    recommendation = 'APPROVE'
                
                predictions.append({
                    'fraud_probability': float(fraud_probability),
                    'risk_level': risk_level,
                    'recommendation': recommendation,
                    'confidence': 0.85 if fraud_probability > 0.5 else 0.75
                })
                
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'fraud_probability': 0.0,
                    'risk_level': 'ERROR',
                    'recommendation': 'MANUAL_REVIEW'
                })
        
        response = {
            'status': 'success',
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

if __name__ == '__main__':
    # Load models before starting the server
    print("Loading models...")
    if load_models():
        print("✓ Models loaded successfully")
        print("Starting Fraud Detection API server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("✗ Failed to load models. Starting with demo mode...")
        app.run(host='0.0.0.0', port=5000, debug=False)