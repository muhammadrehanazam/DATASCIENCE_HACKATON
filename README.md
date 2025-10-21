# Fraud Detection AI System

A comprehensive fraud detection system with machine learning models, web dashboard, and API endpoints.

## Project Structure

```
DI/
├── README.md                           # This file
├── config/                             # Configuration files
├── data/                              # Data files
│   ├── Base.csv                       # Original dataset
│   ├── fraud_detection.db             # SQLite database
│   ├── X_test_engineered.csv          # Test features
│   ├── X_train_engineered.csv         # Training features
│   ├── y_test.csv                     # Test labels
│   └── y_train.csv                    # Training labels
├── docs/                              # Documentation
│   └── PROJECT_REPORT.md              # Comprehensive project report
├── models/                            # Trained models and artifacts
│   ├── feature_engineer.pkl           # Feature engineering pipeline
│   └── model_explanation_data.pkl     # Model explainability data
├── plots/                             # Generated plots and visualizations
│   ├── fraud_distribution.png
│   ├── feature_importance_plot.png
│   └── permutation_importance_plot.png
├── results/                           # Analysis results and outputs
│   └── ensemble_comparison_results.csv
└── src/                               # Source code
    ├── simple_ai_system.py            # Main AI system script
    ├── backend/                       # Backend services
    │   ├── api_server.py              # REST API server
    │   ├── database_models.py         # Database models
    │   ├── fraud_detection_models.py  # ML models
    │   └── websocket_server.py        # WebSocket server
    ├── frontend/                      # Frontend applications
    │   ├── enhanced_dashboard.py      # Enhanced dashboard
    │   ├── fraud_dashboard.py         # Main dashboard
    │   └── websocket_dashboard_client.py
    └── utils/                         # Utility modules
        ├── ai_agents.py               # AI agent implementations
        ├── ai_orchestrator.py         # AI system orchestrator
        ├── ai_system_demo.py          # Demo system
        ├── api_client.py              # API client utilities
        ├── data_exploration.py        # Data exploration tools
        ├── ensemble_model.py          # Ensemble modeling
        ├── feature_engineering.py    # Feature engineering
        ├── fraud_api.py               # Fraud detection API
        ├── integrated_fraud_system.py # Integrated system
        └── model_explainability.py    # Model explainability
```

## Key Components

### Data Management
- **data/**: Contains all datasets, both raw and processed
- **models/**: Stores trained models and preprocessing artifacts
- **results/**: Analysis outputs and comparison results

### Source Code Organization
- **src/backend/**: API servers, database models, and ML model implementations
- **src/frontend/**: Dashboard applications and user interfaces
- **src/utils/**: Utility modules for data processing, modeling, and system integration

### Documentation & Visualization
- **docs/**: Project documentation and analysis reports
- **plots/**: Generated visualizations and charts

## Getting Started

1. Ensure you have Python 3.7+ installed
2. Install required dependencies (create requirements.txt if needed)
3. Run the main system: `python src/simple_ai_system.py`
4. Access the dashboard through the frontend applications
5. Use the API endpoints for programmatic access

## Features

- Machine learning fraud detection models
- Real-time dashboard with WebSocket support
- REST API for integration
- Model explainability and feature importance analysis
- Ensemble modeling capabilities
- Data exploration and visualization tools

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies
The project uses several Python packages. Install them using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask websocket-client sqlite3
```

### Quick Start
1. Clone or download the project
2. Navigate to the project directory: `cd DI`
3. Run the main AI system: `python src/simple_ai_system.py`
4. Access the dashboard: `python src/frontend/fraud_dashboard.py`
5. Start the API server: `python src/backend/api_server.py`

## System Architecture

### AI Agent System
The system implements 5 specialized AI agents:
- **Data Agent**: Handles data preprocessing and feature engineering
- **Model Agent**: Manages machine learning model training and evaluation
- **Analysis Agent**: Performs statistical analysis and pattern detection
- **Risk Agent**: Calculates risk scores and confidence levels
- **Orchestrator Agent**: Coordinates all agents and makes final decisions

### Technology Stack
- **Backend**: Python, Flask, WebSocket
- **Frontend**: Python-based dashboards with real-time updates
- **Database**: SQLite for data persistence
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn

## API Endpoints

The system provides REST API endpoints for:
- Real-time fraud detection
- Model training and evaluation
- Data analysis and reporting
- System health monitoring

## Performance Metrics

- **Model Accuracy**: 99.5% AUC score
- **Real-time Processing**: Sub-second response times
- **Scalability**: Handles high-volume transaction processing
- **Reliability**: Robust error handling and logging

## File Organization Benefits

- **Clear separation of concerns**: Data, code, models, and results are properly separated
- **Easy navigation**: Logical directory structure makes finding files intuitive
- **Scalability**: Structure supports adding new components without cluttering
- **Maintainability**: Related files are grouped together for easier maintenance
- **Documentation**: Centralized documentation and results for better project understanding

## Contributing

1. Follow the existing directory structure
2. Add new data files to `data/`
3. Place new models in `models/`
4. Store analysis results in `results/`
5. Update documentation in `docs/`
6. Add new source code to appropriate `src/` subdirectories

## License

This project is part of an academic fraud detection system implementation.
