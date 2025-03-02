import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
import os

class DataPipeline:
    """
    A comprehensive data preprocessing, transformation, and loading pipeline.
    """
    
    def __init__(self, input_path=None, output_path=None):
        """
        Initialize the data pipeline with optional input and output paths.
        
        Parameters:
        -----------
        input_path : str, optional
            Path to the input data file
        output_path : str, optional
            Path to save the processed data
        """
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        self.preprocessor = None
        self.transformed_data = None
        
    def extract(self, input_path=None, **kwargs):
        """
        Extract data from various sources (CSV, Excel, database, etc.)
        
        Parameters:
        -----------
        input_path : str, optional
            Path to the input data file
        **kwargs : 
            Additional parameters to pass to the pandas read methods
        
        Returns:
        --------
        self : DataPipeline
            Returns self for method chaining
        """
        if input_path:
            self.input_path = input_path
            
        if self.input_path is None:
            raise ValueError("No input path specified")
        
        # Determine file type based on extension
        file_ext = os.path.splitext(self.input_path)[1].lower()
        
        if file_ext == '.csv':
            self.data = pd.read_csv(self.input_path, **kwargs)
        elif file_ext in ['.xls', '.xlsx']:
            self.data = pd.read_excel(self.input_path, **kwargs)
        elif file_ext == '.json':
            self.data = pd.read_json(self.input_path, **kwargs)
        elif file_ext == '.sql':
            # This is a simple example - in practice, you would use a database connection
            from sqlalchemy import create_engine
            engine = create_engine(f'sqlite:///{self.input_path}')
            self.data = pd.read_sql(kwargs.get('query', 'SELECT * FROM data'), engine)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
            
        print(f"Extracted data shape: {self.data.shape}")
        return self
    
    def explore(self):
        """
        Perform basic exploratory data analysis.
        
        Returns:
        --------
        dict : 
            Dictionary with exploration results
        """
        if self.data is None:
            raise ValueError("No data available. Run extract() first.")
            
        exploration = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes,
            'missing_values': self.data.isnull().sum(),
            'duplicates': self.data.duplicated().sum(),
            'numeric_summary': self.data.describe(),
            'categorical_counts': {col: self.data[col].value_counts() 
                                 for col in self.data.select_dtypes(include=['object']).columns}
        }
        
        return exploration
    
    def setup_preprocessor(self, numeric_features=None, categorical_features=None, 
                          drop_features=None, impute_strategy='median',
                          default_transformers=True):
        """
        Set up the preprocessing pipeline with column transformers.
        
        Parameters:
        -----------
        numeric_features : list, optional
            List of numeric column names
        categorical_features : list, optional
            List of categorical column names
        drop_features : list, optional
            List of features to drop
        impute_strategy : str, default='median'
            Strategy for imputing missing values in numeric columns
        default_transformers : bool, default=True
            Whether to use default transformers or custom ones
            
        Returns:
        --------
        self : DataPipeline
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data available. Run extract() first.")
            
        # Identify numeric and categorical features if not specified
        if numeric_features is None:
            numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
        if categorical_features is None:
            categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
            
        # Remove any features to be dropped
        if drop_features:
            numeric_features = [f for f in numeric_features if f not in drop_features]
            categorical_features = [f for f in categorical_features if f not in drop_features]
            
        # Default transformer for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('scaler', StandardScaler())
        ])
        
        # Default transformer for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # This drops any columns not specified
        )
        
        print(f"Preprocessor set up with {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
        return self
    
    def transform(self, data=None):
        """
        Apply transformations to the data.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Data to transform (uses self.data if not provided)
            
        Returns:
        --------
        self : DataPipeline
            Returns self for method chaining
        """
        if data is not None:
            self.data = data
            
        if self.data is None:
            raise ValueError("No data available. Run extract() first.")
            
        if self.preprocessor is None:
            raise ValueError("No preprocessor set up. Run setup_preprocessor() first.")
            
        # Fit and transform the data
        self.transformed_data = pd.DataFrame(
            self.preprocessor.fit_transform(self.data),
        )
        
        # Get feature names from OneHotEncoder
        ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_columns = ohe.get_feature_names_out(
            self.preprocessor.transformers_[1][2]
        )
        
        # Get numeric feature names
        num_columns = self.preprocessor.transformers_[0][2]
        
        # Combine all feature names
        all_columns = list(num_columns) + list(cat_columns)
        
        # Assign column names to transformed data
        self.transformed_data.columns = all_columns
        
        print(f"Transformed data shape: {self.transformed_data.shape}")
        return self
    
    def add_custom_features(self, custom_features_func):
        """
        Add custom features using a provided function.
        
        Parameters:
        -----------
        custom_features_func : callable
            Function that takes data as input and returns a dataframe with new features
            
        Returns:
        --------
        self : DataPipeline
            Returns self for method chaining
        """
        if self.transformed_data is None:
            raise ValueError("No transformed data available. Run transform() first.")
            
        # Apply custom feature function
        new_features = custom_features_func(self.transformed_data)
        
        # Combine with existing features
        self.transformed_data = pd.concat([self.transformed_data, new_features], axis=1)
        
        print(f"Added {new_features.shape[1]} custom features. New shape: {self.transformed_data.shape}")
        return self
    
    def load(self, output_path=None, format='csv'):
        """
        Load the transformed data to the specified destination.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the processed data
        format : str, default='csv'
            Format to save the data (csv, excel, pickle)
            
        Returns:
        --------
        self : DataPipeline
            Returns self for method chaining
        """
        if output_path:
            self.output_path = output_path
            
        if self.output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_path = f"processed_data_{timestamp}"
            
        if self.transformed_data is None:
            raise ValueError("No transformed data available. Run transform() first.")
            
        # Save based on format
        if format.lower() == 'csv':
            self.transformed_data.to_csv(f"{self.output_path}.csv", index=False)
            print(f"Data saved to {self.output_path}.csv")
        elif format.lower() == 'excel':
            self.transformed_data.to_excel(f"{self.output_path}.xlsx", index=False)
            print(f"Data saved to {self.output_path}.xlsx")
        elif format.lower() == 'pickle':
            with open(f"{self.output_path}.pkl", 'wb') as f:
                pickle.dump(self.transformed_data, f)
            print(f"Data saved to {self.output_path}.pkl")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Save the preprocessor for future use
        with open(f"{self.output_path}_preprocessor.pkl", 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"Preprocessor saved to {self.output_path}_preprocessor.pkl")
            
        return self
    
    def get_data(self):
        """Return the current data."""
        return self.data
    
    def get_transformed_data(self):
        """Return the transformed data."""
        return self.transformed_data
    
    def get_preprocessor(self):
        """Return the preprocessor."""
        return self.preprocessor

# Example usage
if __name__ == "__main__":
    # Example: Process a CSV file
    pipeline = DataPipeline()
    
    # Extract data
    pipeline.extract('data.csv')
    
    # Explore data
    exploration_results = pipeline.explore()
    print(f"Data has {exploration_results['shape'][0]} rows and {exploration_results['shape'][1]} columns")
    
    # Set up and apply preprocessing
    pipeline.setup_preprocessor()
    pipeline.transform()
    
    # Example of adding custom features
    def create_interaction_features(data):
        # Example: Create interactions between first two numeric columns
        if data.shape[1] >= 2:
            cols = data.columns[:2]
            interactions = pd.DataFrame({
                f'{cols[0]}_{cols[1]}_interaction': data[cols[0]] * data[cols[1]]
            })
            return interactions
        return pd.DataFrame()
    
    pipeline.add_custom_features(create_interaction_features)
    
    # Save the processed data
    pipeline.load(output_path='processed_data', format='csv')
