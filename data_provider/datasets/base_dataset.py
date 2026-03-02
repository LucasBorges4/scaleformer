def _preprocess_data(self):
        """
        Preprocess the loaded data.
        Common preprocessing steps for all datasets.
        """
        if self.df_raw is None:
            raise ValueError("Data not loaded. _load_data() must set self.df_raw")
            
        # Ensure date column is datetime
        if 'date' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['date'])
        elif 'timestamp' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['timestamp'])
        else:
            raise ValueError("DataFrame must contain a 'date' or 'timestamp' column")
            
        # Extract data columns based on features mode
        if self.features in ['M', 'MS']:
            # Skip 'date' column
            cols_data = [col for col in self.df_raw.columns if col != 'date']
        else:  # 'S'
            if self.target not in self.df_raw.columns:
                raise ValueError(f"Target column '{self.target}' not found")
            cols_data = [self.target]
            
        self.df_data = self.df_raw[cols_data]
        
        # Fit scaler if needed and transform data
        if self.scale:
            self.scaler = StandardScaler()
            # Use training data for fitting (set_type 0 = train)
            if self.flag == 'train':
                train_data = self.df_data.values
                self.scaler.fit(train_data)
                self.data = self.scaler.transform(self.df_data.values)
            elif hasattr(self, 'train_data'):
                self.scaler.fit(self.train_data)
                self.data = self.scaler.transform(self.df_data.values)
            else:
                # Fallback: fit and transform on available data
                self.scaler.fit(self.df_data.values)
                self.data = self.scaler.transform(self.df_data.values)
        else:
            self.data = self.df_data.values
    

    @abstractmethod
    def _split_data(self) -> Tuple[List[int], List[int]]:
        Split data into train/val/test or appropriate subsets.
        Returns:
            tuple: (border1s, border2s)
        """
        pass
        
    def _finalize_data(self):
        """Finalize data by splitting and creating time features."""
        border1s, border2s = self._split_data()
        
        # For prediction mode, always use first (and only) split
        if self.flag == 'pred':
            idx = 0
        else:
            idx = self.set_type
            
        border1 = border1s[idx]
        border2 = border2s[idx]
        
        # Split data sequences
        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2]
        
        # Create time features for this slice
        dates = self.df_raw['date'].iloc[border1:border2]
        self.data_stamp = self._create_time_features(dates)