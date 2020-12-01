class DataProcessor:
    def __init__(self,norm_strategy: str = "Mean", output_dimension: int = None):
        self.norm_strategy = norm_strategy
        self.output_dimension = output_dimension
    def normalize(self,df_data):
            df = df_data.copy() 
            if self.norm_strategy not in ["MinMax","Mean"]:
                raise Exception("Choose norm_strategy between MinMax or Mean")
            columns = list(df_data.columns)
            X = df[columns[:-1]]
            Y = df[columns[-1]]    
            if self.norm_strategy == "Mean":
                df_normalized = (X - X.mean())/X.std()
                df[columns[:-1]] = df_normalized
                return df
            if self.norm_strategy == "MinMax":
                df_normalized = (X - X.min())/(X.max()-X.min())
                df[columns[:-1]] = df_normalized
                return df
      
    def pca(self,df_data):
        columns = list(df_data.columns)
        X = df_data[columns[:-1]]
        Y = df_data[columns[-1]]
        cov = np.cov(X,rowvar = False) 
        u,s,v = np.linalg.svd(cov)
        
        if self.output_dimension  == None:
            explained_variance = 0
            total = sum(s)
            compteur = 0
            while explained_variance < 0.95:
                explained_variance = explained_variance + s[compteur]/total
                compteur = compteur + 1
            U_k = u[:,:compteur] #(n x k)
            X_reduced = np.dot(X,U_k)
            print("data reduced to %d dimensions" %(compteur))
            return X_reduced, Y
        else:
            if len(columns[:-1]) <  self.output_dimension:
                raise Exception("Output Dimension > Input Dimension")        
            U_k = u[:,:self.output_dimension] #(n x k)
            X_reduced = np.dot(X,U_k)
            print("data reduced to %d dimensions" %(self.output_dimension) )
            return X_reduced, Y