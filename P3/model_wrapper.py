from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelWrapper:
    '''
    Wrapper to simplify loading for sklearn models
    '''
    def __init__(self, X, Y, parameters, model):
        self.X = X
        self.Y = Y
        self.parameters = parameters
        self.model = model
    
    def get_model(self):
        '''
        return the model
        '''
        assert (self.model is not None)
        return self.model
    
    def train(self, X, Y, parameters):
        '''
        train on a subset of the data
        '''
        self.model = self.get_model()
        assert (self.model is not None)
        self.model = grid_search.GridSearchCV(self.model, parameters)
        self.model.fit(X,Y)
        
    def predict(self, X_test):
        assert (self.model is not None)
        return self.model.predict(X_test)
        
    def build(self):
        '''
        build the model, report diagnostics
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y)
        parameters = self.parameters
        print("Training model...")
        self.train(X_train, Y_train, parameters)
        print("Done training model.")
        print("Optimal Hyperparameters:")
        for p in self.parameters:
            print(str(p) + ":", self.model.best_estimator_.p)
        print("-----")
        
        training_error = accuracy_score(self.predict(X_train), Y_train)
        validation_error = accuracy_score(self.predict(X_test), Y_test)
        print("Train error:", training_error)
        print("Validation error:", validation_error)  
