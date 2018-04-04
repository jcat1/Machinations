from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import grid_search
from sklearn.grid_search import ParameterGrid

class ModelWrapper:
    '''
    Wrapper to simplify loading for sklearn models
    '''
    def __init__(self, X, Y, model, parameters):
        self.X = X
        self.Y = Y
        self.model = model
        self.parameters = parameters
    
    def get_model(self):
        '''
        return the model
        '''
        assert (self.model is not None)
        return self.model
    
    def train(self, X, Y):
        '''
        train on a subset of the data
        '''
        self.model = self.get_model()
        assert (self.model is not None)
        self.model = grid_search.GridSearchCV(self.model, self.parameters)
        self.model.fit(X,Y)
        
    def predict(self, X_test):
        assert (self.model is not None)
        return self.model.predict(X_test)
        
    def build(self):
        '''
        build the model, report diagnostics
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y)
        print("Training model...")
        self.train(X_train, Y_train)
        print("Done training model.")
        print(self.model.best_estimator_)
        print("-----")
        
        training_error = accuracy_score(self.predict(X_train), Y_train)
        validation_error = accuracy_score(self.predict(X_test), Y_test)
        print("Train error:", training_error)
        print("Validation error:", validation_error)  
