import yaml

from fc_app.boosting import calculate_global_model, calculate_local_model, read_input, calculate_average, write_results
from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition('read input', Role.BOTH)

    def run(self) -> str or None:
        return 'read input'
    
    
@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """
    
    def register(self):
        self.register_transition('local calculation', Role.BOTH)
        
    def run(self) -> str or None:
    
        # Initialize the local and global data
        self.store('local_data', None)
        self.store('global_data', [])

        # Set the paths of the input and output dir
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
    
        self.log('Read config-file...')
        self.read_config()
        input_data = read_input(self.load('INPUT_DIR'), self)
        self.store('files', input_data)
        
        return 'local calculation'

    def read_config(self):
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_boosting']

            self.store('input_filename', config['files']['input'])
            self.store('label_col', config['parameters']['label_col'])
            self.store('output_filename', config['files']['output'])
            self.store('missing_data', config['files']['missing_data'])

            self.store('test_size', config['parameters']['test_size'])
            self.store('n_estimators', config['parameters']['n_estimators'])
            self.store('learning_rate', config['parameters']['learning_rate'])
            self.store('random_state', config['parameters']['random_state'])
            self.store('metric', config['parameters']['metric'])


@app_state('local calculation', Role.BOTH)
class LocalCalculationState(AppState):
    """
    Perform local computation and send the computation data to the coordinator.
    """
    
    def register(self):
        self.register_transition('global calculation', Role.COORDINATOR)
        self.register_transition('wait', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        model = calculate_local_model(self)
        self.store('local_data', model)
        self.send_data_to_coordinator(model)

        if self.is_coordinator:
            self.store('global_data', [])
            return 'global calculation'
        else:
            return 'wait'


@app_state('global calculation', Role.COORDINATOR)
class GlobalCalculationState(AppState):
    """
    The coordinator receives the local model from each client and aggregates them.
    The coordinator broadcasts the global model to the clients.
    """
    
    def register(self):
        self.register_transition('test results', Role.COORDINATOR)
        
    def run(self) -> str or None:
        data = self.gather_data()
        self.store('global_data', data)
        calculate_global_model(self)
        data_to_broadcast = self.load('global_model')
        self.broadcast_data(data_to_broadcast, send_to_self=False)
        return 'test results'
    

@app_state('wait', Role.PARTICIPANT)
class WaitState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """
    
    def register(self):
        self.register_transition('test results', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        global_model = self.await_data()
        self.store('global_model', global_model)
        return 'test results'
        
        
@app_state('test results', Role.BOTH)
class TestResultsState(AppState):
    """
    Calculates the score of global model on local test set.
    """
    
    def register(self):
        self.register_transition('write results', Role.BOTH)
        
    def run(self) -> str or None:
        score = calculate_average(self)
        self.store("score_of_global_model_on_local_test_set", score)
        
        return 'write results'
 
 
@app_state('write results', Role.BOTH)
class WriteResultsState(AppState):
    """
    Writes the score of global model on local test set.
    """
    
    def register(self):
        self.register_transition('terminal', Role.BOTH)
        
    def run(self) -> str or None:
        write_results(self, output_dir=self.load('OUTPUT_DIR'), model=self.load("global_model"),
                      score=self.load("score_of_global_model_on_local_test_set"), plot=None)
        
        self.send_data_to_coordinator('DONE')
            
        if self.is_coordinator:
            self.gather_data()
        
        return 'terminal'
