import json
import queue as q
import redis
import rq
import yaml
from flask import Blueprint, jsonify, request, current_app

from fc_app.boosting import calculate_global_model, calculate_local_model, read_input, calculate_average, write_results
from redis_util import redis_set, redis_get, get_step, set_step

pool = redis.BlockingConnectionPool(host='localhost', port=6379, db=0, queue_class=q.Queue)
r = redis.Redis(connection_pool=pool)

# setting 'available' to False --> no data will be send around.
# Change it to True later to send data from the coordinator to the clients or vice versa.
redis_set('available', False)

# Initializes the app with the first step
set_step('start')
# Initialize the local and global data
redis_set('local_data', None)
redis_set('global_data', [])

# Set the paths of the input and output dir
INPUT_DIR = "/mnt/input"
OUTPUT_DIR = "/mnt/output"

api_bp = Blueprint('api', __name__)
tasks = rq.Queue('fc_tasks', connection=r)


@api_bp.route('/status', methods=['GET'])
def status():
    """
    GET request to /status, if True is returned a GET data request will be send
    :return: JSON with key 'available' and value True or False and 'finished' value True or False
    """
    available = redis_get('available')
    current_app.logger.info('[API] /status GET request ' + str(available) + ' - [STEP]: ' + str(get_step()))

    if get_step() == 'start':
        current_app.logger.info('[STEP] start')
        current_app.logger.info('[API] Federated Boosting App')

    elif get_step() == 'local_calculation':
        current_app.logger.info('[STEP] local_calculation')
        model = calculate_local_model()

        if redis_get('is_coordinator'):
            # if this is the coordinator, directly add the local model to the global_data list
            global_data = redis_get('global_data')
            global_data.append(model)
            redis_set('global_data', global_data)
            current_app.logger.info('[STEP] : waiting_for_clients')
        else:
            # if this is a client, set the local model to local_data and set available to true
            redis_set('local_data', model)
            current_app.logger.info('[STEP] waiting_for_coordinator')
            redis_set('available', True)

        set_step('waiting')

    elif get_step() == 'waiting':
        current_app.logger.info('[STEP] waiting')
        if redis_get('is_coordinator'):
            # check if all clients have sent their models already
            has_client_model_arrived()
        else:
            # the clients wait for the coordinator to finish
            current_app.logger.info('[API] Client waiting for coordinator to finish')

    elif get_step() == 'global_calculation':
        # as soon as all data has arrived the global calculation starts
        current_app.logger.info('[STEP] global_calculation')
        calculate_global_model()
        set_step("broadcast_results")

    elif get_step() == 'broadcast_results':
        # the result is broadcasted to the clients
        current_app.logger.info('[STEP] broadcast_results')
        current_app.logger.info('[API] Share global results with clients')
        redis_set('available', True)
        set_step('test_results')

    elif get_step() == 'test_results':
        current_app.logger.info('[STEP] test_results')
        score = calculate_average()
        redis_set("score_of_global_model_on_local_test_set", score)
        current_app.logger.info('[API] Finalize client')
        set_step('write_results')
    elif get_step() == 'write_results':
        current_app.logger.info('[STEP] Write Results')
        write_results(output_dir=OUTPUT_DIR, model=redis_get("global_model"),
                      score=redis_get("score_of_global_model_on_local_test_set"), plot=None)
        if redis_get('is_coordinator'):
            # The coordinator is already finished now
            redis_set('finished', [True])
        # Coordinator and clients continue with the finalize step
        set_step("finalize")

    elif get_step() == 'finalize':
        current_app.logger.info('[STEP] finalize')
        current_app.logger.info("[API] Finalize")
        if redis_get('is_coordinator'):
            # The coordinator waits until all clients have finished
            if have_clients_finished():
                current_app.logger.info('[API] Finalize coordinator.')
                set_step('finished')
            else:
                current_app.logger.info('[API] Not all clients have finished yet.')
        else:
            # The clients set available true to signal the coordinator that they have written the results.
            redis_set('available', True)

    elif get_step() == 'finished':
        # All clients and the coordinator set available to False and finished to True and the computation is done
        current_app.logger.info('[STEP] finished')
        return jsonify({'available': False, 'finished': True})

    return jsonify({'available': True if available else False, 'finished': False})


@api_bp.route('/data', methods=['GET', 'POST'])
def data():
    """
    GET request to /data sends data to coordinator
    POST request to /data pulls data from coordinator
    :return: GET request: JSON with key 'data' and value data
             POST request: JSON True
    """
    if request.method == 'POST':
        current_app.logger.info('[API] /data POST request')
        if redis_get('is_coordinator'):
            # Get data from clients (as coordinator)
            if get_step() != 'finalize':
                # Get local models of the clients
                global_data = redis_get('global_data')
                global_data.append(request.get_json(True)['data'])
                redis_set('global_data', global_data)
                return jsonify(True)
            else:
                # Get Finished flags of the clients
                request.get_json(True)
                finish = redis_get('finished')
                finish.append(request.get_json(True)['finished'])
                redis_set('finished', finish)
                return jsonify(True)
        else:
            # Get models from coordinator (as client)
            redis_set('global_model', request.get_json(True)['global_model'])
            set_step('test_results')
            return jsonify(True)

    elif request.method == 'GET':
        current_app.logger.info('[API] /data GET request')
        if not redis_get('is_coordinator'):
            # send model to coordinator (as client)
            if get_step() != 'finalize':
                # Send local model to the coordinator
                current_app.logger.info('[API] send model to coordinator')
                redis_set('available', False)
                local_data = redis_get('local_data')
                return jsonify({'data': local_data})
            else:
                # Send finish flag to the coordinator
                current_app.logger.info('[API] send finish flag to coordinator')
                redis_set('available', False)
                set_step('finished')
                return jsonify({'finished': True})
        else:
            # broadcast data to clients (as coordinator)
            redis_set('available', False)
            global_model = redis_get('global_model')
            return jsonify({'global_model': global_model})

    else:
        current_app.logger.info('[API] Wrong request type, only GET and POST allowed')
        return jsonify(True)


@api_bp.route('/setup', methods=['POST'])
def setup():
    """
    set setup params, id is the id of the client, coordinator is True if the client is the coordinator,
    in global_data the data from all clients (including the coordinator) will be aggregated,
    clients is a list of all ids from all clients, nr_clients is the number of clients involved in the app
    :return: JSON True
    """
    set_step('setup')
    current_app.logger.info('[STEP] setup')
    retrieve_setup_parameters()
    read_config()
    input_data = read_input(INPUT_DIR)
    if input_data is None:
        current_app.logger.info('[API] no data was found.')
        return jsonify(False)
    else:
        current_app.logger.info('[API] compute local model ')
        redis_set('files', input_data)
        set_step("local_calculation")
        return jsonify(True)


def retrieve_setup_parameters():
    """
    Retrieve the setup parameters and store them in the redis store
    :return: None
    """
    current_app.logger.info('[API] Retrieve Setup Parameters')
    setup_params = request.get_json()
    current_app.logger.info(setup_params)
    redis_set('id', setup_params['id'])
    is_coordinator = setup_params['master']
    redis_set('is_coordinator', is_coordinator)
    if is_coordinator:
        redis_set('global_data', [])
        redis_set('finished', [])
        redis_set('clients', setup_params['clients'])
        redis_set('nr_clients', len(setup_params['clients']))


def has_client_model_arrived():
    """
    Checks if the models of all clients has arrived.
    :return: None
    """
    current_app.logger.info('[API] Coordinator checks if the models of all clients have arrived')
    global_data = redis_get('global_data')
    nr_clients = redis_get('nr_clients')
    current_app.logger.info(
        '[API] ' + str(len(global_data)) + "/" + str(nr_clients) + " clients have sent their models.")
    if len(global_data) == nr_clients:
        current_app.logger.info('[API] The models of all clients have arrived')
        set_step('global_calculation')
    else:
        current_app.logger.info('[API] The model of at least one client is still missing')


def have_clients_finished():
    """
    Checks if the all clients have finished.
    :return: True if all clients have finished, False otherwise
    """
    current_app.logger.info('[API] Coordinator checks if all clients have finished')
    finish = redis_get('finished')
    nr_clients = redis_get('nr_clients')
    current_app.logger.info('[API] ' + str(len(finish)) + "/" + str(nr_clients) + " clients have finished already.")
    if len(finish) == nr_clients:
        current_app.logger.info('[API] All clients have finished.')
        return True
    else:
        current_app.logger.info('[API] At least one client did not finish yet-')
        return False


def read_config():
    with open(INPUT_DIR + '/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['fc_boosting']

        redis_set('input_filename', config['files']['input'])
        redis_set('label_col', config['parameters']['label_col'])
        redis_set('output_filename', config['files']['output'])
        redis_set('missing_data', config['files']['missing_data'])

        redis_set('test_size', config['parameters']['test_size'])
        redis_set('n_estimators', config['parameters']['n_estimators'])
        redis_set('learning_rate', config['parameters']['learning_rate'])
        redis_set('random_state', config['parameters']['random_state'])
        redis_set('metric', config['parameters']['metric'])
        current_app.logger.info('[API] label_col: ' + str(redis_get('label_col')))
        current_app.logger.info('[API] test_size: ' + str(redis_get('test_size')))
        current_app.logger.info('[API] n_estimators: ' + str(redis_get('n_estimators')))
        current_app.logger.info('[API] learning_rate: ' + str(redis_get('learning_rate')))
        current_app.logger.info('[API] random_state: ' + str(redis_get('random_state')))
        current_app.logger.info('[API] metric: ' + str(redis_get('metric')))
