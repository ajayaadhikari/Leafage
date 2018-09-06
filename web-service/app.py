import random

import numpy

from leafage import use_cases
from leafage.use_cases.all_use_cases import all_data_sets
from leafage.utils.Classifiers import possible_classifiers

__author__ = "Riccardo Satta, TNO"

import hashlib
import inspect
import os
import dill

from flask import Flask, jsonify, request, abort, make_response

from leafage.scenario import Scenario

scenario_cache_dir = "cache/"

app = Flask(__name__)


def get_default_args(func):
    #signature = inspect.signature(func)
    signature = inspect.getargspec(func)
    return dict(zip(signature.args[-len(signature.defaults):], signature.defaults))


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': error.description}), 400)


@app.route('/leafage/api/v1.0/get_available_datasets', methods=['GET'])
def get_available_datasets():
    all_ds = list(all_data_sets.keys())
    return jsonify({'available_datasets': all_ds}), 201


@app.route('/leafage/api/v1.0/get_available_classifiers', methods=['GET'])
def get_available_classifiers():
    all_c = list(possible_classifiers.keys())
    return jsonify({'available_datasets': all_c}), 201


@app.route('/leafage/api/v1.0/initiate_scenario', methods=['POST'])
def initiate_scenario():
    if not request.json:
        abort(400, 'Request must be json.')

    default_params = get_default_args(Scenario.__init__)

    dataset_source = request.json['dataset_source']
    dataset = request.json['dataset']
    classifier_name = request.json['classifier_name']
    classifier_hyper_parameters = request.json.get('classifier_hyper_parameters',
                                                   default_params['classifier_hyper_parameters'])
    random_state = request.json.get('random_state',
                                    default_params['random_state'])
    neighbourhood_sampling_strategy = request.json.get('neighbourhood_sampling_strategy',
                                                       default_params['neighbourhood_sampling_strategy'])
    encoder_classifier = request.json.get('encoder_classifier',
                                          default_params['encoder_classifier'])

    force_regenerate = request.json.get('encoder_classifier', False)

    scenario_ID = hashlib.md5(
        str(dataset_source) +
        str(dataset) +
        str(classifier_name) +
        str(classifier_hyper_parameters) +
        str(random_state) +
        str(neighbourhood_sampling_strategy) +
        str(encoder_classifier)
    ).hexdigest()

    scenario_fname = scenario_cache_dir + ("/" if scenario_cache_dir[-1] is not "/" else "") + scenario_ID + ".dill"

    if os.path.isfile(scenario_fname) and not force_regenerate:
        message = 'A scenario with that ID exists already. You can force regenerating the scenario ' \
                  'by setting \'force_regenerate\':true'
        scenario = dill.load(open(scenario_fname, "rb"))
    else:
        # create Scenario, store and give user the scenario ID
        scenario = Scenario(dataset_source=dataset_source,
                            dataset=dataset,
                            classifier_name=classifier_name,
                            classifier_hyper_parameters=classifier_hyper_parameters,
                            random_state=random_state,
                            neighbourhood_sampling_strategy=neighbourhood_sampling_strategy,
                            encoder_classifier=encoder_classifier)
        message = 'Good news! A scenario was created correctly. You can use now the scenario_ID to get ' \
                  'explanations using one of the "explain" API'

    dill.dump(scenario, open(scenario_fname, "wb"))
    return jsonify({'message': message,
                    'scenario_ID': scenario_ID,
                    'scenario_info':{'data_size': scenario.data.feature_vector.shape,
                                     'data_class_names': list(scenario.data.class_names),
                                     'data_feature_names': list(scenario.data.feature_names),
                                     'sample_data_point': list(scenario.data.feature_vector[0])}}), 201


@app.route('/leafage/api/v1.0/get_random_data_point_from_dataset', methods=['POST'])
def get_random_data_point_from_dataset():
    scenario_ID = request.json['scenario_ID']
    scenario_fname = scenario_cache_dir + ("/" if scenario_cache_dir[-1] is not "/" else "") + scenario_ID + ".dill"

    if not os.path.isfile(scenario_fname):
        abort(400, 'The provided scenario_ID does not exist.')

    scenario = dill.load(open(scenario_fname, "rb"))
    return jsonify(list(random.choice(scenario.data.feature_vector))), 201


@app.route('/leafage/api/v1.0/get_allowed_values_per_feature', methods=['POST'])
def get_allowed_values_per_feature():
    scenario_ID = request.json['scenario_ID']
    scenario_fname = scenario_cache_dir + ("/" if scenario_cache_dir[-1] is not "/" else "") + scenario_ID + ".dill"

    if not os.path.isfile(scenario_fname):
        abort(400, 'The provided scenario_ID does not exist.')

    scenario = dill.load(open(scenario_fname, "rb"))
    feature_names = scenario.data.feature_names

    allowed_values = dict()
    for i in range(len(feature_names)):
        allowed_values[feature_names[i]] = list(numpy.unique(scenario.data.feature_vector[:, i]))

    return jsonify(allowed_values), 201



@app.route('/leafage/api/v1.0/explain', methods=['POST'])
def explain():
    if not request.json:
        abort(400, 'Request must be json.')

    if 'scenario_ID' not in request.json:
        abort(400, 'Please provide a scenario_ID.')

    if not 'test_instance' in request.json:
        abort(400, 'Please provide a test_instance.')

    if not 'nr_of_examples' in request.json:
        abort(400, 'Please set nr_of_examples.')

    scenario_ID = request.json['scenario_ID']
    scenario_fname = scenario_cache_dir + ("/" if scenario_cache_dir[-1] is not "/" else "") + scenario_ID + ".dill"

    if not os.path.isfile(scenario_fname):
        abort(400, 'The provided scenario_ID does not exist.')

    scenario = dill.load(open(scenario_fname, "rb"))

    nr_of_examples = request.json['nr_of_examples']
    test_instance = request.json['test_instance']

    explanation = scenario.get_explanation(test_instance, nr_of_examples)

    return jsonify({'explanation': explanation.to_json()}), 201


if __name__ == '__main__':
    app.run(debug=True)