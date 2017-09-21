import datetime
import sys
import rosparam
import os
import pickle
import subprocess
import time
import yaml
import ast

###########################################
# Helper functions, not directly used by the experiment_coordinator
#########################
def _run_evaluation(command):
    """
    Helper method; Starts the evaluation program as subprocess, since it only works with python2 (thanks, ROS...)
    Yields strings to describe the subprocess' status as it runs.
    """
    eval_process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(eval_process.stdout.readline, ""):
        if stdout_line.startswith('{'): # Is the output a python dict's string representation?
            status_dict = ast.literal_eval(stdout_line)
            status_string = "\r                                                            " +\
                            "                                   \r" +\
                            "\t\t[Active threads " + str(status_dict['active_jobs']) +\
                                               "/" + str(status_dict['max_parallel_jobs']) + "]" +\
                            " [Finished jobs " + str(status_dict['completed_jobs']) +\
                                           "/" + str(status_dict['total_jobs']) + "]" +\
                            " [Completed in " + str(status_dict['estimated_finish_time']) + "]"
        else: # otherwise just yield the string
            status_string = "\t\t" + stdout_line 
        yield status_string
        sys.stdout.flush()
    eval_process.stdout.close()
    return_code = eval_process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

###########################################
# Actual interface functions, those need to be implemented.
#########################
def create_evaluation_function_sample(toplevel_directory, sample):
    """
    DLR-map-matcher-evaluation-specific code to get an evaluation_function Sample from a finished map matcher run.

    :param toplevel_directory: The directory from which the Sample's can be generated.
    :param sample: Sample object which should be filled with data
    """
    sample.origin = toplevel_directory
    sample.translation_errors = []
    sample.rotation_errors = []
    sample.time = datetime.timedelta(0)
    for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
        for file_name in files:
            if file_name == "statistics.pkl":
                if not "MAP_MATCHER_JOB_COMPLETED" in files:
                    raise IOError("Something went wrong while generating this part of the sample,\
                                   couldn't find ", os.path.join(root, "MAP_MATCHER_JOB_COMPLETED"))
                pickle_path = os.path.join(root, file_name)
                eval_result_data = pickle.load(open(pickle_path, 'rb'), encoding='latin1')
                sample.translation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['translation'].values())
                sample.rotation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['rotation'].values())
                sample.time += datetime.timedelta(seconds=float(eval_result_data['timings']['MapMatcherNode::runBatchMode total']['total']))
    
    # Check if there any statistics.pkl were found!
    if len(sample.translation_errors) == 0:
        raise IOError("No statistics.pkl files were found in the toplevel_directory's filetree.")

    for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
        for file_name in files:
            if file_name == "params.yaml":
                # As soon as one params.yaml is found somewhere, we're done.
                # (because all other dirs have exactly the same parameters)
                sample.parameters = rosparam.load_file(os.path.join(root, file_name))[0][0]
                return

def generate_sample(rosparams, sample_generator_config):
    """
    Generates data for a new Sample and returns the path at which the generated data lies.

    :param rosparams: A dict of the complete set of parameters.
    :param sample_generator_config: A dict with other parameters needed for generating the Sample's data.
                                    Make sure params in this dict don't mess up the validity of the SampleDatabase.
                                    The experiment_coordinator only manages the rosparams.
    """

    # Create the path to the folder in which all information for this map matcher run is stored
    env_dir = os.path.join(sample_generator_config['environment'], time.strftime("%Y-%m-%d_%H:%M:%S"))
    if os.path.exists(env_dir):
        raise IOError(env_dir, "already exists, can't generate sample here!")
    os.makedirs(env_dir) # Create the folder on disk
    results_path = os.path.join(env_dir, "results") # Path for map matcher's output
    yaml_path = os.path.join(env_dir, "evaluation.yaml") # Path for map matcher's config
    rosparams_path = os.path.join(env_dir, "params.yaml") # Path for the rosparams
    # Add the paths to the config, so the map_matcher_evaluation script finds it
    sample_generator_config['parameters'] = rosparams_path
    sample_generator_config['results'] = results_path
    # Create *the* map matcher evaluation script config.
    # sample_generator_config may contain some unnecessary fields, but that doesn't hurt.
    mm_yaml_config = {'MapMatcherJobs': {'gpr_auto_entry': sample_generator_config}}
    with open(yaml_path, 'w+') as yaml_file:
        # Store the yaml file on disk. The evaluator will load it again.
        yaml.dump(mm_yaml_config, yaml_file)
    with open(rosparams_path, 'w+') as rosparams_file:
        yaml.dump(rosparams, rosparams_file)

    evaluation_command = [sample_generator_config['evaluator_executable'], yaml_path]
    for arg_string in sample_generator_config['extra_arguments']:
        evaluation_command.append(arg_string)
    print("\t\tStarting evaluation process in directory", env_dir)
    # Run the evaluation and print its output
    for output in _run_evaluation(evaluation_command):
        print(output, end="")
    return results_path
