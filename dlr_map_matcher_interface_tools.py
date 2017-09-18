import rosparam
import os
import pickle

###########################################
# Actual interface functions, those need to be implemented.
#########################
def create_evaluation_function_sample(toplevel_directory, sample):
    """
    DLR-map-matcher-evaluation-specific code to get an evaluation_function Sample from a finished map matcher run.
    """
    sample.origin = toplevel_directory
    for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
        for file_name in files:
            if file_name == "statistics.pkl":
                pickle_path = os.path.join(root, file_name)
                eval_result_data = pickle.load(open(pickle_path, 'rb'), encoding='latin1')
                sample.translation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['translation'].values())
                sample.rotation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['rotation'].values())

    for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
        for file_name in files:
            if file_name == "params.yaml":
                # As soon as one params.yaml is found somewhere, we're done.
                # (because all other dirs have exactly the same parameters)
                sample.parameters = rosparam.load_file(os.path.join(root, file_name))[0][0]
                return

