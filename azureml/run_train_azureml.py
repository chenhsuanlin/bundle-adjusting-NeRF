# azureml/run-train.py
from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

import azureml.train.hyperdrive as hyperdrive
from azureml.train.hyperdrive.parameter_expressions import choice

import argparse
import json, os

def get_hyperparam_dict(run_config):
    param_dict = run_config["param_sampling"]
    for key, value in param_dict.items():
        exec("param_dict[key] = " + value)

    return param_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config_path", help="Path to the run config yaml file")
    parser.add_argument("run_name", help="Name of the newly created run")
    args = parser.parse_args()

    with open(args.run_config_path, "r") as f:
        run_config = json.load(f)

    workspace = Workspace(run_config["subscription_id"], run_config["resource_group"], run_config["workspace_name"])

    if "datastore" in run_config:
        print("Using non-default datastore")
        datastore = workspace.datastores[run_config["datastore"]]
    else:
        datastore = workspace.get_default_datastore()

    ds_path=""
    asset_dir = datastore.path(ds_path).as_mount()
    dataset = Dataset.File.from_files(path=(datastore, run_config["data_dir"]))
    
    compute_target = workspace.compute_targets[run_config["compute_name"]]
    project_folder = os.path.join(os.path.dirname(__file__), "..")
    env_requirements = os.path.join(project_folder, "requirements_azureml.yaml")

    barf_env = Environment.from_conda_specification(name="barf-env", file_path=env_requirements)

    arguments = run_config["script_args"]+[dataset.as_named_input('input').as_mount()]

    hyperparam_dict = get_hyperparam_dict(run_config)

    src = ScriptRunConfig(source_directory=project_folder,
                          script=run_config["entry_script"],
                          arguments=arguments,
                          compute_target=compute_target,
                          environment=barf_env)
    
    if len(hyperparam_dict) > 0:
        param_sampling = hyperdrive.GridParameterSampling(hyperparam_dict)
        if "max_concurrent_runs" in run_config:
            max_concurrent_runs = run_config["max_concurrent_runs"]
            print("max concurrent runs set")
        else:
            max_concurrent_runs = None
        hyperdrive_run_config = hyperdrive.HyperDriveConfig(run_config=src,
                                    hyperparameter_sampling=param_sampling,
                                    primary_metric_name=run_config["primary_metric_name"],
                                    primary_metric_goal=hyperdrive.PrimaryMetricGoal.MINIMIZE,
                                    max_total_runs=run_config["max_total_runs"],
                                    max_concurrent_runs=max_concurrent_runs)
    else:
        print("No parameters to sample running a non-hyperdrive job")
        hyperdrive_run_config = src

    experiment_name = os.getlogin() + "-" + run_config["experiment_name"]

    run_object = Experiment(workspace, name=experiment_name).submit(hyperdrive_run_config, tags={"run_name": args.run_name})
    
    import webbrowser
    webbrowser.open_new(run_object.get_portal_url())
