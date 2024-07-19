# Databricks notebook source
# MAGIC %pip install fairlearn==0.10.0
# MAGIC %pip install pandas==1.0.5
# MAGIC %pip install numpy==1.19.1
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install protobuf==3.17.2

# COMMAND ----------

# from tigerml.core.reports import create_report
from utils import utils
import numpy as np
import pandas as pd
from MLCORE_SDK import mlclient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate, selection_rate
import matplotlib.pyplot as plt
import mlflow
from pyspark.sql.functions import col, when
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, ConfusionMatrixDisplay


# COMMAND ----------

def get_env_vault_scope():
    """
    Returns env and vault scope
    """
    import json
    env = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    ).split("/")[2]
    try:
        if len(dbutils.fs.ls('dbfs:/FileStore/jars/MLCORE_INIT/vault_check.json')) == 1:
            # if env == "qa":
            #     with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check_qa.json", "r") as file:
            #         vault_check_data = json.loads(file.read())
            # else:
            with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check.json", "r") as file:
                vault_check_data = json.loads(file.read())
            if "@" in env:
                return "qa", vault_check_data['client_name']
            return env, vault_check_data['client_name']
        else:
            return env, vault_scope
    except:
        return env, vault_scope

def json_str_to_pythontype(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param

def generate_run_notebook_url(job_id, run_id):
    """
    Generates the databricks job run notebook url in runtime
    """
    workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
    run_notebook_url = f"{workspace_url}/?o={workspace_id}#job/{job_id}/run/{run_id}"
    return run_notebook_url

def fetch_secret_from_dbutils(dbutils, key_name):
    _, vault_scope = get_env_vault_scope()
    return dbutils.secrets.get(scope=vault_scope, key=key_name)
    
def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
    client_id = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-id")
    client_secret = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-secret")
    quota_project_id = dbutils.secrets.get(
        scope=vault_scope, key="gcp-api-quota-project-id"
    )
    refresh_token = dbutils.secrets.get(scope=vault_scope, key="gcp-api-refresh-token")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials

def __upload_blob_to_azure(dbutils, container_name, blob_path, target_path):
    from azure.storage.blob import BlobServiceClient

    try:
        connection_string = fetch_secret_from_dbutils(
            dbutils, "az-api-storage-connection-string"
        )
        # Initialize the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get the container client for the blob object
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Upload the blob file
        with open(blob_path, "rb") as data:
            blob_client = container_client.get_blob_client(target_path)
            blob_client.upload_blob(data)

    except Exception as e:
        print(f"Error came while uploading blob object from azure : {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, blob_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = fetch_secret_from_dbutils(dbutils, "gcp-api-quota-project-id")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(blob_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp","azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az","azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)
def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
    client_id = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-id")
    client_secret = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-secret")
    quota_project_id = dbutils.secrets.get(
        scope=vault_scope, key="gcp-api-quota-project-id"
    )
    refresh_token = dbutils.secrets.get(scope=vault_scope, key="gcp-api-refresh-token")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials

def __upload_blob_to_azure(dbutils, container_name, blob_path, target_path):
    from azure.storage.blob import BlobServiceClient

    try:
        connection_string = fetch_secret_from_dbutils(
            dbutils, "az-api-storage-connection-string"
        )
        # Initialize the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get the container client for the blob object
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Upload the blob file
        with open(blob_path, "rb") as data:
            blob_client = container_client.get_blob_client(target_path)
            blob_client.upload_blob(data)

    except Exception as e:
        print(f"Error came while uploading blob object from azure : {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, blob_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = fetch_secret_from_dbutils(dbutils, "gcp-api-quota-project-id")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(blob_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp","azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az","azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)
def detect_categorical_cols(df, threshold=5):
    """
    Get the Categorical columns with greater than threshold percentage of unique values.
    This function returns the Categorical columns with the unique values in the column
    greater than the threshold percentage.
    Parameters
    ----------
    df: pyspark.sql.DataFrame
    threshold : int , default = 5
        threshold value in percentage
    Returns
    -------
    report_data : dict
        dictionary containing the Numeric column data.
    """
    df = df.toPandas()
    no_of_rows = df.shape[0]
    possible_cat_cols = (
        df.convert_dtypes()
        .select_dtypes(exclude=[np.datetime64, "float", "float64"])
        .columns.values.tolist()
    )
    temp_series = df[possible_cat_cols].apply(
        lambda col: (len(col.unique()) / no_of_rows) * 100 > threshold
    )
    cat_cols = temp_series[temp_series == False].index.tolist()
    return cat_cols


# COMMAND ----------

import json
def json_str_to_list(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param
env, vault_scope = get_env_vault_scope()

# COMMAND ----------

report_directory = dbutils.widgets.get("report_directory")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")
reserved_columns = ['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE']
train_output_path = dbutils.widgets.get("model_data_path")
features = dbutils.widgets.get("feature_columns").split(",")
target = [dbutils.widgets.get("target_columns")]
datalake_env = dbutils.widgets.get("datalake_env").lower()
cloud_provider = dbutils.widgets.get("cloud_provider")
sensitive_columns = json_str_to_list(dbutils.widgets.get("sensitive_variables"))
modelling_task_type = dbutils.widgets.get("modelling_task_type")

# COMMAND ----------

if datalake_env.lower() == "delta" : 
        bias_check_df = utils.df_read(data_path = train_output_path,
                                    spark = spark,
                                    resource_type=datalake_env
                                    ).select(features + target + reserved_columns)
else : 
    bias_check_df = utils.df_read(
        spark = spark,
        data_path=train_output_path.split(".")[-1],
        bq_database_name=train_output_path.split(".")[1],
        bq_project_id=train_output_path.split(".")[0],
        encrypted_service_account=dbutils.secrets.get(vault_scope,"gcp-service-account-encypted"),
        encryption_key=dbutils.secrets.get(vault_scope,"gcp-service-account-private-key"),
        resource_type=datalake_env).select(features + target + reserved_columns)
    
try :
    uc_container_name = str(dbutils.secrets.get(scope=vault_scope, key='uc-container-name'))
    uc_volume_name = str(dbutils.secrets.get(scope=vault_scope, key='uc-volume-name'))
except :
    uc_container_name = None
    uc_volume_name = None

# COMMAND ----------

if not sensitive_columns:
    sensitive_columns = detect_categorical_cols(bias_check_df.select(features))
total_records = bias_check_df.count()

# COMMAND ----------

if modelling_task_type.lower() == "classification" :
    df = bias_check_df
else:
    df = bias_check_df.toPandas()
    threshold_target = df[target].median()
    df[target] = (df[target] >= threshold_target).astype(int)
    df = spark.createDataFrame(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Preprocessing

# COMMAND ----------

dtype_filter =  [val for val in df.columns if val.startswith("dataset_type")] 

from pyspark.sql.functions import col

# Assuming df is your DataFrame
# Filter the DataFrame based on column 'c'
filtered_df_train = df.filter(col(dtype_filter[0]) ==  "train")

# Select all columns except 'c'
filtered_df_train = filtered_df_train.select([col_name for col_name in filtered_df_train.columns if col_name != dtype_filter[0]])

# COMMAND ----------

filtered_df_test = df.filter(col(dtype_filter[0]) ==  "test")

filtered_df_test = filtered_df_test.select([col_name for col_name in filtered_df_test.columns if col_name != dtype_filter[0]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Test Split

# COMMAND ----------

X_train = filtered_df_train.drop(target[0]).toPandas()
y_train = filtered_df_train.select(target[0]).toPandas()[target[0]] # makes this a Series
filtered_df_test = filtered_df_test.limit(500)
X_val  = filtered_df_test.drop(target[0]).toPandas()
y_val = filtered_df_test.select(target[0]).toPandas()[target[0]]

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Fairlearn's ThresholdOptimizer

# COMMAND ----------

from fairlearn.postprocessing import ThresholdOptimizer
import mlflow.pyfunc

class FairlearnThresholdWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, threshold_optimizer, sensitive_cols):
    self.threshold_optimizer = threshold_optimizer
    self.sensitive_cols = sensitive_cols
    
  def predict(self, context, data):
    return self.threshold_optimizer.predict(data.drop(self.sensitive_cols, axis=1), \
                                            sensitive_features=data[self.sensitive_cols], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Establishing a Baseline Model

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier

# Function that actually fits a model and returns its predictions, so we can swap it out later:
def predict_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, params):
  mlflow.xgboost.autolog(disable=False, log_models=True)
  # Balance the positive/negative classes evenly via weighting
  pos_weight = (len(y_train_xgb) / y_train_xgb.sum().item()) - 1
  model = XGBClassifier(use_label_encoder=False, n_jobs=4, n_estimators=2000, random_state=0, scale_pos_weight=pos_weight,
                        max_depth=int(params['max_depth']), learning_rate=params['learning_rate'], min_child_weight=params['min_child_weight'],
                        subsample=params['subsample'], colsample_bytree=params['colsample_bytree'])
  # n_estimators is high above, but, early stopping should stop long before that maximum!
  model.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], eval_metric="logloss", early_stopping_rounds=10)
  return model.predict(X_val_xgb), model

def run_data_experiment(data_experiment_tag, drop_cols=[], predict_fn=predict_xgboost):
  def train_model(params):
    
    mlflow.set_tag("data_experiment", data_experiment_tag)
    op = predict_fn(X_train.drop(drop_cols, axis=1), y_train, X_val.drop(drop_cols, axis=1), y_val, params)
    y_pred = op[0]
    f1 = f1_score(y_val, y_pred)
    
    #if type(op[1])== mlflow.pyfunc.PyFuncModel:
    if isinstance(op[1],FairlearnThresholdWrapper):
      mlflow.pyfunc.log_model("MLmodel", python_model=op[1])
    else:
      mlflow.sklearn.log_model(op[1], 'MLmodel')
    
    mlflow.log_metrics({ 'accuracy': accuracy_score(y_val, y_pred), 'f1_score': f1, 'recall': recall_score(y_val, y_pred) })
    return { 'status': STATUS_OK, 'loss': -f1 }

  search_space = {
    'max_depth':        hp.quniform('max_depth', 2, 10, 1),
    'learning_rate':    hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
    'min_child_weight': hp.qloguniform('min_child_weight', np.log(1), np.log(10), 1),
    'subsample':        hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
  }

  fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=2, trials=SparkTrials(parallelism=12))

# COMMAND ----------

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)

# COMMAND ----------

mlflow.set_experiment(notebook_path)

# COMMAND ----------

run_data_experiment("include_demographics")

# COMMAND ----------

import mlflow


def get_exp_id():
    # List all experiments
    experiments = mlflow.list_experiments()

    # Print experiment details
    for experiment in experiments:
        if "34. Bias_Evaluation_v2" in experiment.name:
            return experiment.experiment_id
        else:
            return None 
            

# COMMAND ----------

current_exp_id =  get_exp_id()

# COMMAND ----------

current_exp_id

# COMMAND ----------


# Finds the run in an experiment with a given tag and the highest F1 score
def find_best_run_id(data_experiment_tag):
  # TODO: change to your notebook's experiment ID, or path to experiment you use:
  return spark.read.format("mlflow-experiment").load(current_exp_id).\
     filter(col("tags.data_experiment") == data_experiment_tag).\
     orderBy(col("metrics.f1_score").desc()).\
     select("run_id").take(1)[0]['run_id']

# Finds the best run for a tag per above, runs the model on validation data and returns results
def get_recid_data(data_experiment_tag, drop_cols=[]):
  run_id = find_best_run_id(data_experiment_tag)
  recid_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/MLmodel")
  recid_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
  recid_df["prediction"] = recid_model.predict(X_val.drop(drop_cols, axis=1)).astype('int32')
  return recid_df

# Utility method to find best model by tag per above, run predictions, compute two confusion matrices by
# race and show them along with their difference
def show_confusion_matrix_diff(data_experiment_tag, drop_cols=[], feature="RACE"):
  recid_df = get_recid_data(data_experiment_tag, drop_cols)
  def get_cm(label):
    data_pd = recid_df[recid_df[feature] == label][[target[0], "prediction"]]
    return confusion_matrix(data_pd[target[0]], data_pd['prediction'], normalize='all')

  not_feature_cm = get_cm(0)
  feature_cm = get_cm(1)

  _, axes = plt.subplots(1, 3, figsize=(20,5), sharey='row')
  plt.rcParams.update({'font.size': 16})
  labels = ["0 Class", "1 Class"]
  ConfusionMatrixDisplay(not_feature_cm, display_labels=labels).plot(ax=axes[0], cmap='Blues', colorbar=False).\
    ax_.set_title(f"Not {feature}")
  ConfusionMatrixDisplay(feature_cm, display_labels=labels).plot(ax=axes[1], cmap='Blues', colorbar=False).\
    ax_.set_title(f"{feature}")
  ConfusionMatrixDisplay(feature_cm - not_feature_cm, display_labels=labels).plot(ax=axes[2], cmap='Purples', colorbar=False).\
    ax_.set_title("Difference") 
  fig = plt.gcf()
  plt.show()
  return fig

# Loads the best model for a tag per above, computes predictions and evaluates fairness metrics.
# Logs them back to MLflow and returns the result for display
def show_fairness_metrics(data_experiment_tag, feature, drop_cols=[]):
  recid_df = get_recid_data(data_experiment_tag, drop_cols)
  metrics = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "selection rate": selection_rate
  }
  mf = MetricFrame(metrics=metrics, y_true=y_val, y_pred=recid_df["prediction"], \
                   sensitive_features=X_val[feature], control_features=y_val).by_group
  # # Update the run with new metrics
  # (fpr_not_af_am, fpr_af_am, _, _) = mf['false positive rate']
  # (_, _, fnr_not_af_am, fnr_af_am) = mf['false negative rate']
  # run_id = find_best_run_id(data_experiment_tag)
  # with mlflow.start_run(run_id=run_id):
  #   mlflow.log_metrics({
  #     "Not Af-Am FPR": fpr_not_af_am,
  #     "Af-Am FPR": fpr_af_am,
  #     "Not Af-Am FNR": fnr_not_af_am,
  #     "Af-Am FNR": fnr_af_am
  #   })
  return mf

# COMMAND ----------

def show_confusion_matrix_diff(data_experiment_tag, drop_cols=[], feature="RACE"):
  recid_df = get_recid_data(data_experiment_tag, drop_cols)
  def get_cm(label):
    if "not" in label:
        label =  int(label.split("_")[-1])
        data_pd = recid_df[recid_df[feature] != label][[target[0], "prediction"]]
    else:
        label =  int(label)
        data_pd = recid_df[recid_df[feature] == label][[target[0], "prediction"]]
    return confusion_matrix(data_pd[target[0]], data_pd['prediction'], normalize='all')

  # Get unique values from the specified column
  unique_values = df.select(col(feature)).distinct().rdd.map(lambda row: row[0]).collect()

  # Convert the unique values to a list
  unique_labels = list(unique_values)

  for label in unique_labels:
    not_feature_cm = get_cm(f"not_{label}")
    feature_cm = get_cm(f"{label}")

  _, axes = plt.subplots(1, 3, figsize=(20,5), sharey='row')
  plt.rcParams.update({'font.size': 16})
  labels = ["0 Class", "1 Class"]
  ConfusionMatrixDisplay(not_feature_cm, display_labels=labels).plot(ax=axes[0], cmap='Blues', colorbar=False).\
    ax_.set_title(f"Not {feature}")
  ConfusionMatrixDisplay(feature_cm, display_labels=labels).plot(ax=axes[1], cmap='Blues', colorbar=False).\
    ax_.set_title(f"{feature}")
  ConfusionMatrixDisplay(feature_cm - not_feature_cm, display_labels=labels).plot(ax=axes[2], cmap='Purples', colorbar=False).\
    ax_.set_title("Difference") 
  fig = plt.gcf()
  plt.show()
  return fig


# COMMAND ----------

fg_include_demographics = []
for val in sensitive_columns:
    fg_include_demographics.append(show_confusion_matrix_diff("include_demographics", feature =val))

# COMMAND ----------

fairness_df_include_demographics = []
for val in sensitive_columns:
    fairness_df_include_demographics.append(show_fairness_metrics("include_demographics", feature= val))

# COMMAND ----------

# MAGIC %md Experiments without demographics 
# MAGIC

# COMMAND ----------

run_data_experiment("exclude_demographics", drop_cols = sensitive_columns)

# COMMAND ----------

fg_exclude_demographics = []
for val in sensitive_columns:
    fg_exclude_demographics.append(show_confusion_matrix_diff("exclude_demographics", feature =val, drop_cols=sensitive_columns))



# COMMAND ----------

fairness_df_exclude_demographics = []
for val in sensitive_columns:
    #fairness_df_include_demographics.append(show_fairness_metrics("include_demographics", feature= val))
    fairness_df_exclude_demographics.append(show_fairness_metrics("exclude_demographics", drop_cols= sensitive_columns,feature=val))


# COMMAND ----------

# MAGIC %md
# MAGIC ###  Explaining Effect of Race on Prediction

# COMMAND ----------

from shap import TreeExplainer, summary_plot
import mlflow

def draw_summary_plot(data_experiment_tag, drop_cols=[]):
  run_id = find_best_run_id(data_experiment_tag)
  model = mlflow.sklearn.load_model(f"runs:/{run_id}/MLmodel")

  train_sample = X_train.drop(drop_cols, axis=1)
  example_samples = np.random.choice(np.arange(len(X_val)), 500, replace=False)
  example = X_val.drop(drop_cols, axis=1).iloc[example_samples]
  fig, ax = plt.subplots()
  explainer = TreeExplainer(model, train_sample, model_output="probability")
  shap_values = explainer.shap_values(example, y=y_val.iloc[example_samples])
  summary_plot(shap_values, example, max_display=10, alpha=0.4, cmap="PiYG", plot_size=(14,6))
  #fig = plt.gcf()
  return fig
  
tree_explainer_fg = draw_summary_plot("include_demographics")

# COMMAND ----------

  
def predict_xgboost_fairlearn(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, params):
  mlflow.xgboost.autolog(disable=True)
  pos_weight = (len(y_train_xgb) / y_train_xgb.sum()) - 1
  # Can't use early stopping
  model = XGBClassifier(use_label_encoder=False, n_jobs=4, n_estimators=200, random_state=0, scale_pos_weight=pos_weight,
                        max_depth=int(params['max_depth']), learning_rate=params['learning_rate'], 
                        min_child_weight=params['min_child_weight'], 
                        subsample=params['subsample'], colsample_bytree=params['colsample_bytree'])
  # Wrap in an optimizer that prioritizes equalized odds while trying to maintain accuracy
  optimizer = ThresholdOptimizer(estimator=model, constraints="equalized_odds", objective="accuracy_score", predict_method="predict_proba")
  sensitive_cols = sensitive_columns
  optimizer.fit(X_train_xgb.drop(sensitive_cols, axis=1), y_train_xgb, sensitive_features=X_train_xgb[sensitive_cols])
  wrapper = FairlearnThresholdWrapper(optimizer, sensitive_cols)
  #mlflow.pyfunc.log_model("MLmodel", python_model=wrapper)
  return wrapper.predict(None, X_val_xgb), wrapper

# COMMAND ----------

run_data_experiment("include_demographics_fairlearn", predict_fn=predict_xgboost_fairlearn)

# COMMAND ----------

fg_include_demographics_mitigated = []
for val in sensitive_columns:
    fg_include_demographics_mitigated.append(show_confusion_matrix_diff("include_demographics_fairlearn", feature =val))

# COMMAND ----------

fairness_df_include_demographics_mitigated = []
for val in sensitive_columns:
    #fairness_df_include_demographics.append(show_fairness_metrics("include_demographics", feature= val))
    fairness_df_include_demographics_mitigated.append(show_fairness_metrics("include_demographics_fairlearn",feature=val))
#show_fairness_metrics("with_demographics_fairlearn3")

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import infer_signature
import pandas as pd
from shap import TreeExplainer

class SHAPCorrectedXGBoostModel(PythonModel):
  
  def __init__(self, booster, explainer, model_columns):
    self.booster = booster
    self.explainer = explainer
    self.sum_shap_indices = [model_columns.tolist().index(c) for c in sensitive_columns]
    
  def predict(self, context, model_input):
    predicted_probs = self.booster.predict_proba(model_input)[:,1]
    shap_values = self.explainer.shap_values(model_input)
    corrected_probs = predicted_probs - shap_values[:,self.sum_shap_indices].sum(axis=1)
    return pd.DataFrame((corrected_probs >= 0.5).astype('int32'))


with mlflow.start_run():
  mlflow.set_tag("data_experiment", "include_demographics_shap")
  run_id = find_best_run_id("include_demographics")
  booster = mlflow.sklearn.load_model(f"runs:/{run_id}/MLmodel")
  explainer = TreeExplainer(booster, X_train, model_output="probability")
  mlflow.pyfunc.log_model("MLmodel", python_model=SHAPCorrectedXGBoostModel(booster, explainer, X_train.columns),
                          input_example=X_val.head(5),
                          signature=infer_signature(X_val.head(20), pd.DataFrame([0] * 20, columns=["prediction"]))) # dummy for type inference

# COMMAND ----------

fg_include_demographics_shap = []
for val in sensitive_columns:
    fg_include_demographics_shap.append(show_confusion_matrix_diff("include_demographics_shap", feature =val))

# COMMAND ----------

#show_fairness_metrics("trial_demographics_shap1")
fairness_df_include_demographics_shap = []
for val in sensitive_columns:
    #fairness_df_include_demographics.append(show_fairness_metrics("include_demographics", feature= val))
    fairness_df_include_demographics_shap.append(show_fairness_metrics("include_demographics_shap",feature=val))


# COMMAND ----------

from pyspark.sql.functions import col, datediff, udf

display(
  spark.read.format("mlflow-experiment").load(current_exp_id).\
     filter(col("metrics.`Af-Am FPR`") > 0).\
     orderBy("start_time").\
     select("tags.data_experiment", "metrics.`Not Af-Am FPR`", "metrics.`Af-Am FPR`", "metrics.`Not Af-Am FNR`", "metrics.`Af-Am FNR`")
)

# COMMAND ----------

from shap import TreeExplainer
from typing import Iterator
import pandas as pd

run_id = find_best_run_id("include_demographics")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/MLmodel")
explainer = TreeExplainer(model, X_val, model_output="probability")

def apply_shap(data_it: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
  for data in data_it:
    yield pd.DataFrame(explainer.shap_values(data.drop(target, axis=1), y=data[target[0]]))
    
shap_values_df = filtered_df_test.mapInPandas(apply_shap, schema=", ".join([f"{c} double" for c in X_val.columns])).cache()
display(shap_values_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TSNE SHAP

# COMMAND ----------

import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# for demo/viz purposes, just pull a thousand examples or so:
shap_values_sample = shap_values_df.sample(0.1, seed=123).toPandas()
embedded = TSNE(n_components=2, init='pca', n_jobs=-1, random_state=42).fit_transform(shap_values_sample)

sns.set(rc = {'figure.figsize':(16,9)})
sns.scatterplot(x=embedded[:,0], y=embedded[:,1], \
                hue=shap_values_sample[sensitive_columns[0]], \
                size=shap_values_sample[sensitive_columns[1]], sizes=(10,100))
fig_tnse = plt.gcf()
plt.legend(loc='upper right')

# COMMAND ----------

dict1 = {
    "Confusion Metrics difference":
        {f"{col}" :   fg_include_demographics[ind] for ind, col in enumerate(sensitive_columns)},
    "Fairness metrics": 
        {f"{col}" :   fairness_df_include_demographics[ind] for ind, col in enumerate(sensitive_columns)},
    "tree_explainer_plot" : tree_explainer_fg
}

dict2 = {
    "Confusion Metrics difference":
        {f"{col}" :   fg_exclude_demographics[ind] for ind, col in enumerate(sensitive_columns)},
    "Fairness metrics": 
        {f"{col}" :   fairness_df_exclude_demographics[ind] for ind, col in enumerate(sensitive_columns)}
}

dict3 = {
    "Confusion Metrics difference":
        {f"{col}" :   fg_include_demographics_mitigated[ind] for ind, col in enumerate(sensitive_columns)},
    "Fairness metrics": 
        {f"{col}" :   fairness_df_include_demographics_mitigated[ind] for ind, col in enumerate(sensitive_columns)},
    "tree_explainer_plot" : tree_explainer_fg
}

dict4 = {
    "Confusion Metrics difference with Sensitive features with SHAP value correction":
        {f"{col}" :   fg_include_demographics_shap[ind] for ind, col in enumerate(sensitive_columns)},
    "Fairness metrics with Sensitive features": 
        {f"{col}" :   fairness_df_include_demographics_shap[ind] for ind, col in enumerate(sensitive_columns)},
    "tree_explainer_plot" : tree_explainer_fg
}

dict5 = {
    "Shap values visualization via TSNE" : fig_tnse
}

report = {
    "Metrics with Sensitive features" : dict1,
    "Metrics without Sensitive features": dict2, 
    "Metrics post bias mitigation with sensitive features" : dict3,
    "Metrics post SHAP correction with sensitive features" : dict4,
    "TSNE visualization of SHAP values" :  fig_tnse
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 1. Prepare a case study : outcome of each chart has to be listed.
# MAGIC 2. Convert it into html

# COMMAND ----------

from tigerml.core.reports import create_report

report_name = f"BiasReport_V2_{int(time.time())}"
create_report(
    report,
    name=report_name,
    path="/dbfs/FileStore/MONITORING/bias_report",
    format=".html",
    columns= 1 ,
)

# COMMAND ----------

from MLCORE_SDK.helpers.auth_helper import get_container_name

az_container_name = get_container_name(dbutils)

if cloud_provider.lower() == "gcp" : 
        container_name=f"{az_container_name}_dev"
else :
        container_name = az_container_name

report_path = report_path = f"/dbfs/FileStore/MONITORING/bias_report/{report_name}"

if cloud_provider.lower() == "databricks_uc" : 
    dbutils.fs.mv("dbfs:" + report_path.split('/dbfs')[-1],f"dbfs:/mnt/{uc_container_name}/{uc_volume_name}/{report_directory}/Bias_Evaluation/BiasReport_V2_{int(time.time())}.html")
else:
    upload_blob_to_cloud(
        container_name=container_name,
        blob_path=f"{report_path}.html",
        dbutils = dbutils ,
        target_path = f"{report_directory}/Bias_Evaluation/BiasReport_V2_{int(time.time())}.html",
        resource_type = cloud_provider)

dbutils.fs.rm("dbfs:" + report_path.split('/dbfs')[-1], True)
print(f"report_directory : {report_directory}")
