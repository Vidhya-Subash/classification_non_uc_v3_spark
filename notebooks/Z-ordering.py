# Databricks notebook source
import yaml
import ast

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none": 
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

# MAGIC %md
# MAGIC ## data_engineering input

# COMMAND ----------

raw_table_configs_ft = solution_config["data_engineering_ft"]["datalake_configs"]["input_tables"]
raw_primary_key_ft=raw_table_configs_ft["source_1"]["primary_keys"]
raw_table_configs_gt = solution_config["data_engineering_gt"]["datalake_configs"]["input_tables"]
raw_primary_key_gt=raw_table_configs_ft["source_1"]["primary_keys"]

# COMMAND ----------

raw_table_path_ft=get_name_space(raw_table_configs_ft)
raw_table_path_gt=get_name_space(raw_table_configs_gt)

# COMMAND ----------

table_path = raw_table_path_ft['source_1']

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {raw_table_path_ft['source_1']}
    ZORDER BY (age)
""")

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {raw_table_path_gt['source_1']}
    ZORDER BY ({raw_primary_key_gt})
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## data_engineering output

# COMMAND ----------

feature_table_configs_ft = solution_config["data_engineering_ft"]["datalake_configs"]['output_tables']
feature_primary_key_ft=feature_table_configs_ft["output_1"]["primary_keys"]
feature_table_configs_gt = solution_config["data_engineering_gt"]["datalake_configs"]['output_tables']
feature_primary_key_gt=feature_table_configs_gt["output_1"]["primary_keys"]

# COMMAND ----------

feature_table_path_ft=get_name_space(feature_table_configs_ft)
feature_table_path_gt=get_name_space(feature_table_configs_gt)

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {feature_table_path_ft['output_1']}
    ZORDER BY (age)
""")

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {feature_table_path_gt['output_1']}
    ZORDER BY ({feature_primary_key_gt})
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering output (fe)

# COMMAND ----------

transformed_table_configs_ft = solution_config["feature_pipelines_ft"]["datalake_configs"]['output_tables']
transformed_primary_key_ft=transformed_table_configs_ft["output_1"]["primary_keys"]
transformed_table_configs_gt = solution_config["feature_pipelines_gt"]["datalake_configs"]['output_tables']
transformed_primary_key_gt=transformed_table_configs_gt["output_1"]["primary_keys"]

# COMMAND ----------

transformed_table_path_ft=get_name_space(transformed_table_configs_ft)
transformed_table_path_gt=get_name_space(transformed_table_configs_gt)

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {transformed_table_path_ft['output_1']}
    ZORDER BY (age)
""")

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {transformed_table_path_gt['output_1']}
    ZORDER BY ({transformed_primary_key_gt})
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Output

# COMMAND ----------


train_table_configs = solution_config["train"]["datalake_configs"]['output_tables']
train_primary_key=train_table_configs["output_1"]["primary_keys"]

# COMMAND ----------

train_table_path=get_name_space(train_table_configs)

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {train_table_path['output_1']}
    ZORDER BY (age)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering output (dpd)

# COMMAND ----------

dpd_table_configs_ft = solution_config["data_prep_deployment_ft"]["datalake_configs"]['output_tables']
dpd_primary_key_ft=dpd_table_configs_ft["output_1"]["primary_keys"]
dpd_table_configs_gt = solution_config["data_prep_deployment_gt"]["datalake_configs"]['output_tables']
dpd_primary_key_gt=dpd_table_configs_gt["output_1"]["primary_keys"]

# COMMAND ----------

dpd_table_path_ft=get_name_space(dpd_table_configs_ft)
dpd_table_path_gt=get_name_space(dpd_table_configs_gt)

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {dpd_table_path_ft['output_1']}
    ZORDER BY (age)
""")

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {dpd_table_path_gt['output_1']}
    ZORDER BY ({dpd_primary_key_gt})
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference output

# COMMAND ----------

Infrence_table_configs = solution_config["inference"]["datalake_configs"]['output_tables']
Infrence_primary_key=Infrence_table_configs["output_1"]["primary_keys"]

Infrence_table_path_gt=get_name_space(Infrence_table_configs)

# COMMAND ----------

spark.sql(f"""
    OPTIMIZE {Infrence_table_path_gt['output_1']}
    ZORDER BY (age)
""")
