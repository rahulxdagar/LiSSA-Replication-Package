## Configuration
```json
{
  "cache_dir" : "./cache-local-r2c/iTrust-133742243",
  "gold_standard_configuration" : {
    "hasHeader" : false,
    "path" : "./datasets/req2code/iTrust/answer.csv"
  },
  "source_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "requirement",
      "path" : "./datasets/req2code/iTrust/UC"
    }
  },
  "target_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "source code",
      "path" : "./datasets/req2code/iTrust/CC"
    }
  },
  "source_preprocessor" : {
    "name" : "artifact",
    "args" : { }
  },
  "target_preprocessor" : {
    "name" : "artifact",
    "args" : { }
  },
  "embedding_creator" : {
    "name" : "ollama",
    "args" : {
      "model" : "bge-m3",
      "num_ctx" : "8192"
    }
  },
  "source_store" : {
    "name" : "custom",
    "args" : { }
  },
  "target_store" : {
    "name" : "custom",
    "args" : {
      "max_results" : "20"
    }
  },
  "classifier" : {
    "name" : "reasoning_ollama",
    "args" : {
      "model" : "qwen2.5-coder:14b",
      "seed" : "133742243",
      "num_ctx" : "16384",
      "num_predict" : "4096",
      "prompt" : "Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' ",
      "use_original_artifacts" : "false",
      "use_system_message" : "true"
    }
  },
  "result_aggregator" : {
    "name" : "any_connection",
    "args" : {
      "source_granularity" : "0",
      "target_granularity" : "0"
    }
  },
  "tracelinkid_postprocessor" : {
    "name" : "req2code",
    "args" : { }
  }
}
```

## Stats
* # TraceLinks (GS): 286
* # Source Artifacts: 131
* # Target Artifacts: 226
## Results
* True Positives: 54
* False Positives: 385
* False Negatives: 232
* Precision: 0.12300683371298406
* Recall: 0.1888111888111888
* F1: 0.14896551724137932
