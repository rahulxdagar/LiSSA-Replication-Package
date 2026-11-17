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
    "name" : "mock",
    "args" : { }
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
* True Positives: 116
* False Positives: 2504
* False Negatives: 170
* Precision: 0.044274809160305344
* Recall: 0.40559440559440557
* F1: 0.07983482450103235
