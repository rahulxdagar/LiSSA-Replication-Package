## Configuration
```json
{
  "cache_dir" : "./cache-local-r2c/SMOS-133742243",
  "gold_standard_configuration" : {
    "hasHeader" : false,
    "path" : "./datasets/req2code/SMOS/answer.csv"
  },
  "source_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "requirement",
      "path" : "./datasets/req2code/SMOS/UC"
    }
  },
  "target_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "source code",
      "path" : "./datasets/req2code/SMOS/CC"
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
* # TraceLinks (GS): 1044
* # Source Artifacts: 67
* # Target Artifacts: 100
## Results
* True Positives: 467
* False Positives: 873
* False Negatives: 577
* Precision: 0.3485074626865672
* Recall: 0.44731800766283525
* F1: 0.3917785234899329
