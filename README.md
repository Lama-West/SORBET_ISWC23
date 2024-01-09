# SORBET: a Siamese Network for Ontology Embeddings Using a Distance-based Regression Loss and BERT

### Overview

  SORBET is an Ontology Embedding algorithm using a novel approach and achieving high-quality concept representations. This documentation provides information on how to run and customize the SORBET codebase.

  Paper: [SORBET: a Siamese Network for Ontology Embeddings Using a Distance-based Regression Loss and BERT](https://link.springer.com/chapter/10.1007/978-3-031-47240-4_30)

  LAMA-WeSt lab: http://www.labowest.ca/

### Getting Started


To use SORBET, follow these steps:

1. Clone the repository:
  ```bash
git clone https://github.com/your-username/sorbet.git
cd sorbet
  ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running SORBET

#### 1. Adding Ontologies to config.json
To run SORBET, you must first add your ontology configuration in the config.json file
Below is an example of a configuration:

```config.json
{
  ...
"bioml_neoplas":{


        "ontologies_folder": "../datasets/bio-ml/snomed-ncit.neoplas",
        "alignments_folder": "../datasets/bio-ml/snomed-ncit.neoplas/refs_equiv/",
    
        "ontologies_in_alignment": ["snomed.neoplas", "ncit.neoplas"],
        "ontologies" : ["snomed.neoplas.owl", "ncit.neoplas.owl"],
        "alignments" : ["train.tsv"],
        "test_alignments": "test.tsv",
        
        "parsing_parameters":
            {
            "use_synonyms": 1,
            "exclude_classes" : ["SynonymType", "Synonym", "Subset", "ObsoleteClass", "Definition", "DbXref"],
            "subclass_of_properties": ["UNDEFINED_part_of"],
            "autocorrect": 0,
            "synonym_extension": 0,
            }
    
        },
  ...
}
```

Description of the parameters:

  * **ontologies_folder**: Path to the folder containing ontology files. (Required)
  * **ontologies**: List of ontology file names to load. If not specified, all ontologies in the folder will load
  * parsing_parameters: Parsing parameters for ontology processing, including:
      * **use_synonyms**: Flag indicating whether to parse synonyms or not.
      * **exclude_classes**: List of classes to ignore
      * **subclass_of_properties**: List of properties to consider as *subclass_of* properties when doing Tree Walks and Semi-negative sampling 
      * **autocorrect**: Flag indicating whether to perform autocorrection on textual data.
      * **synonym_extension**: Flag indicating whether to search for synonyms of concepts by prompting a flan-t5 model.
      * **accronym_resolution**: Flag indicating whether to perform a very simple acronym resolution algorithm.
   
  The following parameters are related to inference tasks (Ontology alignment and Subsumption prediction)
  * **alignments_folder**: Path to the folder containing alignment files. 
  * **ontologies_in_alignment**: Specifiy explicitly which 2 ontologies are to align
  * **alignments**: List of alignment file names to use in the alignment folder
  * **test_alignments**: File name for test alignments


#### 2. Modify train.py
Before starting the training, you must modify the **trainer** variable in the train.py file.
**trainer** is an instance of the **TrainPipeline** class.

The default configuration is as follows:

```python
trainer = TrainPipeline(
    config, 
    ["conference", "anatomy"],
    extra_tracks=None,

    # SORBET Training parameters
    epochs=60,
    lr=1e-6,
    save_model_checkpoints=-1,
    save_embeddings_checkpoints=-1,

    model=SORBET(from_pretrained=config["General"]["model"], pooling_strategy="only_concept"),
    train_walks=TreeWalkConfig(n_branches=(0, 5), use_synonyms=True),
    loader_config={
        "iir": 0.8,
        "inter_soft_r": 0.5,
        "intra_soft_r": 0.2,
        "negative_sampling_strategy": "ontologic_relation",
        "no_hard_negative_samples": False,
        "epoch_over_alignments": False,
        "A": 5,
        "batch_size": 32,
        "n_alignments_per_batch": 8
    },

    # Inference on Ontology Alignment or Subsumption prediction tasks for testing
    run_tasks=True,
    test_size=1.0,
    consider_train_set=False,
    inference_walks=TreeWalkConfig(strategy=WalkStrategy.ONTOLOGICAL_RELATIONS, n_branches=5),
    inference_config={
        "candidate_selector": None,
        "string_matching_optimization": False,
        "matcher": GreedyMatcher(),
        "thresholds": [0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85]
    },
    metrics_config={
        "results_files_path": "./result_alignments",
        "write_rdf": False,
        "write_tsv": False,
        "write_ranking": False,
        "hits": [1, 3, 5, 10],
        "debug_files_path": "./debug"
    },
    tensorboard_writer=writer
)

```

Parameters of TrainPipeline


The **TrainPipeline** class in the SORBET codebase accepts various parameters, allowing users to customize the training pipeline. Below is a detailed description of each parameter:


  * **config**: Ontologies configuration file from config.json
  * **ontologies**: List of ontologies names to be processed, must be keys from config.json (e.g., ["conference", "anatomy"]).
  * **extra_tracks**: Additional ontology tracks only to be used in training. Their final embeddings will not be saved.
  
  * **epochs**: Number of training epochs.
  * **lr**: Learning rate.
  * **save_folder**: Path to the folder where models and embeddings are saved.
  * **save_model_checkpoints**: Frequency of saving model checkpoints (-1 to only save final model, 0 to disable).
  * **save_embeddings_checkpoints**: Frequency of saving embeddings checkpoints (-1 to only save final embeddings, 0 to disable).

  * **model**: Any Model class extending BaseModel. Default is SORBET
  * **train_walks**: TreeWalkConfig object. Configuration for tree walks during training.
  * **loader_config**: Loader hyperparameters.
    * **iir**: Inter-intra sampling ratio. 1.0 will only sample inter-ontology.
    * **inter_soft_r**: Inter-ontology negative sampling rate.
    * **intra_soft_r**: Intra-ontology negative sampling rate.
    * **semi_negative_hop_strategy**: Type of hop for semi-negative sampling. Value can be: "all" for relation, "ontologic_relation" for *subclass_of* relations only, "lineage_path" for any lineage_path, "lineage_path_and_childs" for lineage path and descendance. 
    * **no_hard_negative_samples**: Flag indicating whether to use hard negative samples. (Not used in Semi-Negative sampling)
    * **epoch_over_alignments**: Flag indicating whether to iterate over alignments or concepts in each epoch.
    * **A**: Value for SORBET's hyper-parameter A.
    * **batch_size**: Batch size for training.
    * **n_alignments_per_batch**: Number of positive alignments per batch.

  Parameters below are for testing the embeddings on inference tasks (Ontology alignemnt and Subsumption prediction).
  If you only wish to use SORBET to get embeddings, set run_tasks to False.
  
  * **run_tasks**: Flag indicating whether to perform inference tasks (ontology alignment or subsumption).
  * **test_size**: Proportion of the dataset used for testing (1.0 for using the entire dataset).
  * **consider_train_set**: Flag indicating whether to consider the training set during testing.
  * **inference_walks**: Configuration for tree walks during inference.
  * **inference_config**: Configuration for inference tasks, including candidate selection and matching thresholds.
    * **candidate_selector**: Candidate selector to use before the model inference step.
    * **string_matching_optimization**: Flag indicating whether to consider string equality a 1.0 match, for optimization purposes.
    * **matcher**: An instance of a Matcher class. Default is GreedyMatcher.
    * **thresholds**: Thresholds to test for the matcher.

  * **metrics_config**: Configuration for metrics calculation and result output.
      * **results_files_path**: Path for storing result alignments.
      * **write_rdf**: Flag indicating whether to write result alignments in a RDF file.
      * **write_tsv**: Flag indicating whether to write result alignments in a TSV file.
      * **write_ranking**: Flag indicating whether to write ranking files.
      * **hits**: List of hit values for metrics hits@K (e.g., [1, 3, 5, 10]).
      * **debug_files_path**: Path for storing debug files.
  
  * **tensorboard_writer**: Tensorboard writer for logging training information.


#### 3. Start training

You can now start the training by running train.py
```bash
python train.py
```




