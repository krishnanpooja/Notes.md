Understand the data journey over a production system’s lifecycle and leverage ML metadata and enterprise schemas to address quickly evolving data.
- __Artifact__ - Each time a component produces a result, it generates an artifact. This includes basically everything that is produced by the pipeline, including the data in different stages of transformation, often as a result of feature engineering and the model itself and things like the schema, and metrics and so forth. Basically, everything that's produced, every result that is produced as an artifact.
- __Data Provenance__ -The terms data provenance and data lineage are basically synonyms and they're used interchangeably. Data provenance or lineage is a sequence of the artifacts that are created as we move through the pipeline.
                    - tracking those sequences is really key for debugging and understanding the training process and comparing different training runs that may happen months apart.

- __Data lineage__ is a great way for businesses and organizations to quickly determined how the Data has been used and which Transformations were performed as the Data moved through the pipeline. Data provenance is key to interpreting model results.
- __Data versioning__ - For one thing, because of the size of files that we deal with, which are typically or can be any way, much larger than a code file would ever be. Tools for data versioning are just starting to become available. These include DVC and Git LFS. LFS for large files Storage.

### Introduction to ML Metadata
<img width="907" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/fd708abc-18fb-46c2-ae73-0f2aead7c9fa">

<img width="932" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/b443c191-4a91-4e20-adba-a7c9da3d482b">

In addition to the executor where your code runs, each component also includes two additional parts, the driver and publisher. The executor is where the work of the component is done and that's what makes different components different. Whatever input is needed for the executor, is provided by the driver, which gets it from the metadata store. Finally, the publisher will push the results of running the executor back into the metadata store.

<img width="663" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/e395fd52-27e9-4034-b3ab-85408cfd2a25">

An artifact is an elementary unit of data that gets fed into the ML metadata store and as the data is consumed as input or generated as output of each component. Next there are executions. Each execution is a record of any component run during the ML pipeline workflow, along with its associated runtime parameters. Any artifact or execution will be associated with only one type of component. Artifacts and executions can be clustered together for each type of component separately. This grouping is referred to as the context. A context may hold the metadata of the projects being run, experiments being conducted, details about pipelines, etc. Each of these units can hold additional data describing it in more detail using properties. Next there are types, previously, you've seen several types of units that get stored inside the ML metadata. Each type includes the properties of that type. Lastly, we have relationships. Relationships store the various units getting generated or consumed when interacting with other units. For example, an event is the record of a relationship between an artifact and an execution. So, ML metadata stores a wide range of information about the results of the components and execution runs of a pipeline. It stores artifacts and it stores the executions of each component in the pipeline. It also stores the lineage information for each artifact that is generated. All of this information is represented in metadata objects and this metadata is stored in a back end storage solution.

<img width="952" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/596620a2-fd14-4654-9bab-2e1f7717d5cc">

### Schema Development
__Schema__ -  relational objects summarizing features in a given dataset project. like feature name, type, requrired or optional , valency etc

Over time the data evolve and gets skewed to handle this the schema also gets hanged accordingly.
The ML platform should have following features:
1. reliabilty during data evolution
2. scalabilty during evolution- large amt of data, variable traffic
3. Anamoly detection - data errors, update data schema to accomodate valid changes
4. Schema inspection "" - looking into schema versions, automate processes
  
### Schema Environments
Schema volves along with your business and data at times. Versionsing Schemas is important.

<img width="708" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/e7055f56-4410-4051-8a0f-eb3bee7d6484">

create different nviroments for different schemas like one for training with label and another for production (serving) without label

## Enterpise Data Storage
### Feature Stores
A feature store is a central repository for storing documented, curated and access controlled features.
- Using a feature store enables teams to share, discover and use highly curated features. A feature store makes it easy to discover and consume that feature and that can be both online or offline for both serving and training
- Feature Store helps store data can be used amongst different teams using same data but for different purpose
- It acts as interface between Feature Engineering and model Engineering
Advantages:
1. Avoid duplication
2. Control access
3. purge
4. offline feature processing
5. online feature usage by precomputing and sotring for low latency access

### Data Warehouses
A data warehouse is a technology that aggregates data from one or more sources so that it can be processed and analyzed.
- A data warehouse is usually meant for long running batch jobs and their storage is optimized for read operations.

__Key Features__:
  1.A data warehouse is subject oriented and the information that's stored in it revolves around a topic
  2. The data in data warehouse may be collected from multiple types of sources, such as relational databases or files and so forth.
  3.  The data collected in a data warehouse is usually timestamped to maintain the context of when it was generated.
  4.  Data warehouses are nonvolatile, which means the previous versions of data are not erased when new data is added

__Advantages__:
1. data warehouses offer enhanced ability to analyze your data by time stamping your data.
2. When you store your data in a data warehouse, it follows a consistent schema and that helps improve the data quality and consistency
3. Studies have shown that the return on investment for data warehouses tend to be fairly high for many use cases
4. the read and query efficiency from data warehouses is typically high, giving you fast access to your data.

<img width="909" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/d904054a-bba0-46ac-9fb6-69356e710487">


### Data Lakes
A data lake is a system or repository of data stored in its natural and raw format, which is usually in the form of blobs or files.
A data lake can include structured data like racial databases or semi-structured data like CSV files, or unstructured data like a collection of images or documents, and so forth. 

<img width="794" alt="image" src="https://github.com/krishnanpooja/Notes.md/assets/8016149/8f0ccba4-dbc0-4acf-bb75-2ed0a576212a">
