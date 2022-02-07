#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""NannyML module providing classes and utilities for dealing with model metadata."""

from enum import Enum
from typing import List, Optional

import pandas as pd

NML_METADATA_PARTITION_COLUMN_NAME = 'nml_meta_partition'
NML_METADATA_PREDICTION_COLUMN_NAME = 'nml_meta_prediction'
NML_METADATA_GROUND_TRUTH_COLUMN_NAME = 'nml_meta_ground_truth'
NML_METADATA_IDENTIFIER_COLUMN_NAME = 'nml_meta_identifier'
NML_METADATA_TIMESTAMP_COLUMN_NAME = 'nml_meta_timestamp'

NML_METADATA_COLUMNS = [
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_PREDICTION_COLUMN_NAME,
    NML_METADATA_GROUND_TRUTH_COLUMN_NAME,
    NML_METADATA_IDENTIFIER_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
]


# TODO wording
class FeatureType(str, Enum):
    """An enum indicating what kind of variable a given feature represents.

    The FeatureType enum is a property of a Feature. NannyML uses this information to select the best drift detection
    algorithms for each individual feature.

    We consider the following feature types:

    CONTINUOUS: numeric variables that have an infinite number of values between any two values.
    NOMINAL: has two or more categories, but there is no intrinsic ordering to the categories.
    ORDINAL: similar to a categorical variable, but there is a clear ordering of the categories.
    UNKNOWN: indicates NannyML couldn't detect the feature type with a high enough degree of certainty.
    """

    CONTINUOUS = 'continuous'
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'
    UNKNOWN = 'unknown'


class Feature:
    """Representation of a model feature.

    NannyML requires both model inputs and outputs to perform drift calculation and performance metrics.
    It needs to understand what features a model is made of and what kind of data they might contain.
    The Feature class allows you to provide this information.
    """

    def __init__(self, column_name: str, name: str, feature_type: FeatureType, description: str = None):
        """Creates a new Feature instance.

        The ModelMetadata class contains a list of Features that describe the values that serve as model input.

        Parameters
        ----------
        column_name : str
            The name of the column where the feature is found in the (to be provided) model input/output data.
        name : str
            A (human-friendly) name for the feature.
        feature_type : FeatureType
            The kind of values the data for this feature are.
        description : str
            Some additional information to display within results and visualizations.

        Returns
        -------
        feature: Feature

        """
        self.column_name = column_name
        self.name = name
        self.description = description
        self.feature_type = feature_type

    def __str__(self):
        """String representation of a single Feature."""
        strs = [
            f"Feature: {self.name}",
            '',
            f"{'Column name':25} {self.column_name:25}",
            f"{'Description':25} {self.description:25}",
            f"{'Type':25} {self.feature_type:25}",
            '',
        ]
        return str.join('\n', strs)


# TODO wording
class ModelMetadata:
    """The ModelMetadata class contains all the information nannyML requires.

    To understand the model inputs and outputs you wish it to process, nannyML needs to understand what your model
    (and hence also your model inputs/outputs) look like.
    The ModelMetadata class combines all the information about your model it might need. We call this the model
    metadata, since it does not concern the actual model (e.g .weights or coefficients) but generic information about
    your model.

    These properties are:
    - `model_name` : a human-readable name for the model
    - `model_purpose` : an optional description of the use for your model
    - `model_problem` : the kind of problem your model is trying to solve.
        We currently only support `binary_classification` problems but are planning to support more very soon!
    - `features` : the list of Features for the model
    - `identifier_column_name` : name of the column that contains a value that acts as an identifier for the
        observation, i.e. it is unique over all observations.
    - `prediction_column_name` : name of the column that contains the models' predictions
    - `ground_truth_column_name` : name of the column that contains the ground truth / target / actual.
    - `partition_column_name` : name of the column that contains the partition the observation belongs to.
        Allowed partition values are 'reference' and 'analysis'.
    - `timestamp_column_name` : name of the column that contains the timestamp indicating when the observation occurred.

    """

    # TODO wording
    def __init__(
        self,
        model_name: str,
        model_purpose: str = None,
        model_problem: str = 'binary_classification',
        features: List[Feature] = None,
        identifier_column_name: str = 'id',
        prediction_column_name: str = 'p',
        ground_truth_column_name: str = 'target',
        partition_column_name: str = 'partition',
        timestamp_column_name: str = 'date',
    ):
        """Creates a new ModelMetadata instance.

        Parameters
        ----------
        model_name : string
            A human-readable name for the model. Required.
        model_purpose : string
            An optional description of the use for your model. Optional
        model_problem : string
            The kind of problem your model is trying to solve. Optional, defaults to `binary_classification`.
        features : List[Feature]
            The list of Features for the model. Optional, defaults to `None`.
        identifier_column_name : string
            The name of the column that contains a value that acts as an identifier for the
            observation, i.e. it is unique over all observations. Optional, defaults to `id`
        prediction_column_name : string
            The name of the column that contains the models' predictions. Optional, defaults to `p`.
        ground_truth_column_name : string
            The name of the column that contains the ground truth / target / actual. Optional, defaults to `target`
        partition_column_name : string
            The name of the column that contains the partition the observation belongs to.
            Allowed partition values are 'reference' and 'analysis'. Optional, defaults to `partition`
        timestamp_column_name : string
            The name of the column that contains the timestamp indicating when the observation occurred.
            Optional, defaults to `date`.

        Returns
        -------
        metadata: ModelMetadata
        """
        self.id: int

        self.name = model_name
        self.model_purpose = model_purpose
        self.model_problem = model_problem

        self.identifier_column_name = identifier_column_name
        self.prediction_column_name = prediction_column_name
        self.ground_truth_column_name = ground_truth_column_name
        self.partition_column_name = partition_column_name
        self.timestamp_column_name = timestamp_column_name

        self.features = [] if features is None else features

    def __str__(self):
        """Returns a string representation of a ModelMetadata instance."""
        UNKNOWN = "~ UNKNOWN ~"
        strs = [
            f"Metadata for model {self.name}",
            '',
            '# Warning - unable to identify all essential data',
            f'# Please identify column names for all \'{UNKNOWN}\' values',
            '',
            f"{'Model purpose':25} {self.model_purpose or UNKNOWN:25}",
            f"{'Model problem':25} {self.model_problem or UNKNOWN:25}",
            '',
            f"{'Identifier column':25} {self.identifier_column_name or UNKNOWN:25}",
            f"{'Timestamp column':25} {self.timestamp_column_name or UNKNOWN:25}",
            f"{'Partition column':25} {self.partition_column_name or UNKNOWN:25}",
            f"{'Prediction column':25} {self.prediction_column_name or UNKNOWN:25}",
            f"{'Ground truth column':25} {self.ground_truth_column_name or UNKNOWN:25}",
            '',
            'Features',
            '',
            f"{'Name':20} {'Column':20} {'Type':15} {'Description'}",
        ]
        for f in self.features:
            strs.append(f"{f.name:20} {f.column_name:20} {f.feature_type or 'NA':15} {f.description}")
        return str.join('\n', strs)

    # def asdict(self) -> Dict[str, Any]:
    #     res = {}

    def feature(self, index: int = None, feature: str = None, column: str = None) -> Optional[Feature]:
        """A function used to access a specific model feature.

        Because a model might contain hundreds of features NannyML provides this utility method to filter through
        them and find the exact feature you need.

        Parameters
        ----------
        index : int
            Retrieve a Feature using its index in the features list.
        feature : str
            Retrieve a feature using its name.
        column : str
            Retrieve a feature using the name of the column it has in the model inputs/outputs.

        Returns
        -------
        feature: Feature
            A single Feature matching the search criteria. Returns `None` if none were found matching the criteria
            or no criteria were provided.

        """
        if feature:
            matches = [f for f in self.features if f.name == feature]
            return matches[0] if len(matches) != 0 else None

        if column:
            matches = [f for f in self.features if f.column_name == column]
            return matches[0] if len(matches) != 0 else None

        if index is not None:
            return self.features[index]

        else:
            return None

    def extract_metadata(self, data: pd.DataFrame, add_metadata: bool = True):
        """Tries to extract model metadata from a given data set.

        Manually constructing model metadata can be cumbersome, especially if you have hundreds of features.
        NannyML includes this helper function that tries to do the boring stuff for you using some simple rules.

        Parameters
        ----------
        data : DataFrame
            The dataset containing model inputs and outputs, enriched with the required metadata.
        add_metadata: bool, default=True
            Indicates if NannyML should add its own metadata columns to the original DataFrame.
            These are copies of the just identified metadata columns but with static names, for easier processing
            down the line.

        Returns
        -------
        metadata: ModelMetadata
            A fully initialized ModelMetadata instance.

        Notes
        -----
        NannyML can only make educated guesses as to what kind of data lives where. When NannyML feels to unsure
        about a guess, it will not use it.
        Be sure to always review the results of this method for their correctness and completeness.
        Adjust and complete as you see fit.
        """
        if len(data.columns) == 0:
            return None

        identifiers = _guess_identifiers(data)
        self.identifier_column_name = None if len(identifiers) == 0 else identifiers[0]  # type: ignore

        predictions = _guess_predictions(data)
        self.prediction_column_name = None if len(predictions) == 0 else predictions[0]  # type: ignore

        ground_truths = _guess_ground_truths(data)
        self.ground_truth_column_name = None if len(ground_truths) == 0 else ground_truths[0]  # type: ignore

        partitions = _guess_partitions(data)
        self.partition_column_name = None if len(partitions) == 0 else partitions[0]  # type: ignore

        timestamps = _guess_timestamps(data)
        self.timestamp_column_name = None if len(timestamps) == 0 else timestamps[0]  # type: ignore

        self.features = _extract_features(data)

        if add_metadata:
            self.enrich(data, in_place=True)

        return self

    def enrich(self, data: pd.DataFrame, in_place: bool = False) -> pd.DataFrame:
        """Creates copies of all metadata columns with fixed names.

        Parameters
        ----------
        data: DataFrame
            The data to enrich
        in_place: bool, default=False
            When `True` this function will modify the original DataFrame. When `False` it will operate on
            a copy of the DataFrame.

        Returns
        -------
        enriched_data: DataFrame
            A DataFrame that has all metadata present in NannyML-specific columns.
        """
        if not in_place:
            data = data.copy()

        data[NML_METADATA_IDENTIFIER_COLUMN_NAME] = data[self.identifier_column_name]
        data[NML_METADATA_TIMESTAMP_COLUMN_NAME] = data[self.timestamp_column_name]
        data[NML_METADATA_PREDICTION_COLUMN_NAME] = data[self.prediction_column_name]
        data[NML_METADATA_GROUND_TRUTH_COLUMN_NAME] = data[self.ground_truth_column_name]
        data[NML_METADATA_PARTITION_COLUMN_NAME] = data[self.partition_column_name]

        return data

    @property
    def categorical_features(self) -> List[Feature]:
        """Retrieves all categorical features.

        Returns
        -------
        features: List[Feature]
            A list of all categorical features
        """
        return [f for f in self.features if f.feature_type == FeatureType.NOMINAL]

    @property
    def continuous_features(self) -> List[Feature]:
        """Retrieves all continuous features.

        Returns
        -------
        features: List[Feature]
            A list of all continuous features
        """
        return [f for f in self.features if f.feature_type == FeatureType.CONTINUOUS]


def _guess_identifiers(data: pd.DataFrame) -> List[str]:
    def _guess_if_identifier(col: pd.Series) -> bool:
        return col.name in ['id', 'ident', 'identity', 'identifier', 'uid', 'uuid']

    return [col for col in data.columns if _guess_if_identifier(data[col])]


def _guess_timestamps(data: pd.DataFrame) -> List[str]:
    def _guess_if_timestamp(col: pd.Series) -> bool:
        return col.name in ['date', 'timestamp', 'ts', 'date', 'time']

    return [col for col in data.columns if _guess_if_timestamp(data[col])]


def _guess_predictions(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['p', 'pred', 'prediction', 'out', 'output']

    return [col for col in data.columns if _guess_if_prediction(data[col])]


def _guess_ground_truths(data: pd.DataFrame) -> List[str]:
    def _guess_if_ground_truth(col: pd.Series) -> bool:
        return col.name in ['target', 'ground_truth', 'actual', 'actuals']

    return [col for col in data.columns if _guess_if_ground_truth(data[col])]


def _guess_partitions(data: pd.DataFrame) -> List[str]:
    def _guess_if_partition(col: pd.Series) -> bool:
        return 'partition' in col.name

    return [col for col in data.columns if _guess_if_partition(data[col])]


def _guess_features(data: pd.DataFrame) -> List[str]:
    def _guess_if_feature(col: pd.Series) -> bool:
        return col.name not in _guess_identifiers(data) + _guess_partitions(data) + _guess_predictions(
            data
        ) + _guess_timestamps(data) + _guess_ground_truths(data)

    return [col for col in data.columns if _guess_if_feature(data[col])]


def _extract_features(data: pd.DataFrame) -> List[Feature]:
    feature_columns = _guess_features(data)
    if len(feature_columns) == 0:
        return []

    feature_types = _predict_feature_types(data[feature_columns])

    return [
        Feature(
            name=col,
            column_name=col,
            description=f'extracted feature: {col}',
            feature_type=feature_types.loc[col, 'predicted_feature_type'],
        )
        for col in feature_columns
    ]


INFERENCE_NUM_ROWS_THRESHOLD = 5
INFERENCE_HIGH_CARDINALITY_THRESHOLD = 0.1
INFERENCE_MEDIUM_CARDINALITY_THRESHOLD = 0.01
INFERENCE_LOW_CARDINALITY_THRESHOLD = 0.0


def _predict_feature_types(df: pd.DataFrame):
    def _determine_type(data_type, row_count, unique_count, unique_fraction):
        if row_count < INFERENCE_NUM_ROWS_THRESHOLD:
            return FeatureType.UNKNOWN

        if data_type == 'float64':
            return FeatureType.CONTINUOUS

        if unique_fraction >= INFERENCE_HIGH_CARDINALITY_THRESHOLD:
            return FeatureType.CONTINUOUS

        elif INFERENCE_LOW_CARDINALITY_THRESHOLD <= unique_fraction <= INFERENCE_MEDIUM_CARDINALITY_THRESHOLD:
            return FeatureType.NOMINAL

        else:
            return FeatureType.UNKNOWN

    # nunique: number of unique values
    # count: number of not-None values
    # size: number of values (including None)
    stats = df.agg(['nunique', 'count']).T
    stats['column_data_type'] = df.dtypes

    stats['unique_count_fraction'] = stats['nunique'] / stats['count']
    stats['predicted_feature_type'] = stats.apply(
        lambda r: _determine_type(
            data_type=r['column_data_type'],
            row_count=r['count'],
            unique_count=r['nunique'],
            unique_fraction=r['unique_count_fraction'],
        ),
        axis=1,
    )

    # Just for serialization purposes
    stats['column_data_type'] = str(stats['column_data_type'])

    return stats