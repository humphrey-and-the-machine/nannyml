.. _california-housing:

==========================
California Housing Dataset
==========================

We are using the `California Housing Dataset`_ to create a real data example dataset for
NannyML. There are three steps needed for this process:

- Enriching the data
- Training a Machine Learning Model
- Meeting NannyML Data Requirements


Let’s start by loading the dataset:

.. code-block:: python

    >>> # Import required libraries
    >>> import pandas as pd
    >>> import numpy as np
    >>> import datetime as dt

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import roc_auc_score

    >>> cali = fetch_california_housing(as_frame=True)
    >>> df = pd.concat([cali.data, cali.target], axis=1)
    >>> df.head(2)

+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------+
|    |   MedInc |   HouseAge |   AveRooms |   AveBedrms |   Population |   AveOccup |   Latitude |   Longitude |   MedHouseVal |
+====+==========+============+============+=============+==============+============+============+=============+===============+
|  0 |   8.3252 |         41 |    6.98413 |     1.02381 |          322 |    2.55556 |      37.88 |     -122.23 |         4.526 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------+
|  1 |   8.3014 |         21 |    6.23814 |     0.97188 |         2401 |    2.10984 |      37.86 |     -122.22 |         3.585 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------+


Enriching the data
==================

The things that need to be added to the dataset are:

- A time dimension
- Splitting the data into reference and analysis sets
- A binary classification target

.. code-block:: python

    >>> # add artificial timestamp
    >>> timestamps = [dt.datetime(2020,1,1) + dt.timedelta(hours=x/2) for x in df.index]
    >>> df['timestamp'] = timestamps

    >>> # add partitions
    >>> train_beg = dt.datetime(2020,1,1)
    >>> train_end = dt.datetime(2020,5,1)
    >>> test_beg = dt.datetime(2020,5,1)
    >>> test_end = dt.datetime(2020,9,1)
    >>> df.loc[df['timestamp'].between(train_beg, train_end, inclusive='left'), 'partition'] = 'train'
    >>> df.loc[df['timestamp'].between(test_beg, test_end, inclusive='left'), 'partition'] = 'test'
    >>> df['partition'] = df['partition'].fillna('production')

    >>> # create new classification target - house value higher than mean
    >>> df_train = df[df['partition']=='train']
    >>> df['clf_target'] = np.where(df['MedHouseVal'] > df_train['MedHouseVal'].median(), 1, 0)
    >>> df = df.drop('MedHouseVal', axis=1)
    >>> del df_train

Training a Machine Learning Model
=================================

.. code-block:: python

    >>> # fit classifier
    >>> target = 'clf_target'
    >>> meta = 'partition'
    >>> features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']


    >>> df_train = df[df['partition']=='train']

    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(df_train[features], df_train[target])
    >>> df['y_pred_proba'] = clf.predict_proba(df[features])[:,1]

    >>> # Check roc auc score
    >>> for partition_name, partition_data in df.groupby('partition', sort=False):
    ...     print(partition_name, roc_auc_score(partition_data[target], partition_data['y_pred_proba']))
    train 1.0
    test 0.8737681614409617
    production 0.8224322932364313

Meeting NannyML Data Requirements
=================================

The data are now being splitted so they can be in a form required by NannyML.

.. code-block:: python

    >>> df_for_nanny = df[df['partition']!='train'].reset_index(drop=True)
    >>> df_for_nanny['partition'] = df_for_nanny['partition'].map({'test':'reference', 'production':'analysis'})
    >>> df_for_nanny['identifier'] = df_for_nanny.index

    >>> reference = df_for_nanny[df_for_nanny['partition']=='reference'].copy()
    >>> analysis = df_for_nanny[df_for_nanny['partition']=='analysis'].copy()
    >>> analysis_target = analysis[['identifier', 'clf_target']].copy()
    >>> analysis = analysis.drop('clf_target', axis=1)

The ``reference`` dataframe represents the reference :term:`Partition` and the ``analysis``
dataframe represents the analysis partition. The ``analysis_target`` dataframe contains the targets
for the analysis partition that is provided separately.


.. _California Housing Dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
