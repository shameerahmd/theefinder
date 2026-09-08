import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


# =========================================================
# THEEFINDER
# TRAIN VERIFIED STAGE-A AND STAGE-B MODELS
# =========================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "training"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
)


INPUT_FILE = (
    TRAINING_DIR
    / "theefinder_verified_ml_features.csv"
)


STAGE_A_MODEL_FILE = (
    MODEL_DIR
    / "stage_a_random_forest.joblib"
)


STAGE_B_MODEL_FILE = (
    MODEL_DIR
    / "stage_b_industrial_random_forest.joblib"
)


STAGE_A_PREDICTIONS_FILE = (
    TRAINING_DIR
    / "stage_a_cv_predictions.csv"
)


STAGE_B_PREDICTIONS_FILE = (
    TRAINING_DIR
    / "stage_b_cv_predictions.csv"
)


STAGE_A_IMPORTANCE_FILE = (
    TRAINING_DIR
    / "stage_a_feature_importance.csv"
)


STAGE_B_IMPORTANCE_FILE = (
    TRAINING_DIR
    / "stage_b_feature_importance.csv"
)


METRICS_FILE = (
    TRAINING_DIR
    / "model_evaluation_summary.json"
)


EXPECTED_ROWS = 76

RANDOM_STATE = 42

N_SPLITS = 5


# =========================================================
# Stage-A features
#
# IMPORTANT:
# No persistence/history features are allowed here.
#
# This prevents collection-method leakage because
# Forest/Agriculture do not have equivalent historical
# baselines.
# =========================================================


STAGE_A_FEATURES = [

    "frp",

    "confidence_score",
    "confidence_available",

    "is_daytime",
    "daynight_available",

    "distance_to_industry_m",
    "industry_distance_available",
    "industrial_proximity_score",
    "industrial_feature_count_5km",

    "tree_cover_pct",
    "shrubland_pct",
    "grassland_pct",
    "cropland_pct",
    "built_up_pct",
    "bare_sparse_pct",
    "water_pct",
    "wetland_pct",
    "mangrove_pct",
]


# =========================================================
# Stage-B features
#
# Stage B is only applied after Stage A predicts
# INDUSTRIAL.
#
# Persistence is allowed because all verified industrial
# samples have independently audited historical baselines.
#
# historical_average_frp / frp_ratio are deliberately
# excluded from this first prototype model because their
# missingness is high.
# =========================================================


STAGE_B_FEATURES = (

    STAGE_A_FEATURES
    +
    [
        "detections_30d",
        "active_days_30d",
        "persistence_score",
    ]
)


STAGE_A_CLASSES = [
    "AGRICULTURAL_OPEN",
    "FOREST",
    "INDUSTRIAL",
]


STAGE_B_CLASSES = [
    "INDUSTRIAL_FIRE",
    "PERSISTENT_INDUSTRIAL_SOURCE",
]


# =========================================================
# Utility
# =========================================================


def validate_columns(
    dataframe,
    columns,
):

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]


    if missing:

        raise RuntimeError(
            "Required columns are missing:\n"
            +
            "\n".join(
                missing
            )
        )


# =========================================================
# Convert feature columns to numeric
# =========================================================


def prepare_numeric_features(
    dataframe,
    feature_names,
):

    result = dataframe[
        feature_names
    ].copy()


    for column in feature_names:

        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )


    return result


# =========================================================
# ML pipeline
#
# Missing distance_to_industry_m has a specific meaning:
# no industrial OSM feature was found within 5 km.
#
# Therefore use 6000 m rather than median imputation.
#
# Other numeric missing values use training-fold median.
# =========================================================


def build_pipeline(
    feature_names,
    stage,
):

    distance_column = [
        "distance_to_industry_m"
    ]


    other_columns = [
        feature
        for feature in feature_names
        if feature
        != "distance_to_industry_m"
    ]


    preprocessing = ColumnTransformer(

        transformers=[

            (
                "industry_distance",

                SimpleImputer(
                    strategy="constant",
                    fill_value=6000.0,
                ),

                distance_column,
            ),

            (
                "other",

                SimpleImputer(
                    strategy="median",
                ),

                other_columns,
            ),
        ],

        remainder="drop",

        verbose_feature_names_out=False,
    )


    if stage == "A":

        classifier = RandomForestClassifier(

            n_estimators=500,

            max_depth=6,

            min_samples_leaf=2,

            max_features="sqrt",

            class_weight=
                "balanced_subsample",

            random_state=
                RANDOM_STATE,

            n_jobs=-1,
        )


    else:

        classifier = RandomForestClassifier(

            n_estimators=500,

            max_depth=4,

            min_samples_leaf=1,

            max_features="sqrt",

            class_weight=
                "balanced",

            random_state=
                RANDOM_STATE,

            n_jobs=-1,
        )


    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessing,
            ),

            (
                "classifier",
                classifier,
            ),
        ]
    )


# =========================================================
# Group-aware cross-validation
# =========================================================


def group_cross_validate(
    dataframe,
    feature_names,
    target_column,
    classes,
    stage,
):

    X = prepare_numeric_features(
        dataframe,
        feature_names,
    )


    y = (
        dataframe[
            target_column
        ]
        .astype(
            str
        )
        .reset_index(
            drop=True
        )
    )


    groups = (
        dataframe[
            "group_id"
        ]
        .astype(
            str
        )
        .reset_index(
            drop=True
        )
    )


    X = X.reset_index(
        drop=True
    )


    cv = StratifiedGroupKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE,
    )


    pipeline = build_pipeline(
        feature_names,
        stage,
    )


    predictions = np.empty(
        len(
            dataframe
        ),
        dtype=object,
    )


    fold_numbers = np.zeros(
        len(
            dataframe
        ),
        dtype=int,
    )


    print()

    print(
        f"Stage {stage} "
        f"group-aware folds:"
    )


    for fold, (
        train_index,
        test_index,
    ) in enumerate(

        cv.split(
            X,
            y,
            groups,
        ),

        start=1,
    ):

        train_groups = set(
            groups.iloc[
                train_index
            ]
        )


        test_groups = set(
            groups.iloc[
                test_index
            ]
        )


        overlap = (
            train_groups
            &
            test_groups
        )


        if overlap:

            raise RuntimeError(
                f"Group leakage detected "
                f"in Stage {stage}, "
                f"fold {fold}."
            )


        y_train = y.iloc[
            train_index
        ]


        y_test = y.iloc[
            test_index
        ]


        print()

        print(
            f"  Fold {fold}:"
        )


        print(
            f"    Train rows: "
            f"{len(train_index)}"
        )


        print(
            f"    Test rows: "
            f"{len(test_index)}"
        )


        print(
            f"    Train groups: "
            f"{len(train_groups)}"
        )


        print(
            f"    Test groups: "
            f"{len(test_groups)}"
        )


        print(
            "    Test class counts:"
        )


        print(
            y_test
            .value_counts()
            .to_string()
        )


        fold_model = clone(
            pipeline
        )


        fold_model.fit(
            X.iloc[
                train_index
            ],
            y_train,
        )


        fold_predictions = (
            fold_model.predict(
                X.iloc[
                    test_index
                ]
            )
        )


        predictions[
            test_index
        ] = fold_predictions


        fold_numbers[
            test_index
        ] = fold


    # =====================================================
    # Evaluation
    # =====================================================


    accuracy = accuracy_score(
        y,
        predictions,
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions,
        )
    )


    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )


    report = classification_report(
        y,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )


    matrix = confusion_matrix(
        y,
        predictions,
        labels=classes,
    )


    metrics = {

        "rows":
            int(
                len(
                    dataframe
                )
            ),

        "unique_groups":
            int(
                groups.nunique()
            ),

        "folds":
            N_SPLITS,

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "classes":
            classes,

        "classification_report":
            report,

        "confusion_matrix":
            matrix.tolist(),
    }


    predictions_dataframe = dataframe[
        [
            "example_id",
            "dataset_source",
            "stage_a_label",
            "stage_b_label",
            "group_id",
        ]
    ].copy()


    predictions_dataframe[
        "fold"
    ] = fold_numbers


    predictions_dataframe[
        "actual"
    ] = y.values


    predictions_dataframe[
        "predicted"
    ] = predictions


    predictions_dataframe[
        "correct"
    ] = (
        predictions_dataframe[
            "actual"
        ]
        ==
        predictions_dataframe[
            "predicted"
        ]
    ).astype(
        int
    )


    # =====================================================
    # Train final model on all verified data
    # =====================================================


    final_model = clone(
        pipeline
    )


    final_model.fit(
        X,
        y,
    )


    return (
        metrics,
        predictions_dataframe,
        final_model,
    )


# =========================================================
# Feature importance
# =========================================================


def save_feature_importance(
    fitted_pipeline,
    path,
):

    preprocessing = (
        fitted_pipeline[
            "preprocessor"
        ]
    )


    classifier = (
        fitted_pipeline[
            "classifier"
        ]
    )


    transformed_names = (
        preprocessing
        .get_feature_names_out()
    )


    importances = (
        classifier
        .feature_importances_
    )


    dataframe = pd.DataFrame(
        {
            "feature":
                transformed_names,

            "importance":
                importances,
        }
    )


    dataframe = dataframe.sort_values(
        by="importance",
        ascending=False,
    )


    dataframe.to_csv(
        path,
        index=False,
    )


    return dataframe


# =========================================================
# Print metrics
# =========================================================


def print_metrics(
    title,
    metrics,
):

    print()

    print("=" * 78)

    print(
        title
    )

    print("=" * 78)


    print()

    print(
        f"Rows: "
        f"{metrics['rows']}"
    )


    print(
        f"Unique groups: "
        f"{metrics['unique_groups']}"
    )


    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )


    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )


    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )


    print()

    print(
        "Per-class metrics:"
    )


    for class_name in (
        metrics[
            "classes"
        ]
    ):

        values = metrics[
            "classification_report"
        ][
            class_name
        ]


        print(
            f"  {class_name}"
        )


        print(
            f"    Precision: "
            f"{values['precision']:.4f}"
        )


        print(
            f"    Recall:    "
            f"{values['recall']:.4f}"
        )


        print(
            f"    F1:        "
            f"{values['f1-score']:.4f}"
        )


        print(
            f"    Support:   "
            f"{int(values['support'])}"
        )


    print()

    print(
        "Confusion matrix:"
    )


    matrix = np.array(
        metrics[
            "confusion_matrix"
        ]
    )


    matrix_dataframe = pd.DataFrame(
        matrix,

        index=[
            f"actual_{label}"
            for label
            in metrics[
                "classes"
            ]
        ],

        columns=[
            f"pred_{label}"
            for label
            in metrics[
                "classes"
            ]
        ],
    )


    print(
        matrix_dataframe.to_string()
    )


# =========================================================
# Main
# =========================================================


def main():

    print("=" * 78)

    print(
        "THEEFINDER - TRAIN VERIFIED ML MODELS"
    )

    print("=" * 78)


    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Verified ML feature dataset "
            "not found:\n"
            f"{INPUT_FILE}"
        )


    dataframe = pd.read_csv(
        INPUT_FILE
    )


    print()

    print(
        f"Feature rows loaded: "
        f"{len(dataframe)}"
    )


    if len(
        dataframe
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows "
            f"but found {len(dataframe)}."
        )


    required_columns = (

        [
            "example_id",
            "dataset_source",
            "stage_a_label",
            "stage_b_label",
            "group_id",
        ]

        +
        list(
            set(
                STAGE_A_FEATURES
                +
                STAGE_B_FEATURES
            )
        )
    )


    validate_columns(
        dataframe,
        required_columns,
    )


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =====================================================
    # Quality gate
    # =====================================================


    print()

    print("=" * 78)

    print(
        "MODEL QUALITY GATE"
    )

    print("=" * 78)


    print()

    print(
        "Stage-A features:"
    )


    for feature in STAGE_A_FEATURES:

        print(
            f"  {feature}"
        )


    print()

    print(
        "Stage-B features:"
    )


    for feature in STAGE_B_FEATURES:

        print(
            f"  {feature}"
        )


    # Ensure no historical leakage accidentally enters A.
    forbidden_stage_a = [

        feature
        for feature in STAGE_A_FEATURES

        if (
            feature.startswith(
                "historical_"
            )
            or
            feature
            in {
                "detections_30d",
                "active_days_30d",
                "persistence_score",
                "frp_ratio",
            }
        )
    ]


    if forbidden_stage_a:

        raise RuntimeError(
            "Persistence/history leakage "
            "found in Stage-A features:\n"
            +
            "\n".join(
                forbidden_stage_a
            )
        )


    print()

    print(
        "Stage-A persistence leakage check: PASS"
    )


    # =====================================================
    # Stage A
    # =====================================================


    stage_a_dataframe = (
        dataframe.copy()
    )


    print()

    print(
        "Stage-A class counts:"
    )


    print(
        stage_a_dataframe[
            "stage_a_label"
        ]
        .value_counts()
        .to_string()
    )


    (
        stage_a_metrics,
        stage_a_predictions,
        stage_a_model,
    ) = group_cross_validate(

        dataframe=
            stage_a_dataframe,

        feature_names=
            STAGE_A_FEATURES,

        target_column=
            "stage_a_label",

        classes=
            STAGE_A_CLASSES,

        stage=
            "A",
    )


    stage_a_predictions.to_csv(
        STAGE_A_PREDICTIONS_FILE,
        index=False,
    )


    joblib.dump(
        stage_a_model,
        STAGE_A_MODEL_FILE,
    )


    stage_a_importance = (
        save_feature_importance(
            stage_a_model,
            STAGE_A_IMPORTANCE_FILE,
        )
    )


    # =====================================================
    # Stage B
    # =====================================================


    stage_b_dataframe = dataframe[
        dataframe[
            "stage_a_label"
        ]
        ==
        "INDUSTRIAL"
    ].copy()


    print()

    print(
        "Stage-B class counts:"
    )


    print(
        stage_b_dataframe[
            "stage_b_label"
        ]
        .value_counts()
        .to_string()
    )


    if len(
        stage_b_dataframe
    ) != 27:

        raise RuntimeError(
            "Expected 27 industrial examples "
            "for Stage B."
        )


    fire_count = int(
        (
            stage_b_dataframe[
                "stage_b_label"
            ]
            ==
            "INDUSTRIAL_FIRE"
        ).sum()
    )


    if fire_count < N_SPLITS:

        raise RuntimeError(
            "Not enough INDUSTRIAL_FIRE examples "
            "for five-fold Stage-B evaluation."
        )


    (
        stage_b_metrics,
        stage_b_predictions,
        stage_b_model,
    ) = group_cross_validate(

        dataframe=
            stage_b_dataframe,

        feature_names=
            STAGE_B_FEATURES,

        target_column=
            "stage_b_label",

        classes=
            STAGE_B_CLASSES,

        stage=
            "B",
    )


    stage_b_predictions.to_csv(
        STAGE_B_PREDICTIONS_FILE,
        index=False,
    )


    joblib.dump(
        stage_b_model,
        STAGE_B_MODEL_FILE,
    )


    stage_b_importance = (
        save_feature_importance(
            stage_b_model,
            STAGE_B_IMPORTANCE_FILE,
        )
    )


    # =====================================================
    # Print results
    # =====================================================


    print_metrics(
        "STAGE-A 5-FOLD GROUP-AWARE RESULTS",
        stage_a_metrics,
    )


    print_metrics(
        "STAGE-B 5-FOLD GROUP-AWARE RESULTS",
        stage_b_metrics,
    )


    print()

    print("=" * 78)

    print(
        "TOP STAGE-A FEATURES"
    )

    print("=" * 78)


    print(
        stage_a_importance
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


    print()

    print("=" * 78)

    print(
        "TOP STAGE-B FEATURES"
    )

    print("=" * 78)


    print(
        stage_b_importance
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


    # =====================================================
    # Save metrics
    # =====================================================


    metrics = {

        "dataset_rows":
            int(
                len(
                    dataframe
                )
            ),

        "evaluation":
            (
                "5-fold "
                "StratifiedGroupKFold"
            ),

        "random_state":
            RANDOM_STATE,

        "stage_a": {
            **stage_a_metrics,

            "features":
                STAGE_A_FEATURES,
        },

        "stage_b": {
            **stage_b_metrics,

            "features":
                STAGE_B_FEATURES,

            "warning":
                (
                    "Prototype evaluation only. "
                    "INDUSTRIAL_FIRE has only "
                    "5 independently verified events."
                ),
        },
    }


    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


    # =====================================================
    # Final validation
    # =====================================================


    print()

    print("=" * 78)

    print(
        "MODEL TRAINING COMPLETE"
    )

    print("=" * 78)


    print()

    print(
        "Stage-A model:"
    )

    print(
        STAGE_A_MODEL_FILE
    )


    print()

    print(
        "Stage-B model:"
    )

    print(
        STAGE_B_MODEL_FILE
    )


    print()

    print(
        "Evaluation metrics:"
    )

    print(
        METRICS_FILE
    )


    print()

    print(
        "Stage-A predictions:"
    )

    print(
        STAGE_A_PREDICTIONS_FILE
    )


    print()

    print(
        "Stage-B predictions:"
    )

    print(
        STAGE_B_PREDICTIONS_FILE
    )


    print()

    print(
        "RESULT: VERIFIED_MODELS_TRAINED"
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Stage-B metrics must be presented "
        "as prototype results because only "
        "five independently verified "
        "industrial-fire events are available."
    )


if __name__ == "__main__":

    main()