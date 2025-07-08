from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS


def get_task_suite_configs():
    TASK_SUITE_CONFIGS.update(
        {
            "ruler_4k": {
                "tasks": [
                    f"ruler_4k_{task_type}"
                    for task_type in [
                        "cwe",
                        "fwe",
                        "niah_multikey_1",
                        "niah_multikey_2",
                        "niah_multikey_3",
                        "niah_multiquery",
                        "niah_multivalue",
                        "niah_single_1",
                        "niah_single_2",
                        "niah_single_3",
                        "qa_1",
                        "qa_2",
                        "vt",
                    ]
                ],
                "primary_metric": "macro",
            },
            "sciriff_bioqa": {
                "tasks": [
                    "sciriff_bioasq_factoid_qa",  # Abstractive
                    "sciriff_bioasq_general_qa",  # Abstractive
                    "sciriff_bioasq_yesno_qa",  # Y/N
                    "sciriff_covid_deepset_qa",  # Extractive
                    "sciriff_pubmedqa_qa",  # Y/N
                ],
                "primary_metric": "macro",
            },
        },
    )
    return TASK_SUITE_CONFIGS
