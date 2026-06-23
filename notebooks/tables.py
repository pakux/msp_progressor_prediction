import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
<<<<<<< HEAD
def _():
    return


@app.cell
def setup_1():
    import numpy as np
=======
def read_data():
>>>>>>> 6862864bf16eddba65e9db7336471394fbe42978
    import marimo as mo
    import pandas as pd
    from os.path import join, abspath, isfile
    from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact

    tests = ['pst', 'mdt', 'wst', 'cst']
    models = ['sfcn', 'ssl-finetuned', 'lora', 'dense', 'swin']

    batchsize = 16
    imagesize = 96

    eval_dir = '/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/evaluations/'

    baseline_characteristics_df = pd.read_csv("baseline_characteristics.csv")
    baseline_characteristics_df.mstype

    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('rr', 'Relapsing Remitting MS')
    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('sp', 'Secondary Progressive MS')
    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('pp', 'Primary Progressive MS')
    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('Secondary Progressing MS', 'Secondary Progressive MS')

    # dataset_df = pd.read_csv("eid_distribution.csv") # TODO: delete the file when this works
    dataset_df = pd.read_csv('data/dataset.csv')
    dataset_df = dataset_df.drop(columns=['site', 'sex'])
    baseline_characteristics_df = baseline_characteristics_df.merge(
        right=dataset_df, left_on="mpi", right_on="eid"
    )

    baseline_characteristics_df = baseline_characteristics_df.query('dataset.notnull()')

    def progression_col(testname):
        return f"worst_progressor_2ycutoff_{testname}_2z"
    return (
        baseline_characteristics_df,
        batchsize,
        dataset_df,
        eval_dir,
        fisher_exact,
        imagesize,
        isfile,
        join,
        mannwhitneyu,
        mo,
        models,
        np,
        pd,
        progression_col,
        tests,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read eid distribution from  `training`, `validation` and `external test` datasets.
    We need to keep in mind, that some ID's are not available for all tests.

    Maybe I should

    |                | Total | PST | MDT | WST | CST |
    | -------------- | ---   | --- | --- | --- | --- |
    | Task Completed |  nt   | nt  | nt  | nt. | nt  |
    | Age  mean (SD) |
    | sex female (%) |
    """)
    return


@app.cell
def characteristics(
    baseline_characteristics_df,
    fisher_exact,
    mannwhitneyu,
    np,
    pd,
    progression_col,
    tests,
):
    demographics = {}
    demographics["dataset"] = ["all n", "all value"]

    _sex_ds = len(baseline_characteristics_df.query('sex.notnull()'))
    _sex_f = len(baseline_characteristics_df.query('sex == "female"'))
    demographics["sex"] = [_sex_ds, f"{_sex_f} ({_sex_f / _sex_ds:0.2%})"]



    demographics["age"] = [
        len(baseline_characteristics_df.query("age.notnull()")),
    f"{baseline_characteristics_df.age.mean():0.1f} ± {   baseline_characteristics_df.age.std():0.2f}"
    ]
    demographics["mstype"] = [
        len(baseline_characteristics_df.query("mstype.notnull()")), ""
    ]


    demographics["pdds"] = [
        len(baseline_characteristics_df.query("pdds_scr.notnull()")),
    f"{baseline_characteristics_df.pdds_scr.mean():0.1f} ± {baseline_characteristics_df.pdds_scr.std():0.2f}"
    ]

    demographics["mstype_cis"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query(
                'mstype == "Clinically Isolated Syndrome"'
            )
        ),
    ]
    demographics["mstype_prms"] = [
       pd.NA,
        len(
            baseline_characteristics_df.query('mstype == "Progressive Relapsing MS"')
        ),  
    ] 
    demographics["mstype_rrms"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query('mstype == "Relapsing Remitting MS"')
        ),
    ]
    demographics["mstype_spms"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query(
                'mstype == "Secondary Progressive MS"'
            )
        ),
    ]
    demographics["mstype_ppms"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query('mstype == "Primary Progressive MS"')
        ),
    ]

    for _t in tests:
        demographics[f"progressors_{_t}"] = [pd.NA, pd.NA]


    for ds in["training", "validation", "external test"]:

        demographics["dataset"].append(f"{ds} n")
        demographics["dataset"].append(f"{ds} value")
        if ds in ["validation", "external test"]:
            demographics["dataset"].append(f"{ds} p") 

        _sex_ds = len(baseline_characteristics_df.query('dataset == @ds and sex.notnull()'))
        _sex_f = len(baseline_characteristics_df.query('dataset == @ds and sex == "female"'))
        demographics["sex"].append(_sex_ds)
        demographics["sex"].append( f"{_sex_f} ({_sex_f / _sex_ds:0.2%})")

        # calculade p-value for test against training
        if ds in ["validation", "external test"]:
        
            _sex_training = np.array([
                [
                    len(baseline_characteristics_df.query('sex == "female" and dataset=="training"')), 
                    len(baseline_characteristics_df.query('sex == "male" and dataset=="training"'))
                                     ],
                [
                    len(baseline_characteristics_df.query('sex == "female" and dataset==@ds')), 
                    len(baseline_characteristics_df.query('sex == "male" and dataset==@ds')) 
                ]
            ])
            _odds_ratio, _p_value = fisher_exact(_sex_training)
            demographics["sex"].append(f"{_p_value:0.3f}") 

    
        demographics["age"].append( len(
                baseline_characteristics_df.query(
                    "dataset == @ds and age.notnull()"
                )
            )
        )
        demographics["age"].append(
            f'{baseline_characteristics_df.query("dataset== @ds").age.mean():0.1f} ± {   baseline_characteristics_df.query("dataset== @ds").age.std():0.2f}'

        )

        if ds in ["validation", "external test"]:
            _, _p = mannwhitneyu(
                baseline_characteristics_df.query('dataset=="training"')['age'].dropna().to_list(),
                baseline_characteristics_df.query('dataset==@ds')['age'].dropna().to_list()
            )
            print(_p)
            demographics["age"].append(f"{_p:0.3f}")


    
        demographics["mstype"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and mstype.notnull()"
                )
            )
        )
    
        demographics["mstype"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            demographics["mstype"].append(pd.NA)

        demographics["pdds"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and pdds_scr.notnull()"
                )
            )
        )
        demographics["pdds"].append(
    f'{baseline_characteristics_df.query("dataset == @ds").pdds_scr.mean():0.1f} ± {baseline_characteristics_df.query("dataset == @ds").pdds_scr.std():0.2f}' )

        if ds in ["validation", "external test"]:
            _, _p = mannwhitneyu(
                baseline_characteristics_df.query('dataset=="training"')['pdds_scr'].dropna().to_list(),
                baseline_characteristics_df.query('dataset==@ds')['pdds_scr'].dropna().to_list()
            )
            print(_p)
        
            demographics["pdds"].append(f"{_p:0.3f}")


        demographics["mstype_cis"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Clinically Isolated Syndrome"'
                )
            )
        )
        demographics["mstype_cis"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _cis_training = np.array([
                    [
                        len(baseline_characteristics_df.query('mstype !=  "Clinically Isolated Syndrome" and dataset=="training"')), 
                        len(baseline_characteristics_df.query('mstype ==  "Clinically Isolated Syndrome" and dataset=="training"'))
                                         ],
                    [
                        len(baseline_characteristics_df.query('mstype == "Clinically Isolated Syndrome" and dataset==@ds')), 
                        len(baseline_characteristics_df.query('mstype == "Clinically Isolated Syndrome" and dataset==@ds')) 
                    ]
                ])
            _odds_ratio, _p_value = fisher_exact(_cis_training)
            demographics["mstype_cis"].append(f"{_p_value:0.3f}") 


        demographics["mstype_prms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset== @ds and mstype == "Progressive Relapsing MS"'
                )
            )
        )
        demographics["mstype_prms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _cis_training = np.array([
                    [
                        len(baseline_characteristics_df.query('mstype !=  "Progressive Relapsing MS" and dataset=="training"')), 
                        len(baseline_characteristics_df.query('mstype ==  "Progressive Relapsing MS" and dataset=="training"'))
                                         ],
                    [
                        len(baseline_characteristics_df.query('mstype == "Progressive Relapsing MS" and dataset==@ds')), 
                        len(baseline_characteristics_df.query('mstype == "Progressive Relapsing MS" and dataset==@ds')) 
                    ]
                ])
            _odds_ratio, _p_value = fisher_exact(_cis_training)
            demographics["mstype_prms"].append(f"{_p_value:0.3f}") 


        
            # demographics["mstype_"].append(pd.NA)
    
        demographics["mstype_rrms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Relapsing Remitting MS"'
                )
            )
        )    
        demographics["mstype_rrms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _cis_training = np.array([
                    [
                        len(baseline_characteristics_df.query('mstype !=  "Relapsing Remitting MS" and dataset=="training"')), 
                        len(baseline_characteristics_df.query('mstype ==  "Relapsing Remitting MS" and dataset=="training"'))
                                         ],
                    [
                        len(baseline_characteristics_df.query('mstype == "Relapsing Remitting MS" and dataset==@ds')), 
                        len(baseline_characteristics_df.query('mstype == "Relapsing Remitting MS" and dataset==@ds')) 
                    ]
                ])
            _odds_ratio, _p_value = fisher_exact(_cis_training)
            demographics["mstype_rrms"].append(f"{_p_value:0.3f}") 

        demographics["mstype_spms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Secondary Progressive MS"'
                )
            )
        )
        demographics["mstype_spms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _spms_training = np.array([
                    [
                        len(baseline_characteristics_df.query('mstype !=  "Secondary Progressive MS" and dataset=="training"')), 
                        len(baseline_characteristics_df.query('mstype ==  "Secondary Progressive MS" and dataset=="training"'))
                                         ],
                    [
                        len(baseline_characteristics_df.query('mstype == "Secondary Progressive MS" and dataset==@ds')), 
                        len(baseline_characteristics_df.query('mstype == "Secondary Progressive MS" and dataset==@ds')) 
                    ]
                ])
            _odds_ratio, _p_value = fisher_exact(_spms_training)
            demographics["mstype_spms"].append(f"{_p_value:0.3f}") 

        demographics["mstype_ppms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Primary Progressive MS"'
                )
            )
        )
        demographics["mstype_ppms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            _ppms_training = np.array([
                    [
                        len(baseline_characteristics_df.query('mstype !=  "Primary Progressive MS" and dataset=="training"')), 
                        len(baseline_characteristics_df.query('mstype ==  "Primary Progressive MS" and dataset=="training"'))
                                         ],
                    [
                        len(baseline_characteristics_df.query('mstype == "Primary Progressive MS" and dataset==@ds')), 
                        len(baseline_characteristics_df.query('mstype == "Primary Progressive MS" and dataset==@ds')) 
                    ]
                ])
            _odds_ratio, _p_value = fisher_exact(_ppms_training)
            demographics["mstype_ppms"].append(f"{_p_value:0.3f}") 

    
        for _t in tests:
            _n = len(baseline_characteristics_df.query(f"dataset == @ds and ~{progression_col(_t)}.isna()"))
            _n_training = len(baseline_characteristics_df.query(f'dataset == "training" and ~{progression_col(_t)}.isna()'))
        
            _n_progressors = len(baseline_characteristics_df.query(f"dataset == @ds and {progression_col(_t)} == 1"))
        
            demographics[f"progressors_{_t}"].append(_n)
            demographics[f"progressors_{_t}"].append(
                f'{_n_progressors:0.0f} ({_n_progressors/ _n:0.2%})'
                )
            if ds in ["validation", "external test"]:
                # TODO: add here mann-whitney-u
               # _progressor_training = np.array([
               #      [
               #          _n, 
               #         _n_training
               #                           ],
               #      [
               #          _n, 
               #          _n_progressors
               #      ]
               # ])
               # print(_progressor_training)
               # _odds_ratio, _p_value = fisher_exact(_progressor_training)
               # demographics[f"progressors_{_t}"].append(f"{_p_value:0.4f}")
                demographics[f"progressors_{_t}"].append(pd.NA)
    # pd.DataFrame(demographics)
    demographics
    pd.DataFrame(demographics).transpose()
    return (demographics,)


@app.cell
def _(demographics):
    demographics
    return


@app.cell
def _(baseline_characteristics_df, demographics, fisher_exact, np):
    _sex_training = np.array([
        [
            len(baseline_characteristics_df.query('sex == "female" and dataset=="training"')), 
            len(baseline_characteristics_df.query('sex == "male" and dataset=="training"'))
                             ],
        [
            len(baseline_characteristics_df.query('sex == "female" and dataset=="validation"')), 
            len(baseline_characteristics_df.query('sex == "male" and dataset=="validation"')) 
        ]
    ])

    _odds_ratio, _p_value = fisher_exact(_sex_training)

    demographics["validation p"] = []


    _sex_training = np.array([
        [
            len(baseline_characteristics_df.query('sex == "female" and dataset=="training"')), 
            len(baseline_characteristics_df.query('sex == "male" and dataset=="training"'))
                             ],
        [
            len(baseline_characteristics_df.query('sex == "female" and dataset=="external test"')), 
            len(baseline_characteristics_df.query('sex == "male" and dataset=="external test"')) 
        ]
    ])

    print(_sex_training)
    _odds_ratio, _p_value = fisher_exact(_sex_training.transpose())

    # chi2_contingency(_sex_training)

    _p_value
     # sex_training
    return


@app.cell
def _(demographics):
    demographics
    return


@app.cell
def _(demographics, pd):
    pd.DataFrame(demographics)
    return


app._unparsable_cell(
    """
    \"\"\"
    The user wants me to add p-values to the demographics dictionary that was created in the previous cell. I need to calculate p-values for:
    1. Comparison between training and validation datasets
    2. Comparison between training and external test datasets

    For each demographic variable:
    - sex: Would use chi-squared test (categorical)
    - age: Would use t-test or Mann-Whitney U test (continuous)
    - mstype: Would use chi-squared test (categorical)
    - pdds: Would use t-test or Mann-Whitney U test (continuous)
    - Each mstype subtype: Chi-squared or Fisher's exact test
    - progressors for each test: Chi-squared or Fisher's exact test

    I need to use scipy.stats for these tests. Let me create a new cell that:
    1. Calculates p-values for each variable comparing:
       - training vs validation
       - training vs external test
    2. Adds these p-values to a new DataFrame

    Let me write the code:
    \"\"\"

    from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, fisher_exact
    import numpy as np

    _pvalues = {}
    _pvalues[\"dataset\"] = [\"training vs validation\", \"training vs external test\"]

    def add_pvalue(var_name, test_type):
        _pvalues[var_name] = []

        for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
            ds1, ds2 = comparison

            if test_type == \"chi2\":
                # For categorical variables (sex, mstype, progressors)
                # Create contingency table
                # This will depend on the specific variable
                pass
            elif test_type == \"mannwhitney\":
                # For continuous variables (age, pdds)
                pass
    \"\"\"

    I'll continue developing the code for calculating p-values, focusing on the statistical test implementations for different variable types. I'll carefully handle different scenarios for categorical and continuous variables, ensuring robust statistical comparisons between datasets.

    For the gender comparison, I'm constructing a contingency table that captures the distribution of male and female subjects across different datasets. Using Fisher's exact test, I'll compare the training dataset against two external datasets, systematically capturing p-values to assess potential statistically significant differences.

    The process involves creating a 2x2 contingency table for each comparison, tracking male and female subject counts across the specified datasets. I'll store these p-values to evaluate potential variations in demographic composition.

    For age comparisons, I'll employ a Mann-Whitney U test to examine potential statistical differences in age distributions between the training and external datasets, filtering out null values and comparing median ages.

    The next step involves creating a contingency table to assess multiple sclerosis (MS) type distributions, focusing on clinically significant subtypes across different dataset comparisons.

    By flattening the contingency table and applying a chi-squared test, I'll statistically evaluate whether the observed subgroup variations are significantly different from expected random distributions. I'll track p-values for each dataset comparison, carefully handling potential sample size or statistical distribution challenges.

    The subsequent analysis will examine Parkinson's disease duration, similarly using Mann-Whitney U tests to compare disease progression across different dataset configurations. I'll appreciate the nuanced approach to comparing MS subtypes across datasets. The Fisher's Exact test provides a robust method for analyzing categorical data with small sample sizes, allowing precise statistical comparisons between groups. By systematically comparing Clinically Isolated Syndrome (CIS) counts, the analysis can reveal meaningful epidemiological patterns or potential diagnostic variations across different research datasets.

    The statistical approach suggests a careful, methodical investigation of MS subtype distributions, which is crucial for understanding disease progression and demographic characteristics in neurological research.

    The contingency tables are being constructed to capture categorical relationships, enabling precise statistical hypothesis testing through Fisher's Exact test. This method allows rigorous comparison of subtype distributions between different study populations.

    The statistical analysis systematically examines differences in MS subtypes across datasets, focusing on observed counts versus expected distributions. By applying chi-squared tests, the researcher can determine whether observed variations are statistically significant or potentially due to random chance.

    The code prepares contingency tables for different MS subtypes, which will facilitate comprehensive statistical comparisons across groups. These tables capture the raw counts necessary for subsequent statistical testing, enabling precise evaluation of subtype prevalence and potential differences.

    I'll examine the subtype distribution, noting the complexity of creating meaningful contingency tables given the overlapping and mutually exclusive nature of MS subtypes. The challenge lies in constructing a statistically valid representation that accurately reflects the dataset's characteristics.

    The current approach requires careful reconsideration to ensure statistical rigor and accurate representation of the MS subtype variations across different datasets. I compare the proportions of MS subtypes across datasets using a chi-squared test. By creating contingency tables for different MS types like RRMS, SPMS, PPMS, and CIS, I can analyze statistical differences in subtype frequencies between training and external datasets. The test evaluates whether the observed distributions significantly deviate from expected frequencies.

    I systematically construct 2x2 contingency tables, extracting subtype counts for each dataset and applying the chi-squared test to assess potential statistical significance in subtype prevalence.

    I'll track the p-values for each MS subtype comparison, storing them in a dictionary for potential further statistical analysis or visualization.

    For neurocognitive tests, I'll prepare to examine progression rates using similar statistical comparison methods, setting up a framework to investigate potential performance differences across different dataset configurations.

    I'll compare neurocognitive test outcomes between training and alternative datasets, applying Fisher's Exact test to evaluate potential statistical significance in progression rates.

    I'll systematically build a panda DataFrame from the collected statistical analysis results, enabling structured visualization of comparative p-values across different tests and dataset comparisons.

    The approach allows precise examination of neurological progression across multiple cognitive test types, utilizing robust statistical testing methods to identify meaningful differences.

    Key observations emerge regarding sample sizes and prevalence estimates, suggesting careful interpretation is needed given the limited context and potential methodological nuances in the original statistical approach.

    The implementation focuses on creating a systematic analysis framework for p-values across different dataset comparisons, with flexibility to handle various statistical tests and demographic variables. The goal appears to be generating a comprehensive overview of statistical relationships between training and validation/testing datasets.

    I'll need to carefully map out the data structure, ensuring each demographic variable can be compared across different dataset configurations. The key will be creating a robust comparison mechanism that can handle potential missing values and different test types.

    The initialization of the p-values dataset suggests a methodical approach to tracking statistical significance across multiple dimensions, with the added complexity of managing potential data sparsity.

    I'll start constructing the comparison loop, focusing on extracting training data and preparing for subsequent dataset comparisons. The goal is to systematically analyze each demographic or clinical feature across different dataset groups.

    The current strategy involves iteratively processing each variable, preparing contingency tables and applying appropriate statistical tests to evaluate group differences.

    I notice a critical error in the dataset filtering approach. Instead of using invalid column references, I need to directly filter the baseline characteristics DataFrame using the correct column name (`dataset`). This means modifying the filtering logic to extract data specifically for the training dataset and other comparison groups.

    The corrected approach will involve using dataset-specific filtering methods, potentially using the unique dataset values I've previously identified: \"training\", \"validation\", and \"external test\". I'll carefully construct the filtering logic to ensure accurate data extraction and comparison. The data and code reveal a systematic approach to analyzing demographic and clinical characteristics across different datasets. The method involves extracting and computing key metrics like sex distribution, age statistics, and multiple sclerosis subtypes for training, validation, and external test cohorts. The use of `pd.NA` suggests intentional handling of missing or undefined values during the data processing workflow.

    The code demonstrates a structured exploration of patient demographics, with careful tracking of dataset-specific variations in clinical parameters. By iterating through different MS subtypes and calculating counts, the analysis provides a comprehensive overview of the population's clinical characteristics.

    The approach suggests a systematic comparison method across datasets, potentially serving as a foundational step for subsequent statistical analysis and modeling in neurological research.</think>```python

    \"\"\"

    _pvalues = {}
    _pvalues[\"dataset\"] = [\"training vs validation\", \"training vs external test\"]

    # Helper function for safe Mann-Whitney U test
    def safe_mannwhitneyu(x, y):
        # Remove NaN values
        x_clean = x.dropna()
        y_clean = y.dropna()
        if len(x_clean) > 0 and len(y_clean) > 0:
            try:
               stat, p = mannwhitneyu(x_clean, y_clean)
               return p
            except:
                return pd.NA
        return pd.NA

    # Helper function for safe chi-squared test
    def safe_chi2_contingency(table):
        try:
            chi2, p, dof, expected = chi2_contingency(table)
            return p
        except:
            return pd.NA

    # Helper function for safe Fisher's exact test
    def safe_fisher_exact(table):
        try:
            odds_ratio, p = fisher_exact(table)
            return p
        except:
            return pd.NA

    # Sex (chi-squared or fisher's exact)
    _pvalues[\"sex\"] = []
    for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
        ds1, ds2 = comparison
        # Create contingency table: [female1, male1] vs [female2, male2]
        female1 = len(baseline_characteristics_df.query(f'dataset == \"{ds1}\" and sex == \"female\"'))
        male1 = len(baseline_characteristics_df.query(f'dataset == \"{ds1}\" and sex == \"male\"'))
            
    return pd.NA

    # Helper function for safe chi-squared test
    def safe_chi2_contingency(table):
        try:
            chi2, p, dof, expected = chi2_contingency(table)
            return p
        except:
            return pd.NA

    # Helper function for safe Fisher's exact test
    def safe_fisher_exact(table):
        try:
            odds_ratio, p = fisher_exact(table)
            return p
        except:
            return pd.NA

        female2 = len(baseline_characteristics_df.query(f'dataset == \"{ds2}\" and sex == \"female\"'))
        male2 = len(baseline_characteristics_df.query(f'dataset == \"{ds2}\" and sex == \"male\"'))
        table = [[female1, male1], [female2, male2]]
        _pvalues[\"sex\"].append(safe_fisher_exact(table))

    # Age (Mann-Whitney U test)
    _pvalues[\"age\"] = []
    for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
        ds1, ds2 = comparison
        age1 = baseline_characteristics_df.query(f'dataset == \"{ds1}\"').age
        age2 = baseline_characteristics_df.query(f'dataset == \"{ds2}\"').age
        _pvalues[\"age\"].append(safe_mannwhitneyu(age1, age2))

    # MS type (chi-squared)
    _pvalues[\"mstype\"] = []
    for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
        ds1, ds2 = comparison
        # Create contingency table for all MS types
        types1 = baseline_characteristics_df.query(f'dataset == \"{ds1}\" and mstype.notnull()').mstype.value_counts()
        types2 = baseline_characteristics_df.query(f'dataset == \"{ds2}\" and mstype.notnull()').mstype.value_counts()

        # Get all unique types
        all_types = set(types1.index) | set(types2.index)

        # Build table
        table = [[types1.get(t, 0) for t in all_types],
                 [types2.get(t, 0) for t in all_types]]
        _pvalues[\"mstype\"].append(safe_chi2_contingency(table))

    # PDDS (Mann-Whitney U test)
    _pvalues[\"pdds\"] = []
    for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
        ds1, ds2 = comparison
        pdds1 = baseline_characteristics_df.query(f'dataset == \"{ds1}\"').pdds_scr
        pdds2 = baseline_characteristics_df.query(f'dataset == \"{ds2}\"').pdds_scr
        _pvalues[\"pdds\"].append(safe_mannwhitneyu(pdds1, pdds2))

    # MS type subtypes
    for mstype_var in [\"mstype_cis\", \"mstype_prms\", \"mstype_rrms\", \"mstype_spms\", \"mstype_ppms\"]:
        _pvalues[mstype_var] = []
        for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
            ds1, ds2 = comparison
            # Get counts (total vs specific subtype)
            total1 = len(baseline_characteristics_df.query(f'dataset == \"{ds1}\" and mstype.notnull()'))
            total2 = len(baseline_characteristics_df.query(f'dataset == \"{ds2}\" and mstype.notnull()'))

            # Parse the mstype_var name to get the condition
            mstype_map = {
                \"mstype_cis\": \"Clinically Isolated Syndrome\",
                \"mstype_prms\": \"Progressive Relapsing MS\",
                \"mstype_rrms\": \"Relapsing Remitting MS\",
                \"mstype_spms\": \"Secondary Progressive MS\",
                \"mstype_ppms\": \"Primary Progressive MS\"
            }

            subtype1 = len(baseline_characteristics_df.query(f'dataset == \"{ds1}\" and mstype == \"{mstype_map[mstype_var]}\"'))
            subtype2 = len(baseline_characteristics_df.query(f'dataset == \"{ds2}\" and mstype == \"{mstype_map[mstype_var]}\"'))

            # Create table: [count_of_interest, other_count] for each dataset
            table = [[subtype1, total1 - subtype1], [subtype2, total2 - subtype2]]
            _pvalues[mstype_var].append(safe_chi2_contingency(table))

    # Progressors for each test (Fisher's exact test)
    for test in tests:
        _pvalues[f\"progressors_{test}\"] = []
        for comparison in [(\"training\", \"validation\"), (\"training\", \"external test\")]:
            ds1, ds2 = comparison
            col = progression_col(test)

            # Get counts in the progressor column
            prog1 = len(baseline_characteristics_df.query(f'dataset == \"{ds1}\" and ~{col}.isna() and {col} == 1'))
            non_prog1 = len(baseline_characteristics_df.query(f'dataset == \"{ds1}\" and ~{col}.isna() and {col} == 0'))
            prog2 = len(baseline_characteristics_df.query(f'dataset == \"{ds2}\" and ~{col}.isna() and {col} == 1'))
            non_prog2 = len(baseline_characteristics_df.query(f'dataset == \"{ds2}\" and ~{col}.isna() and {col} == 0'))

            table = [[prog1, non_prog1], [prog2, non_prog2]]
            _pvalues[f\"progressors_{test}\"].append(safe_fisher_exact(table))

    pd.DataFrame(_pvalues).transpose()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demographics


    """)

    mo.md(r"""
    \begin{tabular}{lrr}
    \toprule
        Category & Count & Percentage \\
    \midrule

    \bottomrule
    \end{tabular}
    """)
    return


@app.cell
def _(baseline_characteristics_df):
    mstype_counts = baseline_characteristics_df["mstype"].value_counts()
    mstype_counts_percentage = (
        mstype_counts / len(baseline_characteristics_df.query("mstype.notnull()"))
    ) * 100

    print(mstype_counts)
    print(mstype_counts_percentage)
    return


@app.cell
def _(dataset_df):
    for _ds in dataset_df.dataset.unique():
        _per = len(dataset_df.query(f'dataset == "{_ds}"')) * 100.0 / len(dataset_df) 
        print(f'{_ds}: {_per}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Median time between MRI and baseline functional assessment
    """)
    return


@app.cell
def _(baseline_characteristics_df, pd):
    pd.to_datetime(baseline_characteristics_df.sty_date.str.replace('.0',''), unit="s", errors="coerce")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demographics

    | Category | Count | Percentage |
    | --- | --- | --- |
    | Clinically Isolated Syndrome |  |  |
    | Relapsing Remitting MS |  |  |
    | Secondary Progressive MS |  |  |
    | Primary Progressive MS |  |  |
    | Other |  |  |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Performances
    """)
    return


@app.cell
def _(batchsize, eval_dir, imagesize, isfile, join, models, pd, tests):
    _performance = {}

    _performance['test'] = []
    _performance['modality'] = []
    _performance['model'] = []
    _performance['AUROC'] = []
    _performance['AUPRC'] = []
    _performance['Accuracy'] = []
    _performance['F1-Score'] = [] 


    for _neurotest in tests:
        for _modality in ['t1w', 'flair']:
            for _model in models:
                _performance['test'].append(_neurotest)
                _performance['modality'].append(_modality)    
                _performance['model'].append(_model)
                _ftable = join(eval_dir, 'summary', _model, 'test', 'mspaths2', _modality, f'worst_progressor_2ycutoff_{_neurotest}_2z_e1000_b{batchsize}_im{imagesize}.csv')
                if isfile(_ftable):
                    _df = pd.read_csv(_ftable)
                    _performance['AUROC'].append(f'{_df["AUROC"][0]:0.2f} ({_df["AUROC_CI_lower"][0]:0.2f} | {_df["AUROC_CI_upper"][0]:0.2f}) ')
                    _performance['AUPRC'].append(f'{_df["AUPRC"][0]:0.2f} ({_df["AUPRC_CI_lower"][0]:0.2f} | {_df["AUPRC_CI_upper"][0]:0.2f}) ')
                else:
                   _performance['AUROC'].append(pd.NA)
                   _performance['AUPRC'].append(pd.NA)

                _ftable = join(eval_dir, 'metrics', _model, 'test', 'mspaths2', _modality, f'worst_progressor_2ycutoff_{_neurotest}_2z_e1000_b{batchsize}_im{imagesize}.csv')

                if isfile(_ftable):    
                    _df = pd.read_csv(_ftable)
                    _performance['Accuracy'].append(_df.accuracy[0])
                    _performance['F1-Score'].append(_df.f1[0])
                else:
                    _performance['Accuracy'].append(pd.NA)
                    _performance['F1-Score'].append(pd.NA)


    pd.DataFrame(_performance)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
