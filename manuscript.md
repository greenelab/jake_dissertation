---
title: Jake Crawford dissertation title
keywords:
- gene-expression
- cancer-genomics
- machine-learning
- optimization
- domain-adaptation
lang: en-US
date-meta: '2023-09-04'
author-meta:
- Jake Crawford
header-includes: |
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta property="og:type" content="article" />
  <meta name="dc.title" content="Jake Crawford dissertation title" />
  <meta name="citation_title" content="Jake Crawford dissertation title" />
  <meta property="og:title" content="Jake Crawford dissertation title" />
  <meta property="twitter:title" content="Jake Crawford dissertation title" />
  <meta name="dc.date" content="2023-09-04" />
  <meta name="citation_publication_date" content="2023-09-04" />
  <meta property="article:published_time" content="2023-09-04" />
  <meta name="dc.modified" content="2023-09-04T17:21:30+00:00" />
  <meta property="article:modified_time" content="2023-09-04T17:21:30+00:00" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Jake Crawford" />
  <meta name="citation_author_institution" content="Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA" />
  <meta name="citation_author_orcid" content="0000-0001-6207-0782" />
  <meta name="twitter:creator" content="@jjc2718" />
  <link rel="canonical" href="https://greenelab.github.io/jake_dissertation/" />
  <meta property="og:url" content="https://greenelab.github.io/jake_dissertation/" />
  <meta property="twitter:url" content="https://greenelab.github.io/jake_dissertation/" />
  <meta name="citation_fulltext_html_url" content="https://greenelab.github.io/jake_dissertation/" />
  <meta name="citation_pdf_url" content="https://greenelab.github.io/jake_dissertation/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://greenelab.github.io/jake_dissertation/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://greenelab.github.io/jake_dissertation/v/00639a09cb825bd0de97caca4e71aaf4da3e5181/" />
  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/jake_dissertation/v/00639a09cb825bd0de97caca4e71aaf4da3e5181/" />
  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/jake_dissertation/v/00639a09cb825bd0de97caca4e71aaf4da3e5181/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://greenelab.github.io/jake_dissertation/v/00639a09cb825bd0de97caca4e71aaf4da3e5181/))
was automatically generated
from [greenelab/jake_dissertation@00639a0](https://github.com/greenelab/jake_dissertation/tree/00639a09cb825bd0de97caca4e71aaf4da3e5181)
on September 4, 2023.
</em></small>



## Authors



+ **Jake Crawford**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-6207-0782](https://orcid.org/0000-0001-6207-0782)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [jjc2718](https://github.com/jjc2718)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [jjc2718](https://twitter.com/jjc2718)
    <br>
  <small>
     Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA
  </small>


::: {#correspondence}
✉ — Correspondence possible via [GitHub Issues](https://github.com/greenelab/jake_dissertation/issues)

:::


## Abstract {.page_break_before}




## Chapter 1

* Modeling strategies (copy existing review)

* Machine learning for cancer transcriptomics (add some new text)


## Chapter 1: Background

This chapter was formatted for this dissertation to provide background information and context for the following chapters. Some elements of the second subsection on machine learning modeling techniques were previously published in the _Current Opinion in Biotechnology_ journal as "Incorporating biological structure into machine learning models in biomedicine" (https://doi.org/10.1016/j.copbio.2019.12.021).

**Contributions:**
For the unpublished parts of this chapter, I was the sole author.
For the published parts of this chapter, I wrote the original draft of the review paper, which was edited based on feedback from Casey S. Greene and anonymous reviewers.



### Introduction

Precision oncology, or the selection of cancer treatments based on molecular or cellular features of patients' tumors, has become a fundamental part of the standard of care for some cancers [@doi:10.1093/annonc/mdx707].
Although each tumor is unique, the successes of precision oncology reinforce the idea that there are commonalities that can be understood and therapeutically targeted.
Targeted therapies that have been successfully applied across cancer types and patient subsets include _HER2_ (_ERBB2_) inhibitors in breast and stomach cancer [@doi:10.1093/jnci/djp341], BTK inhibitors in various hematological malignancies [@doi:10.1186/s13045-022-01353-w], and _EGFR_ inhibitors across a variety of carcinomas [@doi:10.1186/s13045-022-01311-6].
The genes and mutations that drive cancer are often specific to a given cancer type or subtype, but they tend to converge on a few pathways [@doi:10.1016/j.cell.2018.03.035; @doi:10.1016/j.cell.2020.11.045], making more general targeted treatments possible.

The past decade has seen an expansion in the size and diversity of cancer genomics datasets, both publicly available and otherwise.
The Cancer Genome Atlas (TCGA) Pan-Cancer Atlas [@pancanatlas] is a large, public human tumor sample dataset, containing >10,000 samples from 33 different cancer types, each profiled for varying -omics types with associated clinical information.
There are also public datasets containing model system data, including the Cancer Cell Line Encyclopedia (CCLE) containing -omics data from human-derived cancer cell lines [@doi:10.1038/s41586-019-1186-3], and the Genomics of Drug Sensitivity in Cancer (GDSC) dataset containing drug sensitivity data for thousands of the same cell lines across hundreds of drugs [@doi:10.1093/nar/gks1111].
These datasets exhibit heterogeneity on multiple levels.
Overall, they vary in size, with TCGA having about an order of magnitude more tumor samples than the number of cell lines in CCLE.
Going a level deeper, the cancer types within them vary in size as well: TCGA has 1,218 breast cancer samples with gene expression data, but only 265 soft tissue sarcoma samples, and only 45 cholangiocarcinoma samples.

In modern machine learning research using text and images, there is a trend toward bigger models capable of solving broader arrays of tasks.
Foundation models, trained on large datasets to generalize to new tasks with no or minimal task-specific fine-tuning, are in many cases competitive with task-specific models [@arxiv:2205.09911], although they are not without unique caveats [@arxiv:2108.07258].
Similarly, in genomics, early examples of foundation models are beginning to appear [@arxiv:2306.15794; @doi:10.1101/2023.04.30.538439; @doi:10.1101/2023.05.29.542705].
Training foundation models on pan-cancer, pan-omics data would be a natural extension of these ideas, which could improve power to detect correlations between biomarkers and phenotypes of interest, or to identify drug susceptibilities in patient sub-populations.

As a whole, this dissertation explores ways in which the structure of large, public pan-cancer datasets can present unexpected challenges and caveats for machine learning.
TCGA and CCLE both contain data from various -omics types (feature groups) and samples from diverse cancer types/tissues of origin (sample groups).
There are additional, less obvious forms of structure in these data such as patient sub-populations and sample collection locations, which we will not address directly in this dissertation but which can affect model training and performance as well.

This chapter, Chapter 1, describes existing work at the intersection of cancer -omics and machine learning, which will provide context for the following chapters.
In Chapter 2, we show that the choice of optimization method can affect model selection and tuning, for prediction from cancer transcriptomic data.
Chapter 3 explores the relative information content of -omics types/feature groups in TCGA, showing that gene expression tends to contain the most information relative to cancer driver mutations, but most -omics types can serve as effective, and likely somewhat redundant, readouts.
In Chapter 4, we test generalization across cancer types in TCGA and across datasets (CCLE to TCGA and vice-versa), showing that smaller models do not tend to generalize better across contexts, and cross-validation performance is in most cases a sufficient model selection criterion.
Finally, in Chapter 5, we conclude by summarizing the implications of these results and discussing future directions.


## Cancer -omics data and applications

### Publicly available cancer -omics data resources

A wealth of public cancer genomics and multi-omics human sample resources have been generated in the past decade.
As mentioned in the introduction, the TCGA Pan-Cancer Atlas [@pancanatlas] contains data spanning 33 cancer types and multiple -omics data types, including mutation, CNV, gene expression, miRNA, DNA methylation, reverse phase protein array (RPPA) proteomics data, and clinical outcome data [@doi:10.1016/j.cell.2018.02.052].
The International Cancer Genome Consortium (ICGC) data portal is an initiative to unite and harmonize data from many worldwide cancer projects including TCGA, mostly focused on DNA/somatic mutation data but containing some gene expression and other -omics data [@doi:10.1038/s41587-019-0055-9].
The Pan-Cancer Analysis of Whole Genomes (PCAWG) project attempts to expand from the whole-exome sequencing provided by TCGA to whole-genome sequencing, providing data and analysis for 2,658 whole genome cancer samples [@doi:10.1038/s41586-020-1969-6]
The American Association for Cancer Research (AACR)'s Project GENIE (Genomics, Evidence, Neoplasia, Information, Exchange) is another large-scale initiative to share genomic data, with the intention of complementing TCGA and allowing for external validation of methods and biological findings [@doi:10.1158/2159-8290.CD-17-0151].
Unlike TCGA, which contains whole-exome sequencing data, the GENIE dataset is primarily comprised of targeted sequencing panels of a subset of cancer-relevant genes.

In addition to samples derived from human tumors or neoplasms, data from model systems such as cancer cell lines and mouse models are an important element of therapeutic development.
The Cancer Cell Line Encyclopedia (CCLE) contains a variety of uniformly processed -omics data across more than 1000 human-derived cell lines, including somatic mutations, CNV data, gene fusion information, and gene expression [@doi:10.1038/s41586-019-1186-3].
The Cancer Dependency Map (DepMap) complements CCLE with information about cancer cell line vulnerabilities, derived from CRISPR and RNAi knockout screens [@doi:10.1038/s41467-021-21898-7; @doi:10.1038/s41588-021-00819-w].
The Connectivity Map (CMap) and LINCS L1000 project aims to catalog the responses of cell lines to both genetic and pharmacological perturbations, identifying the changes to gene expression and protein expression that result [@doi:10.1016/j.cell.2017.10.049].
The GDSC and PRISM drug screening datasets provide cell viability dose-response readings for many of the cell lines in CCLE, after perturbation with small molecules [@doi:10.1093/nar/gks1111; @doi:10.1038/s43018-019-0018-6].
Aside from cell lines, the PDX Encyclopedia is a dataset of patient-derived xenograft (PDX) mouse model data, including more than 1000 models with mutation, CNV, and gene expression data for each [@doi:10.1038/nm.3954].
The National Cancer Institute's Patient-Derived Models Repository (PDMR) also contains mutation and gene expression profiles for mouse models and patient-derived tumor organoids, or tumoroids [@doi:10.1038/s41467-021-25177-3; @pdmr], although it is still under development.

### Applications of machine learning in cancer genomics

Historically, one common use of -omics data in cancer has been to define subtypes, or clinically relevant patient subsets that may have similar prognosis or respond similarly to therapy.
Many studies have sought to distinguish tumor samples from control/normal samples, to identify subtypes of a particular cancer type, or to distinguish samples of a particular cancer type/tissue of origin from samples of other cancer types (e.g. [@doi:10.1186/s12920-020-0677-2; @doi:10.3389/fbioe.2020.00737; @doi:10.1109/TBME.2012.2225622; @doi:10.1186/s13073-023-01176-5]).
External validation is difficult, however, since samples in TCGA were taken from patients who had already been clinically diagnosed with a particular cancer type or subtype, i.e. without using machine learning.
Potentially a more clinically relevant way to frame the problem is to classify cancers of unknown primary (CUP), which are metastatic cancers where the primary site cannot be identified in the clinic.
Machine learning approaches have identified cell lineages and developmental trajectories for CUP samples [@doi:10.1158/2159-8290.CD-21-1443] and integrated electronic health record (EHR) data and genomic data to suggest targeted therapies for CUP patients [@doi:10.1038/s41591-023-02482-6].
Relatedly, distinguishing primary samples from metastatic samples, or predicting metastatic potential of primary samples, is another classification problem which -omics data has been used for [@doi:10.1038/s41467-019-13825-8; @doi:10.1101/2020.09.07.286583; @doi:10.1371/journal.pcbi.1009956]

Prediction of drug response from genomic data, often combined with clinical features or other metadata, is a machine learning problem with clear clinical applications.
Given the availability and uniformity of the cell line data in CCLE, and drug response data in GDSC, PRISM and other cell line datasets, many method development efforts have centered on these data sources.
Examples include prediction of drug response from integrated multi-omics data [@doi:10.1093/bioinformatics/btz318], prediction of drug response using perturbation modeling via CMap as an intermediate step [@doi:10.1093/bioinformatics/btz158], and prediction of drug response via single-cell transcriptomic data [@doi:10.1101/2022.01.11.475728], among many others reviewed in [@doi:10.1093/bib/bbab294; @doi:10.1038/s41467-022-34277-7; @doi:10.1038/s41598-023-39179-2].
Large datasets of human-derived genomic data with associated drug response annotations are more difficult to find.
Still, there have been attempts to develop and/or validate models on human data, including for prediction of immunotherapy response which benefits from applications across a wide range of cancer types [@doi:10.1038/s41587-021-01070-8; @doi:10.1016/j.ccell.2023.06.006; @doi:10.1101/2020.09.03.260265].
Prognosis or patient survival prediction from multi-omics data is another area of modeling that leverages widely available clinical metadata, reviewed in detail in many existing papers [@doi:10.1186/1471-2288-12-102; @doi:10.1093/bib/bbu003; @doi:10.1186/s12885-021-08796-3; @doi:10.1016/j.csbj.2014.11.005].

Much of our work, described later in this thesis, stems from the idea of predicting the mutation status in key driver genes of cancer samples, based on functional readouts such as gene expression [@doi:10.1158/1078-0432.CCR-13-1943; @doi:10.1016/j.celrep.2018.03.046; @doi:10.1186/s13059-020-02021-3; @doi:10.1371/journal.pone.0241514].
At first consideration, an accurate mutation status classifier may not seem particularly useful, since for a patient sample a clinician could simply sequence the genome, or select genes in the genome, to determine driver mutation status.
One application of accurate mutation status classifiers, however, is to identify samples with a similar phenotype to those with a driver mutation, but _without_ the mutation being present in DNA sequencing data.
Observed examples of this phenomenon include the "BRCAness" phenotype in tumors without observed _BRCA1_/_BRCA2_ mutations [@doi:10.1038/nrc.2015.21], and the "Ph-like" leukemia phenotype in the absence of the Philadelphia chromosome fusion [@doi:10.1182/asheducation-2016.1.561], among others.
Following this line of reasoning, algorithms have been developed to identify mutations that "phenocopy" known cancer drivers [@doi:10.1142/9789811215636_0031; @doi:10.1101/2022.07.28.501874], and to integrate this information into drug response prediction pipelines to define larger and more accurate patient subgroups [@doi:10.1038/s41525-022-00328-7].
Related machine learning approaches to genomic prediction/phenotype identification include methods for identifying DNA damage repair deficiencies based on genomic data [@doi:10.1038/nm.4292; @doi:10.1038/s43018-022-00474-y] and for identifying synthetic lethal relationships for use in targeted therapy selection [@doi:10.1016/j.cell.2021.03.030].
Such methods could be useful for defining broader and more representative patient groups than would be possible based solely on somatic mutation status, that may exhibit similar tumor phenotypes or respond to similar therapies.
For example, in "basket" clinical trials where patients are included across cancer types based on the presence or absence of individual molecular markers [@doi:10.1200/jco.2014.58.2007], including "phenocopies" could improve efficacy for some targeted therapies.


## Machine learning modeling strategies for -omics data

It can be challenging to distinguish signal from noise in biomedical datasets, and machine learning methods are particularly hampered when the amount of available training data is small.
Incorporating biomedical knowledge into machine learning models can reveal patterns in noisy data [@doi:10.1038/nrg.2017.38] and aid model interpretation [@doi:10.1016/j.cell.2018.05.056].
Biological knowledge can take many forms, including genomic sequences, pathway databases, gene interaction networks, and knowledge hierarchies such as the Gene Ontology [@doi:10.1093/nar/gky1055].
However, there is often no canonical way to encode these structures as real-valued predictors.
Modelers must creatively decide how to encode biological knowledge that they expect will be relevant to the task.

Biomedical datasets often contain more input predictors than data samples [@doi:10.1109/JPROC.2015.2494198; @arxiv:1611.09340].
A genetic study may genotype millions of single nucleotide polymorphisms (SNPs) in thousands of individuals, or a gene expression study may profile the expression of thousands of genes in tens of samples.
Thus, it can be useful to include prior information describing relationships between predictors to inform the representation learned by the model.
This contrasts with non-biological applications of machine learning, where one might fit a model on millions of images [@doi:10.1109/CVPR.2009.5206848] or tens of thousands of documents [@url:https://www.aclweb.org/anthology/P11-1015/], making inclusion of prior information unnecessary.
There are many complementary ways to incorporate heterogeneous sources of biomedical data into the learning process, which have been covered in review papers elsewhere [@doi:10.3389/fgene.2019.00381; @doi:10.1016/j.inffus.2018.09.012].
These include feature extraction or representation learning prior to modeling and/or other data integration methods that do not necessarily involve customizing the model itself.

### Sequence models

Early neural network models primarily used hand-engineered sequence features as input to a fully connected neural network [@doi:10.1093/nar/gku1058; @doi:10.1126/science.1254806] (Figure {@fig:sequence_features}).
As convolutional neural network (CNN) approaches matured for image processing and computer vision, researchers leveraged biological sequence proximity similarly.
CNNs are a neural network variant that groups input data by spatial context to extract features for prediction.

The definition of "spatial context" is specific to the input: one might group image pixels that are nearby in 2D space, or genomic base pairs that are nearby in the linear genome.
In this way, CNNs consider context without making strong assumptions about exactly how much context is needed or how it should be encoded; the data informs the encoding.
A detailed description of how CNNs are applied to sequences can be found in Angermueller et al. [@doi:10.15252/msb.20156651].

![
    Contrasting approaches to extracting features from DNA or RNA sequence data.
    Early models defined features of interest by hand based on prior knowledge about the prediction or clustering problem of interest, such as GC content or sequence melting point, as depicted in the left branch in the figure.
    Convolutional models, depicted in the right branch, use sequence convolutions to derive features directly from sequence proximity, without requiring quantities of interest to be identified before the model is trained.
    Red or blue emphasis denotes inputs to the predictive model (either the hand-defined numeric features on the left or the outputs of convolutional filters on the right).
](images/biopriors/sequence_features_revised.svg){#fig:sequence_features .white}

#### Applications in regulatory biology

Many early applications of deep learning to biological sequences were in regulatory biology.
Early CNNs for sequence data predicted binding protein sequence specificity from DNA or RNA sequence [@doi:10.1038/nbt.3300], variant effects from noncoding DNA sequence [@doi:10.1038/nmeth.3547], and chromatin accessibility from DNA sequence [@doi:10.1101/gr.200535.115].

Recent sequence models take advantage of hardware advances and methodological innovation to incorporate more sequence context and rely on fewer modeling assumptions.
BPNet, a CNN that predicts transcription factor binding profiles from DNA sequences, accurately mapped known locations of binding motifs in mouse embryonic stem cells [@doi:10.1101/737981].
BPNet considers 1000 base pairs of context around each position when predicting binding probabilities with a technique called dilated convolutions [@arxiv:1511.07122], which is particularly important because motif spacing and periodicity can influence binding.
cDeepbind [@doi:10.1101/345140] combines RNA sequences with information about secondary structure to predict RNA binding protein affinities.
Its convolutional model acts on a feature vector combining sequence and structural information, using context for both to inform predictions.
APARENT [@doi:10.1016/j.cell.2019.04.046] is a CNN that predicts alternative polyadenylation (APA) from a training set of over 3 million synthetic APA reporter sequences.
These diverse applications underscore the power of modern deep learning models to synthesize large sequence datasets.

Models that consider sequence context have also been applied to epigenetic data.
DeepSignal [@doi:10.1093/bioinformatics/btz276] is a CNN that uses contextual electrical signals from Oxford Nanopore single-molecule sequencing data to predict 5mC or 6mA DNA methylation status.
MRCNN [@doi:10.1186/s12864-019-5488-5] uses sequences of length 400, centered at CpG sites, to predict 5mC methylation status.
Deep learning models have also been used to predict gene expression from histone modifications [@doi:10.1101/329334; @doi:10.1093/bioinformatics/bty612].
Here, a neural network model consisting of long short-term memory (LSTM) units was used to encode the long-distance interactions of histone marks in both the 3' and 5' genomic directions.
In each of these cases, proximity in the linear genome helped model the complex interactions between DNA sequence and epigenome.

#### Applications in variant calling and mutation detection

Identification of genetic variants also benefits from models that include sequence context.
DeepVariant [@doi:10.1038/nbt.4235] applies a CNN to images of sequence read pileups, using read data around each candidate variant to accurately distinguish true variants from sequencing errors.
CNNs have also been applied to single molecule (PacBio and Oxford Nanopore) sequencing data [@doi:10.1038/s41467-019-09025-z], using a different sequence encoding that results in better performance than DeepVariant on single molecule data.
However, many variant calling models still use hand-engineered sequence features as input to a classifier, including current state-of-the-art approaches to insertion/deletion calling [@doi:10.1101/601450; @doi:10.1101/628222].
Detection of somatic mutations is a distinct but related challenge to detection of germline variants, and has also recently benefitted from use of CNNs [@doi:10.1038/s41467-019-09027-x].

### Network- and pathway-based models

Rather than operating on sequences, many machine learning models in biomedicine operate on inputs that lack intrinsic order.
Models may make use of gene expression data matrices from RNA sequencing or microarray experiments in which rows represent samples and columns represent genes.
To account for relationships between genes, one might incorporate known interactions or correlations when making predictions or generating a low-dimensional representation of the data (Figure {@fig:network_models}).
This is comparable to the manner in which sequence context pushes models to consider nearby base pairs similarly.

![
    The relationships between genes provide structure that can be incorporated into machine learning models.
    One common approach is to use a network or collection of gene sets to embed the data in a lower-dimensional space, in which genes that are in the same gene sets or that are well-connected in the network have a similar representation in the lower-dimensional space.
    The embedded data can then be used for classification or clustering tasks.
    The "x" values in the data table represent gene expression measurements.
](images/biopriors/network_models_revised.svg){#fig:network_models .white}

#### Applications in transcriptomics

Models built from gene expression data can benefit from incorporating gene-level relationships.
One form that this knowledge commonly takes is a database of gene sets, which may represent biological pathways or gene signatures for a biological state of interest.
PLIER [@doi:10.1038/s41592-019-0456-1] uses gene set information from MSigDB [@doi:10.1073/pnas.0506580102] and cell type markers to extract a representation of gene expression data that corresponds to biological processes and reduces technical noise.
The resulting gene set-aligned representation accurately decomposed cell type mixtures.
MultiPLIER [@doi:10.1016/j.cels.2019.04.003] applied PLIER to the recount2 gene expression compendium [@doi:10.1038/nbt.3838] to develop a model that shares information across multiple tissues and diseases, including rare diseases with limited sample sizes.
PASNet [@doi:10.1186/s12859-018-2500-z] uses MSigDB to inform the structure of a neural network for predicting patient outcomes in glioblastoma multiforme (GBM) from gene expression data.
This approach aids interpretation, as pathway nodes in the network with high weights can be inferred to correspond to certain pathways in GBM outcome prediction.

Gene-level relationships can also be represented with networks.
Network nodes typically represent genes and real-valued edges may represent interactions or correlations between genes, often in a tissue or cell type context of interest.
Network-based stratification [@doi:10.1038/nmeth.2651] is an early example of a method for utilizing gene interaction network data in this manner, applied to improve subtyping across several cancer types.
More recently, netNMF-sc [@doi:10.1101/544346] incorporates coexpression networks [@doi:10.1093/nar/gkw868] as a smoothing term for dimension reduction and dropout imputation in single-cell gene expression data.
The coexpression network improves performance for identifying cell types and cell cycle marker genes, as compared to using raw gene expression or other single-cell dimension reduction methods.
Combining gene expression data with a network-derived smoothing term also improved prediction of patient drug response in acute myeloid leukemia [@arxiv:1906.10670] and detection of mutated cancer genes [@doi:10.1038/s41598-017-03141-w].
PIMKL [@doi:10.1038/s41540-019-0086-3] combines network and pathway data to predict disease-free survival from breast cancer cohorts.
This method takes as input both RNA-seq gene expression data and copy number alteration data, but can also be applied to gene expression data alone.

Gene regulatory networks can also augment models for gene expression data.
These networks describe how the expression of genes is modulated by biological regulators such as transcription factors, microRNAs, or small molecules.
creNET [@doi:10.1038/s41598-018-19635-0] integrates a gene regulatory network, derived from STRING [@doi:10.1093/nar/gku1003], with a sparse logistic regression model to predict phenotypic response in clinical trials for ulcerative colitis and acute kidney rejection.
The gene regulatory information allows the model to identify the biological regulators associated with the response, potentially giving mechanistic insight into differential clinical trial response.
GRRANN [@doi:10.1186/s12859-017-1984-2], which was applied to the same data as creNET, uses a gene regulatory network to inform the structure of a neural network.
Several other methods [@doi:10.1093/nar/gkx681; @doi:10.1093/bioinformatics/bty945] have also used gene regulatory network structure to constrain the structure of a neural network, reducing the number of parameters to be fit and facilitating interpretation.

#### Applications in genetics

Approaches that incorporate gene set or network structure into genetic studies have a long history [@doi:10.1093/biostatistics/kxl007; @doi:10.1093/bioinformatics/btn081].
Recent applications include expression quantitative trait loci (eQTL) mapping studies, which aim to identify associations between genetic variants and gene expression.
netReg [@doi:10.1093/bioinformatics/btx677] implements a graph-regularized dual LASSO algorithm for eQTL mapping [@doi:10.1093/bioinformatics/btu293] in a publicly available R package.
This model smooths regression coefficients simultaneously based on networks describing associations between genes (target variables in the eQTL regression model) and between variants (predictors in the eQTL regression model).
eQTL information is also used in conjunction with genetic variant information to predict phenotypes, in an approach known as Mendelian randomization (MR).
In [@doi:10.1111/biom.13072], a smoothing term derived from a gene regulatory network is used in an MR model.
The model with the network smoothing term, applied to a human liver dataset, more robustly identifies genes that influence enzyme activity than a network-agnostic model.
As genetic datasets grow, we expect that researchers will continue to develop models that leverage gene set and network databases.

### Other models incorporating biological structure

Knowledge about biological entities is often organized in an ontology, which is a directed graph that encodes relationships between entities (see Figure {@fig:ontology_models} for a visual example).
The Gene Ontology (GO) [@doi:10.1093/nar/gky1055] describes the relationships between cellular subsystems and other attributes describing proteins or genes.
DCell [@doi:10.1038/nmeth.4627] uses GO to inform the connectivity of a neural network predicting the effects of gene deletions on yeast growth.
DCell performs comparably to an unconstrained neural network for this task.
Additionally, it is easier to interpret: a cellular subsystem with high neuron outputs under a particular gene deletion can be inferred to be strongly affected by the gene deletion, providing a putative genotype-phenotype association.
DeepGO [@doi:10.1093/bioinformatics/btx624] uses a similar approach to predict protein function from amino acid sequence with a neural network constrained by the dependencies of GO.
However, a follow-up paper by the same authors [@doi:10.1093/bioinformatics/btz595] showed that this hierarchy-aware approach can be outperformed by a hierarchy-naive CNN, which uses only amino acid sequence and similarity to labeled training set proteins.
This suggests a tradeoff between interpretability and predictive accuracy for protein function prediction.

![
    Directed graph-structured data, such as an ontology or phylogenetic tree, can be incorporated into machine learning models.
    Here, the connections in the neural network used to predict a set of labels parallel those in the tree graph.
    This type of constraint can also be useful in model interpretation: for example, if the nodes in the right tree branch have high neuron outputs for a given sample, then the subsystem encoded in the right branch of the tree graph is most likely important in making predictions for that sample.
    The "x" values in the data table represent gene expression measurements.
](images/biopriors/ontology_models_revised.svg){#fig:ontology_models .white}

Phylogenetic trees, or hierarchies describing the evolutionary relationships between species, can be useful for a similar purpose.
glmmTree [@doi:10.3389/fmicb.2018.01391] uses a phylogenetic tree describing the relationship between microorganisms to improve predictions of age based on gut microbiome data.
The same authors combine a similar phylogeny smoothing strategy with sparse regression to model caffeine intake and smoking status based on microbiome data [@doi:10.3389/fmicb.2018.03112].
Phylogenetic trees can also describe the relationships between subclones of a tumor, which are fundamental to understanding cancer evolution and development.
Using a tumor phylogeny inferred from copy number aberration (CNA) sequencing data as a smoothing term for deconvolving tumor subclones provided more robust predictions than a phylogeny-free model [@doi:10.1007/978-3-030-17083-7_11].
The tree structure of the phylogeny and the subclone mixture model are fit jointly to the CNA data.

Depending on the application, other forms of structure or prior knowledge can inform predictions and interpretation of the model's output.
CYCLOPS [@doi:10.1073/pnas.1619320114] uses a circular node autoencoder [@doi:10.1162/neco.1996.8.2.390] to order periodic gene expression data and estimate circadian rhythms.
The authors validated the method by correctly ordering samples without temporal labels and identifying genes with known circadian expression.
They then applied it to compare gene expression in normal and cancerous liver biopsies, identifying drug targets with circadian expression as candidates for chronotherapy.
NetBiTE [@arxiv:1808.06603] uses drug-gene interaction information from GDSC [@doi:10.1093/nar/gks1111], in addition to protein interaction data, to build a tree ensemble model with splits that are biased toward high-confidence drug-gene interactions.
The model predicts sensitivity to drugs that inhibit critical signaling pathways in cancer, showing improved predictive performance compared to random forests, another commonly used tree ensemble model.

### Conclusions

As the quantity and richness of biomedical data has increased, sequence repositories and interaction databases have expanded and become more robust.
This raises opportunities to integrate these resources into the structure of machine learning models.
There have been several past attempts to benchmark and compare approaches to integrating structured data into predictive models in biomedicine, including the evaluation in [@doi:10.1371/journal.pone.0034796] and more recent studies in [@arxiv:1905.02295] and [@arxiv:1910.09600].
Extending and broadening benchmarking efforts such as these will be vital, improving our understanding of the suitability of problem domains and datasets for the classes of methods described in this review.
In the future, we foresee that incorporating structured biomedical data will become commonplace for improving model interpretability and boosting performance when sample size is limited.


## Chapter 2: optimization strongly influences model selection in transcriptomic prediction

This chapter has been posted as a preprint on bioRxiv (https://www.biorxiv.org/content/10.1101/2023.06.26.546586v1) and submitted for publication at Bioinformatics Advances as "Optimizer’s dilemma: optimization strongly influences model selection in transcriptomic prediction".

**Contributions**:
I designed and ran the experiments, created the figures, wrote the initial draft of the manuscript, and edited the manuscript. Maria Chikina gave feedback on an initial version of the manuscript, gave guidance on experimental design, and edited the manuscript. Casey S. Greene gave feedback and guidance on experiments, and edited the manuscript.



### Abstract

#### Motivation

Most models can be fit to data using various optimization approaches.
While model choice is frequently reported in machine-learning-based research, optimizers are not often noted.
We applied two different implementations of LASSO logistic regression implemented in Python's scikit-learn package, using two different optimization approaches (coordinate descent and stochastic gradient descent), to predict driver mutation presence or absence from gene expression across 84 pan-cancer driver genes.
Across varying levels of regularization, we compared performance and model sparsity between optimizers.

#### Results

After model selection and tuning, we found that coordinate descent (implemented in the `liblinear` library) and SGD tended to perform comparably.
`liblinear` models required more extensive tuning of regularization strength, performing best for high model sparsities (more nonzero coefficients), but did not require selection of a learning rate parameter.
SGD models required tuning of the learning rate to perform well, but generally performed more robustly across different model sparsities as regularization strength decreased.
Given these tradeoffs, we believe that the choice of optimizers should be clearly reported as a part of the model selection and validation process, to allow readers and reviewers to better understand the context in which results have been generated.

#### Availability and implementation

The code used to carry out the analyses in this study is available at <https://github.com/greenelab/pancancer-evaluation/tree/master/01_stratified_classification>. Performance/regularization strength curves for all genes in the Vogelstein et al. 2013 dataset are available at <https://doi.org/10.6084/m9.figshare.22728644>.


### Introduction

Gene expression profiles are widely used to classify samples or patients into relevant groups or categories, both preclinically [@doi:10.1371/journal.pcbi.1009926; @doi:10.1093/bioinformatics/btaa150] and clinically [@doi:10.1200/JCO.2008.18.1370; @doi:10/bp4rtw].
To extract informative gene features and to perform classification, a diverse array of algorithms exist, and different algorithms perform well across varying datasets and tasks [@doi:10.1371/journal.pcbi.1009926].
Even within a given model class, multiple optimization methods can often be applied to find well-performing model parameters or to optimize a model's loss function.
One commonly used example is logistic regression.
The widely used scikit-learn Python package for machine learning [@url:https://jmlr.org/papers/v12/pedregosa11a.html] provides two modules for fitting logistic regression classifiers: `LogisticRegression`, which uses the `liblinear` coordinate descent method [@url:https://www.jmlr.org/papers/v9/fan08a.html] to find parameters that optimize the logistic loss function, and `SGDClassifier`, which uses stochastic gradient descent [@online-learning] to optimize the same loss function.

Using scikit-learn, we compared the `liblinear` (coordinate descent) and SGD optimization techniques for prediction of driver mutation status in tumor samples, across a wide variety of genes implicated in cancer initiation and development [@doi:10.1126/science.1235122].
We applied LASSO (L1-regularized) logistic regression, and tuned the strength of the regularization to compare model selection between optimizers.
We found that across a variety of models (i.e. varying regularization strengths), the training dynamics of the optimizers were considerably different: models fit using `liblinear` tended to perform best at fairly high regularization strengths (100-1000 nonzero features in the model) and overfit easily with low regularization strengths.
On the other hand, after tuning the learning rate, models fit using SGD tended to perform well across both higher and lower regularization strengths, and overfitting was less common.

Our results caution against viewing optimizer choice as a "black box" component of machine learning modeling.
The observation that LASSO logistic regression models fit using SGD tended to perform well for low levels of regularization, across diverse driver genes, runs counter to conventional wisdom in machine learning for high-dimensional data which generally states that explicit regularization and/or feature selection is necessary.
Comparing optimizers or model implementations directly is rare in applications of machine learning for genomics, and our work shows that this choice can affect generalization and interpretation properties of the model significantly.
Based on our results, we recommend considering the appropriate optimization approach carefully based on the goals of each individual analysis.


### Methods

#### Data download and preprocessing

To generate binary mutated/non-mutated gene labels for our machine learning model, we used mutation calls for TCGA Pan-Cancer Atlas samples from MC3 [@doi:10.1016/j.cels.2018.03.002] and copy number threshold calls from GISTIC2.0 [@doi:10.1186/gb-2011-12-4-r41].
MC3 mutation calls were downloaded from the Genomic Data Commons (GDC) of the National Cancer Institute, at <https://gdc.cancer.gov/about-data/publications/pancanatlas>.
Thresholded copy number calls are from an older version of the GDC data and are available here: <https://figshare.com/articles/dataset/TCGA_PanCanAtlas_Copy_Number_Data/6144122>.
We removed hypermutated samples, defined as two or more standard deviations above the mean non-silent somatic mutation count, from our dataset to reduce the number of false positives (i.e., non-driver mutations).
Any sample with either a non-silent somatic variant or a copy number variation (copy number gain in the target gene for oncogenes and copy number loss in the target gene for tumor suppressor genes) was included in the positive set; all remaining samples were considered negative for mutation in the target gene.

RNA sequencing data for TCGA was downloaded from GDC at the same link provided above for the Pan-Cancer Atlas.
We discarded non-protein-coding genes and genes that failed to map and removed tumors that were measured from multiple sites.
After filtering to remove hypermutated samples and taking the intersection of samples with both mutation and gene expression data, 9074 total TCGA samples remained.

#### Cancer gene set construction

In order to study mutation status classification for a diverse set of cancer driver genes, we started with the set of 125 frequently altered genes from Vogelstein et al. [@doi:10.1126/science.1235122] (all genes from Table S2A).
For each target gene, in order to ensure that the training dataset was reasonably balanced (i.e., that there would be enough mutated samples to train an effective classifier), we included only cancer types with at least 15 mutated samples and at least 5% mutated samples, which we refer to here as "valid" cancer types.
In some cases, this resulted in genes with no valid cancer types, which we dropped from the analysis.
Out of the 125 genes originally listed in the Vogelstein et al. cancer gene set, we retained 84 target genes after filtering for valid cancer types.

#### Classifier setup and optimizer comparison details

We trained logistic regression classifiers to predict whether or not a given sample had a mutational event in a given target gene using gene expression features as explanatory variables.
Based on our previous work, gene expression is generally effective for this problem across many target genes, although other -omics types can be equally effective in many cases [@doi:10.1186/s13059-022-02705-y].
Our model was trained on gene expression data (X) to predict mutation presence or absence (y) in a target gene.
To control for varying mutation burden per sample and to adjust for potential cancer type-specific expression patterns, we included one-hot encoded cancer type and log~10~(sample mutation count) in the model as covariates.
Since gene expression datasets tend to have many dimensions and comparatively few samples, we used a LASSO penalty to perform feature selection [@doi:10.1111/j.2517-6161.1996.tb02080.x].
LASSO logistic regression has the advantage of generating sparse models (some or most coefficients are 0), as well as having a single tunable hyperparameter which can be easily interpreted as an indicator of regularization strength, or model complexity.

To compare model selection across optimizers, we first split the "valid" cancer types into train (75%) and test (25%) sets.
We then split the training data into "subtrain" (66% of the training set) data to train the model on, and "holdout" (33% of the training set) data to perform model selection, i.e. to use to select the best-performing regularization parameter, and the best-performing learning rate for SGD in the cases where multiple learning rates were considered.
In each case, these splits were stratified by cancer type, i.e. each split had as close as possible to equal proportions of each cancer type included in the dataset for the given driver gene.

#### LASSO parameter range selection and comparison between optimizers

The scikit-learn implementations of coordinate descent (in `liblinear`/`LogisticRegression`) and stochastic gradient descent (in `SGDClassifier`) use slightly different parameterizations of the LASSO regularization strength parameter. `liblinear`'s logistic regression solver optimizes the following loss function:

$$\hat{w} = \text{argmin}_{w} \ (C \cdot \ell(X, y; w)) + ||w||_1$$

where $\ell(X, y; w)$ denotes the negative log-likelihood of the observed data $(X, y)$ given a particular choice of feature weights $w$.`SGDClassifier` optimizes the following loss function:

$$\hat{w} = \text{argmin}_{w} \ \ell(X, y; w) + \alpha ||w||_1$$

<!--_ -->

which is equivalent with the exception of the LASSO parameter which is formulated slightly differently, as $\alpha = \frac{1}{C}$.
The result of this slight difference in parameterization is that `liblinear` $C$ values vary inversely with regularization strength (higher values = less regularization, or greater model complexity) and `SGDClassifier` $\alpha$ values vary directly with regularization strength (lower values = less regularization, or greater model complexity).

For the `liblinear` optimizer, we trained models using $C$ values evenly spaced on a logarithmic scale between (10^-3^, 10^7^); i.e. the output of `numpy.logspace(-3, 7, 21)`.
For the SGD optimizer, we trained models using the inverse range of $\alpha$ values between (10^-7^, 10^3^), or `numpy.logspace(-7, 3, 21)`.
These hyperparameter ranges were intended to give evenly distributed coverage across genes that included "underfit" models (predicting only the mean or using very few features, poor performance on all datasets), "overfit" models (performing perfectly on training data but comparatively poorly on cross-validation and test data), and a wide variety of models in between that typically included the best fits to the cross-validation and test data.

For ease of visual comparison in our figures, we plot the SGD $\alpha$ parameter directly, and the `liblinear` $C$ parameter inversely (i.e. $\frac{1}{C}$).
This orients the x-axes of the relevant plots in the same direction: lower values represent lower regularization strength or higher model complexity, and higher values represent higher regularization strength or lower model complexity, for both optimizers.

#### SGD learning rate selection

scikit-learn's `SGDClassifier` provides four built-in approaches to learning rate scheduling: `constant` (a single, constant learning rate), `optimal` (a learning rate with an initial value selected using a heuristic based on the regularization parameter and the data loss, that decreases across epochs), `invscaling` (a learning rate that decreases exponentially by epoch), and `adaptive` (a learning rate that starts at a constant value, which is divided by 5 each time the training loss fails to decrease for 5 straight epochs).
The `optimal` learning rate schedule is used by default.

When we compared these four approaches, we used a constant learning rate of 0.0005, and an initial learning rate of 0.1 for the `adaptive` and `invscaling` schedules.
We also tested a fifth approach that we called "`constant_search`", in which we tested a range of constant learning rates in a grid search on a validation dataset, then evaluated the model on the test data using the best-performing constant learning rate by validation AUPR.
For the grid search, we used the following range of constant learning rates: {0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01}.
Unless otherwise specified, results for SGD in the main paper figures used the `constant_search` approach, which performed the best in our comparison between schedulers.



### Results

#### `liblinear` and SGD LASSO models perform comparably, but `liblinear` is sensitive to regularization strength

For each of the 125 driver genes from the Vogelstein et al. 2013 paper, we trained models to predict mutation status (presence or absence) from RNA-seq data, derived from the TCGA Pan-Cancer Atlas.
For each optimizer, we trained LASSO logistic regression models across a variety of regularization parameters (see Methods for parameter range details), achieving a variety of different levels of model sparsity (Supplementary Figure {@fig:compare_sparsity}).
We repeated model fitting/evaluation across 4 cross-validation splits x 2 replicates (random seeds) for a total of 8 different models per parameter.
Cross-validation splits were stratified by cancer type.

Previous work has shown that pan-cancer classifiers of Ras mutation status are accurate and biologically informative [@doi:10.1016/j.celrep.2018.03.046].
We first evaluated models for KRAS mutation prediction.
As model complexity increases (more nonzero coefficients) for the `liblinear` optimizer, we observed that performance increases then decreases, corresponding to overfitting for high model complexities/numbers of nonzero coefficients (Figure {@fig:optimizer_compare_mutations}A).
On the other hand, for the SGD optimizer, we observed consistent performance as model complexity increases, with models having no nonzero coefficients performing comparably to the best (Figure {@fig:optimizer_compare_mutations}B).
In this case, top performance for SGD (a regularization parameter of 10^-1^) is slightly better than top performance for `liblinear` (a regularization parameter of 1 / 3.16 x 10^2^): we observed a mean test AUPR of 0.722 for SGD vs. mean AUPR of 0.692 for `liblinear`.

To determine how relative performance trends with `liblinear` tend to compare across the genes in the Vogelstein dataset at large, we looked at the difference in performance between optimizers for the best-performing models for each gene (Figure {@fig:optimizer_compare_mutations}C).
The distribution is centered around 0 and more or less symmetrical, suggesting that across the gene set, `liblinear` and SGD tend to perform comparably to one another.
We saw that for 52/84 genes, performance for the best-performing model was better using SGD than `liblinear`, and for the other 32 genes performance was better using `liblinear`.
In order to quantify whether the overfitting tendencies (or lack thereof) also hold across the gene set, we plotted the difference in performance between the best-performing model and the largest (least regularized) model; classifiers with a large difference in performance exhibit strong overfitting, and classifiers with a small difference in performance do not overfit (Figure {@fig:optimizer_compare_mutations}D).
For SGD, the least regularized models tend to perform comparably to the best-performing models, whereas for `liblinear` the distribution is wider suggesting that overfitting is more common.

![
**A.** Performance vs. inverse regularization parameter for KRAS mutation status prediction, using the `liblinear` coordinate descent optimizer. Dotted lines indicate top performance value of the opposite optimizer.
**B.** Performance vs. regularization parameter for KRAS mutation status prediction, using the SGD optimizer. "Holdout" dataset is used for SGD learning rate selection, "test" data is completely held out from model selection and used for evaluation.
**C.** Distribution of performance difference between best-performing model for `liblinear` and SGD optimizers, across all 84 genes in Vogelstein driver gene set. Positive numbers on the x-axis indicate better performance using `liblinear`, and negative numbers indicate better performance using SGD.
**D.** Distribution of performance difference between best-performing model and largest (least regularized) model, for `liblinear` and SGD, across all 84 genes. Smaller numbers on the y-axis indicate less overfitting, and larger numbers indicate more overfitting.
](images/optimizers/figure_1.png){#fig:optimizer_compare_mutations width="100%"}

#### SGD is sensitive to learning rate selection

The SGD results shown in Figure {@fig:optimizer_compare_mutations} select the best-performing learning rate using a grid search on the holdout dataset, independently for each regularization parameter.
We also compared against other learning rate scheduling approaches implemented in scikit-learn (see Methods for implementation details and grid search specifications).
For KRAS mutation prediction, we observed that the choice of initial learning rate and scheduling approach affects performance significantly, and other approaches to selecting the learning rate performed poorly relative to `liblinear` (black dotted lines in Figure {@fig:sgd_lr_schedulers}) and to the grid search.
We did not observe an improvement in performance over `liblinear` or the grid search for learning rate schedulers that decrease across epochs (Figure {@fig:sgd_lr_schedulers}A, C, and D), nor did we see comparable performance when we selected a single constant learning rate for all levels of regularization without the preceding grid search (Figure {@fig:sgd_lr_schedulers}B).
Notably, scikit-learn's default "optimal" learning rate schedule performed relatively poorly for this problem, suggesting that tuning the learning rate and selecting a well-performing scheduler is a critical component of applying SGD successfully for this problem (Figure {@fig:sgd_lr_schedulers}D).
We observed similar trends across all genes in the Vogelstein gene set, with other learning rate scheduling approaches performing poorly in aggregate relative to both `liblinear` and SGD with the learning rate grid search (Supplementary Figure {@fig:compare_all_lr}).

![
**A.** Performance vs. regularization parameter for KRAS mutation prediction, using SGD optimizer with adaptive learning rate scheduler. Dotted line indicates top performance value using `liblinear`, from Figure {@fig:optimizer_compare_mutations}A.
**B.** Performance vs. regularization parameter, using SGD optimizer with constant learning rate scheduler and a learning rate of 0.0005.
**C.** Performance vs. regularization parameter, using SGD optimizer with inverse scaling learning rate scheduler.
**D.** Performance vs. regularization parameter, using SGD optimizer with "optimal" learning rate scheduler.
](images/optimizers/figure_2.png){#fig:sgd_lr_schedulers width="100%"}

#### `liblinear` and SGD result in different models, with varying loss dynamics

We sought to determine whether there was a difference in the sparsity of the models resulting from the different optimization schemes.
In general across all genes, the best-performing SGD models mostly tend to have many nonzero coefficients, but with a distinct positive tail, sometimes having few nonzero coefficients.
By contrast, the `liblinear` models are generally sparser with fewer than 2500 nonzero coefficients, out of ~16100 total input features, and a much narrower tail (Figure {@fig:optimizer_coefs}A).
The sum of the coefficient magnitudes, however, tends to be smaller on average across all levels of regularization for SGD than for `liblinear` (Figure {@fig:optimizer_coefs}B).
This effect is less pronounced for the other learning rate schedules shown in Figure {@fig:sgd_lr_schedulers}, with the other options resulting in larger coefficient magnitudes (Supplementary Figure {@fig:coef_weights_lr}).
These results suggest that the models fit by `liblinear` and SGD navigate the tradeoff between bias and variance in slightly different ways: `liblinear` tends to produce sparser models (more zero coefficients) as regularization increases, but if the learning rate is properly tuned, SGD coefficients tend to have smaller overall magnitudes as regularization increases.


We also compared the components of the loss function across different levels of regularization between optimizers.
The LASSO logistic regression loss function can be broken down into a data-dependent component (the log-loss) and a parameter magnitude dependent component (the L1 penalty), which are added to get the total loss that is minimized by each optimizer; see Methods for additional details.
As regularization strength decreases for `liblinear`, the data loss collapses to near 0, and the L1 penalty dominates the overall loss (Figure {@fig:optimizer_coefs}C).
For SGD, on the other hand, the data loss decreases slightly as regularization strength decreases but remains relatively high (Figure {@fig:optimizer_coefs}D).
Other SGD learning rate schedules have similar loss curves to the `liblinear` results, although this does not result in improved classification performance (Supplementary Figure {@fig:loss_lr}).

![
**A.** Distribution across genes of the number of nonzero coefficients included in best-performing LASSO logistic regression models. Violin plot density estimations are clipped at the ends of the observed data range, and boxes show the median/IQR.
**B.** Distribution across genes of the sum of model coefficient weights for best-performing LASSO logistic regression models.
**C.** Decomposition of loss function for models fit using `liblinear` across regularization levels. 0 values on the y-axis are rounded up to machine epsilon; i.e. 2.22 x 10^-16^.
**D.** Decomposition of loss function for models fit using SGD across regularization levels. 0 values on the y-axis are rounded up to machine epsilon; i.e. 2.22 x 10^-16^.
](images/optimizers/figure_3.png){#fig:optimizer_coefs width="100%"}


### Discussion

Our work shows that optimizer choice presents tradeoffs in model selection for cancer transcriptomics.
We observed that LASSO logistic regression models for mutation status prediction fit using stochastic gradient descent were highly sensitive to learning rate tuning, but they tended to perform robustly across diverse levels of regularization and sparsity.
Coordinate descent implemented in `liblinear` did not require learning rate tuning, but generally performed best for a narrow range of fairly sparse models, overfitting as regularization strength decreased.
Tuning of regularization strength for `liblinear`, and learning rate (and regularization strength to a lesser degree) for SGD, are critical steps which must be considered as part of analysis pipelines.
The sensitivity we observed to these details highlights the importance of reporting exactly what optimizer was used, and how the relevant hyperparameters were selected, in studies that use machine learning models for transcriptomic data.

To our knowledge, the phenomenon we observed with SGD has not been documented in other applications of machine learning to genomic or transcriptomic data.
In recent years, however, the broader machine learning research community has identified and characterized implicit regularization for SGD in many settings, including overparametrized or feature-rich problems as is often the case in transcriptomics [@arxiv:2108.04552; @arxiv:2003.06152; @url:http://proceedings.mlr.press/v134/zou21a.html].
The resistance we observed of SGD-optimized models to decreased performance on held-out data as model complexity increases is often termed "benign overfitting": overfit models, in the sense that they fit the training data perfectly and perform worse on the test data, can still outperform models that do not fit the training data as well or that have stronger explicit regularization.
Benign overfitting has been attributed to optimization using SGD [@url:http://proceedings.mlr.press/v134/zou21a.html; @doi:10.1145/3446776], and similar patterns have been observed for both linear models and deep neural networks [@doi:10.1073/pnas.1907378117; @arxiv:1611.03530].

Existing gene expression prediction benchmarks and pipelines typically use a single model implementation, and thus a single optimizer.
We recommend thinking critically about optimizer choice, but this can be challenging for researchers that are inexperienced with machine learning or unfamiliar with how certain models are fit under the hood.
For example, R's `glmnet` package uses a cyclical coordinate descent algorithm to fit logistic regression models [@doi:10.18637/jss.v033.i01], which would presumably behave similarly to `liblinear`, but this is somewhat opaque in the `glmnet` documentation itself.
Increased transparency and documentation in popular machine learning packages with respect to optimization, especially for models that are difficult to fit or sensitive to hyperparameter settings, would benefit new and unfamiliar users.


Related to what we see in our SGD-optimized models, there exist other problems in gene expression analysis where using all available features is comparable to, or better than, using a subset.
For example, using the full gene set improves correlations between preclinical cancer models and their tissue of origin, as compared to selecting genes based on variability or tissue-specificity [@doi:10.1101/2023.04.11.536431].
On the other hand, when predicting cell line viability from gene expression profiles, selecting features by Pearson correlation improves performance over using all features, similar to our `liblinear` classifiers [@doi:10.1101/2020.02.21.959627].
In future work, it could be useful to explore if the coefficients found by `liblinear` and SGD emphasize the same pathways or functional gene sets, or if there are patterns to which mutation status classifiers (or other cancer transcriptomics classifiers) perform better with more/fewer nonzero coefficients.


### Data and code availability

The data analyzed during this study were previously published as part of the TCGA Pan-Cancer Atlas project [@doi:10.1038/ng.2764], and are available from the NIH NCI Genomic Data Commons (GDC).
The scripts used to download and preprocess the datasets for this study are available at <https://github.com/greenelab/pancancer-evaluation/tree/master/00_process_data>, and the code used to carry out the analyses in this study is available at <https://github.com/greenelab/pancancer-evaluation/tree/master/01_stratified_classification>, both under the open-source BSD 3-clause license.
Equivalent versions of Figure {@fig:optimizer_compare_mutations}A and {@fig:optimizer_compare_mutations}B for all 84 genes in the Vogelstein et al. 2013 gene set are available on Figshare at <https://doi.org/10.6084/m9.figshare.22728644>, under a CC0 license.
This manuscript was written using Manubot [@doi:10.1371/journal.pcbi.1007128] and is available on GitHub at <https://github.com/greenelab/optimizer-manuscript> under the CC0-1.0 license.
This research was supported in part by the University of Pittsburgh Center for Research Computing through the resources provided. Specifically, this work used the HTC cluster, which is supported by NIH award number S10OD028483.


### Supplementary Material

![Number of nonzero coefficients (model sparsity) across varying regularization parameter settings for KRAS mutation prediction using SGD and `liblinear` optimizers.](images/optimizers/supp_figure_1.png){#fig:compare_sparsity width="100%"}

![Distribution of performance difference between best-performing model for `liblinear` and SGD optimizers, across all 84 genes in Vogelstein driver gene set, for varying SGD learning rate schedulers. Positive numbers on the x-axis indicate better performance using `liblinear`, and negative numbers indicate better performance using SGD.](images/optimizers/supp_figure_2.png){#fig:compare_all_lr width="100%" .page_break_before}

![Sum of absolute value of coefficients + 1 for KRAS mutation prediction using SGD and `liblinear` optimizers, with varying learning rate schedules for SGD. Similar to the figures in the main paper, the `liblinear` x-axis represents the inverse of the $C$ regularization parameter; SGD x-axes represent the untransformed $\alpha$ parameter.](images/optimizers/supp_figure_3.png){#fig:coef_weights_lr width="100%" .page_break_before}

![Decomposition of loss function into data loss and L1 penalty components for KRAS mutation prediction using SGD optimizer, across regularization levels, using varying learning rate schedulers. 0 values on the y-axis are rounded up to machine epsilon, i.e. 2.22 x 10^-16^.](images/optimizers/supp_figure_4.png){#fig:loss_lr width="100%" .page_break_before}




## Chapter 3: Widespread redundancy in -omics profiles of cancer mutation states

This chapter has been published in _Genome Biology_ (https://doi.org/10.1186/s13059-022-02705-y) as "Widespread redundancy in -omics profiles of cancer mutation states".

**Contributions:**
JC: conceptualization, methodology, software, visualization, writing - original draft, writing - review and editing
BCC: methodology, writing - review and editing
MC: methodology, writing - review and editing
CSG: conceptualization, funding acquisition, methodology, supervision, writing - review and editing


### Abstract

#### Background

In studies of cellular function in cancer, researchers are increasingly able to choose from many -omics assays as functional readouts.
Choosing the correct readout for a given study can be difficult, and which layer of cellular function is most suitable to capture the relevant signal remains unclear.

#### Results

We consider prediction of cancer mutation status (presence or absence) from functional -omics data as a representative problem that presents an opportunity to quantify and compare the ability of different -omics readouts to capture signals of dysregulation in cancer.
From the TCGA Pan-Cancer Atlas that contains genetic alteration data, we focus on RNA sequencing, DNA methylation arrays, reverse phase protein arrays (RPPA), microRNA, and somatic mutational signatures as -omics readouts.
Across a collection of genes recurrently mutated in cancer, RNA sequencing tends to be the most effective predictor of mutation state.
We find that one or more other data types for many of the genes are approximately equally effective predictors.
Performance is more variable between mutations than that between data types for the same mutation, and there is little difference between the top data types.
We also find that combining data types into a single multi-omics model provides little or no improvement in predictive ability over the best individual data type.

#### Conclusions

Based on our results, for the design of studies focused on the functional outcomes of cancer mutations, there are often multiple -omics types that can serve as effective readouts, although gene expression seems to be a reasonable default option.


### Background

Although cancer can be initiated and driven by many different genetic alterations, these tend to converge on a limited number of pathways or signaling processes [@doi:10.1016/j.cell.2018.03.035].
As driver mutation status alone confers limited prognostic information, a comprehensive understanding of how diverse genetic alterations perturb central pathways is vital to precision medicine and biomarker identification efforts [@doi:10.7554/eLife.39217; @doi:10.1038/ncomms12096].
While many methods exist to distinguish driver mutations from passenger mutations based on genomic sequence characteristics [@doi:10.1073/pnas.1616440113; @doi:10.1038/s41467-019-11284-9; @doi:10.1371/journal.pcbi.1006658], until recently it has been a challenge to connect driver mutations to downstream changes in gene expression and cellular function within individual tumor samples.

The Cancer Genome Atlas (TCGA) Pan-Cancer Atlas provides uniformly processed, multi-platform -omics measurements across tens of thousands of samples from 33 cancer types [@doi:10.1038/ng.2764].
Enabled by this publicly available data, a growing body of work on linking the presence of driving genetic alterations in cancer to downstream gene expression changes has emerged.
Recent studies have considered Ras pathway alteration status in colorectal cancer [@doi:10.1158/1078-0432.CCR-13-1943], alteration status across many cancer types in Ras genes [@doi:10.1016/j.celrep.2018.03.046; @doi:10.1093/bib/bbaa258], _TP53_ [@doi:10.1016/j.celrep.2018.03.076], and _PIK3CA_ [@doi:10.1371/journal.pone.0241514], and alteration status across cancer types in frequently mutated genes [@doi:10.1186/s13059-020-02021-3].
More broadly, other groups have drawn on similar ideas to distinguish between the functional effects of different alterations in the same driver gene [@doi:10.1101/2020.06.02.128850], to link alterations with similar gene expression signatures within cancer types [@doi:10.1142/9789811215636_0031], and to identify trans-acting expression quantitative trait loci (trans-eQTLs) in germline genetic studies [@doi:10.1101/2020.05.07.083386].

These studies share a common thread: they each combine genomic (point mutation and copy number variation) data with transcriptomic (RNA sequencing) data within samples to interrogate the functional effects of genetic variation.
RNA sequencing is ubiquitous and cheap, and its experimental and computational methods are relatively mature, making it a vital tool for generating insight into cancer pathology [@doi:10.1038/nrg.2017.96].
Some driver mutations, however, are known to act indirectly on gene expression through varying mechanisms.
For example, oncogenic _IDH1_ and _IDH2_ mutations in glioma have been shown to interfere with histone demethylation, which results in increased DNA methylation and blocked cell differentiation [@doi:10.1016/j.ccr.2010.03.017; @doi:10.1093/jnci/djq497; @doi:10.1056/NEJMoa0808710; @doi:10.1038/nature10860].
Other genes implicated in aberrant DNA methylation in cancer include the TET family of genes [@doi:10.1016/j.tig.2014.07.005] and _SETD2_ [@doi:10.1101/cshperspect.a026468].
Certain driver mutations, such as those in DNA damage repair genes, may lead to detectable patterns of somatic mutation [@doi:10.1038/nrg3729].
Additionally, correlation between gene expression and protein abundance in cancer cell lines is limited, and proteomics data could correspond more directly to certain cancer phenotypes and pathway perturbations [@doi:10.1016/j.cell.2019.12.023].
In these contexts and others, integrating different data modalities or combining multiple data modalities could be more effective than relying solely on gene expression as a functional signature.

Here, we compare -omics data types profiled in the TCGA Pan-Cancer Atlas to evaluate use as a multivariate functional readout of genetic alterations in cancer.
We focus on gene expression (RNA sequencing data), DNA methylation (27K and 450K probe chips), reverse phase protein array (RPPA), microRNA expression, and mutational signatures data [@doi:10.1038/s41586-020-1943-3] as possible readouts.
Prior studies have identified univariate correlations of CpG site methylation [@doi:10.1371/journal.pcbi.1005840; @doi:10.1186/s12920-018-0425-z] and correlations of RPPA protein profiles [@doi:10.1186/s13073-018-0591-9] with the presence or absence of certain driver mutations.
Other relevant past work includes linking point mutations and copy number variants (CNVs) with changes in methylation and expression at individual genes [@doi:10.1093/bioinformatics/btr019; @doi:10.1093/bib/bbw037] and identifying functional modules that are perturbed by somatic mutations [@doi:10.1093/bioinformatics/btq182; @doi:10.1038/ncomms9554].
However, direct comparison among different data types for this application is lacking, particularly in the multivariate case where we consider changes to -omics-derived gene signatures rather than individual genes in isolation.

We select a collection of potential cancer drivers with varying functions and roles in cancer development.
We use mutation status in these genes as labels to train classifiers, using each of the data types listed as training data, in a pan-cancer setting; we follow similar methods to the elastic net logistic regression approach described in Way et al. 2018 [@doi:10.1016/j.celrep.2018.03.046] and Way et al. 2020 [@doi:10.1186/s13059-020-02021-3].
We show that there is considerable predictive signal for many genes relative to a cancer-type corrected baseline and that gene expression tends to provide good predictions of mutation state across most genes.
Surprisingly, we find that for a variety of genes, multiple data types are approximately equally effective predictors.
We observe similar results for pan-cancer survival prediction across the same data types with little separation between the top-performing data types.
In addition, we observe that combining data types into a single multi-omics model for mutation prediction provides little, if any, performance benefit over the most performant model using a single data type.
Our results will help to inform the design of future functional genomics studies in cancer, suggesting that for many strong drivers with clear functional signatures, different -omics measurements can provide similar information content.


### Methods

#### Mutation data download and preprocessing

To generate binary mutated/non-mutated gene labels for our machine learning model, we used mutation calls for TCGA samples from MC3 [@doi:10.1016/j.cels.2018.03.002] and copy number threshold calls from GISTIC2.0 [@doi:10.1186/gb-2011-12-4-r41].
MC3 mutation calls were downloaded from the Genomic Data Commons (GDC) of the National Cancer Institute, at [`https://gdc.cancer.gov/about-data/publications/pancanatlas`](https://gdc.cancer.gov/about-data/publications/pancanatlas).
Copy number threshold calls are from an older version of the GDC data, and are available here: [`https://figshare.com/articles/dataset/TCGA_PanCanAtlas_Copy_Number_Data/6144122`](https://figshare.com/articles/dataset/TCGA_PanCanAtlas_Copy_Number_Data/6144122).
We removed hypermutated samples (defined as five or more standard deviations above the mean non-silent somatic mutation count) from our dataset to reduce the number of false positives (i.e., non-driver mutations).
After this filtering, 9,074 TCGA samples with mutation and copy number data remained.
Any sample with a non-silent somatic variant in the target gene was included in the positive set.
We also included copy number gains in the target gene for oncogenes and copy number losses in the target gene for tumor suppressor genes in the positive set; all remaining samples were considered negative for mutation in the target gene.

#### Omics data download and preprocessing

RNA sequencing, 27K and 450K methylation array, microRNA, and RPPA datasets for TCGA samples were all downloaded from GDC, at the same link provided above.
Mutational signatures information for TCGA samples with whole-exome sequencing data was downloaded from the International Cancer Genome Consortium (ICGC) data portal, at [`https://dcc.icgc.org/releases/PCAWG/mutational_signatures/Signatures_in_Samples/SP_Signatures_in_Samples`](https://dcc.icgc.org/releases/PCAWG/mutational_signatures/Signatures_in_Samples/SP_Signatures_in_Samples).
For our experiments, we used only the "single base signature" (SBS) mutational signatures, generated in [@doi:10.1038/s41586-020-1943-3].
In general, before training classifiers or extracting PCA components from all of the data types, we standardized (took z-scores of) each column/feature of all data types.
For the RNA sequencing dataset, we generally used only the top 8,000 gene features by mean absolute deviation as predictors in our single-omics models, except where specified otherwise.
For the RPPA, microRNA, and mutational signatures datasets, all columns/features were used.

To remove missing values from the methylation datasets, we removed the 10 samples with the most missing values, then performed mean imputation for probes with 1 or 2 values missing.
All probes with missing values remaining after sample filtering and imputation were dropped from the analysis.
This left us with 20,040 CpG probes in the 27K methylation dataset and 370,961 CpG probes in the 450K methylation dataset.
For experiments where "raw" methylation data was used, we used the top 100,000 probes in the 450K dataset by mean absolute deviation for computational efficiency, and we used all of the 20,040 probes in the 27K dataset.
For experiments where "compressed" methylation data was used, we used principal component analysis (PCA), as implemented in the `scikit-learn` Python library [@url:https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html], to extract the top 5,000 principal components from the methylation datasets.
We initially applied the beta-mixture quantile normalization (BMIQ) method [@doi:10.1093/bioinformatics/bts680] to correct for variability in signal intensity between type I and type II probes, but we observed that this had no effect on our results.
We report uncorrected results in the main paper for simplicity.

#### Construction of a set of cancer genes

To get a comprehensive picture of classification performance across a wide variety of cancer-related genes, we integrated several curated gene sets from the literature into a single "merged" cancer gene set.
The individual gene sets we integrated were from Vogelstein et al. [@doi:10.1126/science.1235122] (all genes from Table S2A), Bailey et al. [@doi:10.1016/j.cell.2018.02.060] (only genes annotated as "pan-cancer" drivers in Table S1), and the COSMIC Cancer Gene Census [@doi:10.1038/s41568-018-0060-1] (all Tier 1 genes annotated as "somatic").
In addition, the COSMIC CGC dataset contains 3 possible "roles in cancer" for each gene: oncogene, TSG, and fusion gene; for this analysis we dropped genes that are annotated only as fusion genes (i.e. no oncogene or TSG annotation).
These filters resulted in a starting dataset of 511 cancer-related genes, which we reduced further for each experiment based on the number of mutated (i.e. positively labeled) samples as described in the next section.

#### Comparing data modalities

We made three main comparisons in this study: one between different sets of genes using only expression data, one comparing expression and DNA methylation data types, and one comparing all data types.
This choice in comparisons was mainly due to sample size limitations, as running a single comparison using all data types would force us to use only samples that are profiled for every data type, which would discard a large number of samples that lack profiling on only one or a few data types.
Thus, for each of the three comparisons, we used the intersection of TCGA samples having measurements for all of the datasets being compared in that experiment.
This resulted in three distinct sets of samples: 9,074 samples shared between {expression, mutation} data, 7,981 samples shared between {expression, mutation, 27K methylation, 450K methylation}, and 5,226 samples shared between {expression, mutation, 27K methylation, 450K methylation, RPPA, microRNA, mutational signatures}.
When we dropped samples between experiments as progressively more data types were added, we observed that the dropped samples had approximately the same cancer type proportions as the dataset as a whole.
In other words, samples that were profiled for one data type but not another did not tend to come exclusively from one or a few cancer types.
Exceptions included acute myeloid leukemia (LAML) which had no samples profiled in the RPPA data, and ovarian cancer (OV) which had only 8 samples with 450K methylation data.
More detailed information on cancer type proportions profiled for each data type is provided in Additional File 1: Fig. S1 and Additional File 2.

For each target gene, in order to ensure that the training dataset was reasonably balanced (i.e. that there would be enough mutated samples to train an effective classifier), we included only cancer types with at least 15 mutated samples and at least 5% mutated samples, which we refer to here as "valid" cancer types.
After applying these filters, the number of valid cancer types remaining for each gene varied based on the set of samples used: more data types resulted in fewer shared samples, and fewer samples generally meant fewer valid cancer types.
In some cases, this resulted in genes with no valid cancer types, which we dropped from the analysis.
Out of the 511 genes from the "merged" cancer gene set described in the previous section, for the analysis using {expression, mutation} data we retained 268 target genes, for the {expression, mutation, 27k methylation, 450k methylation} analysis we retained 272 genes, and for the analysis using all data types we retained 217 genes.

We additionally explored mutation prediction from gene expression alone using three gene sets of equal size: the cancer-related genes from the merged dataset described previously, a set of frequently mutated genes in TCGA, and a set of random genes with mutations profiled by MC3.
To match the size of the merged cancer gene set, we took the 268 most frequently mutated genes in TCGA as quantified by MC3, all of which had at least one valid cancer type.
For the random gene set, we first filtered to the set of all genes with one or more valid cancer types by the same criteria (15 total samples mutated and at least 5% of samples mutated), then sampled 268 of the resulting 1,348 genes uniformly at random.
Based on the results of the gene expression experiments, we used the merged cancer-related gene set for all subsequent experiments comparing -omics data types.

#### Training classifiers to detect cancer mutations

We trained logistic regression classifiers to predict whether or not a given sample had a mutational event in a given target gene using data from various -omics datasets as explanatory variables.
Our model was trained on -omics data ($X$) to predict mutation presence or absence ($y$) in a target gene.
To control for varying mutation burden per sample and to adjust for potential cancer type-specific expression patterns, we included one-hot encoded cancer type and log~10~(sample mutation count) in the model as covariates.
Since our -omics datasets tend to have many dimensions and comparatively few samples, we used an elastic net penalty to prevent overfitting [@doi:10.1111/j.1467-9868.2005.00503.x] in line with the approach used in Way et al. 2018 [@doi:10.1016/j.celrep.2018.03.046] and Way et al. 2020 [@doi:10.1186/s13059-020-02021-3].
Elastic net logistic regression finds the feature weights $\hat{w} \in \mathbb{R}^{p}$ solving the following optimization problem:

$$\hat{w} = \text{argmin}_{w} \ \ell(X, y; w) + \alpha \lambda||w||_1 + \frac{1}{2}\alpha (1 - \lambda) ||w||_2$$

where $i \in \{1, \dots, n\}$ denotes a sample in the dataset, $X_i \in \mathbb{R}^{p}$ denotes features (omics measurements) from the given sample, $y_i \in \{0, 1\}$ denotes the label (mutation presence/absence) for the given sample, and $\ell(\cdot)$ denotes the negative log-likelihood of the observed data given a particular choice of feature weights, i.e.

$$\ell(X, y; w) = -\sum_{i=1}^{n} y_i \log\left(\frac{1}{1 + e^{-w^{\top}X_i}}\right) + (1 - y_i) \log\left(1 - \frac{1}{1 + e^{-w^{\top}X_i}}\right)$$

This optimization problem leaves two hyperparameters to select: $\alpha$ (controlling the tradeoff between the data log-likelihood and the penalty on large feature weight values), and $\lambda$ (controlling the tradeoff between the L1 penalty and L2 penalty on the weight values).
Although the elastic net optimization problem does not have a closed form solution, the loss function is convex, and iterative optimization algorithms are commonly used for finding reasonable solutions.
For fixed values of $\alpha$ and $\lambda$, we solved for $\hat{w}$ using stochastic gradient descent, as implemented in `scikit-learn`'s `SGDClassifier` method.

Given weight values $\hat{w}$, it is straightforward to predict the probability of a positive label (mutation in the target gene) $P(y^{*} = 1 \mid X^{*}; \hat{w})$ for a test sample $X^{*}$:

$$P(y^{*} = 1 \mid X^{*}; \hat{w}) = \frac{1}{1 + e^{-\hat{w}^{\top}X^{*}}}$$

and the probability of no mutation in the target gene, $P(y^{*} = 0 \mid X^{*}; \hat{w})$, is given by (1 - the above quantity).

For each target gene, we evaluated model performance using two replicates of 4-fold cross-validation, where train and test splits were stratified by cancer type and sample type.
That is, each training set/test set combination had equal proportions of each cancer type (BRCA, SKCM, COAD, etc.) and each sample type (primary tumor, recurrent tumor, etc.).
To choose the elastic net hyperparameters, we used 3-fold nested cross-validation, with a grid search over the following hyperparameter ranges: $\lambda$ = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] and $\alpha$ = [0.0001, 0.001, 0.01, 0.1, 1, 10].
Using the grid search results, for each evaluation fold we selected the set of hyperparameters with the optimal area under the precision-recall curve (AUPR), averaged over the three inner folds.

#### Evaluating mutation prediction classifiers

Area under the receiver-operator curve (AUROC) [@doi:10.1016/j.patrec.2005.10.010] and area under the precision-recall curve (AUPR) [@doi:10.1145/65943.65945] are metrics that are frequently used to quantify classification performance for a continuous or probabilistic output, such as that provided by logistic regression.
These metrics summarize performance across a variety of binary label thresholds, rather than requiring choice of a single threshold to determine positive or negative predictions.
In the main text, we report results using AUPR, summarized using average precision.
AUPR has been shown to distinguish between models more accurately than AUROC when there are few positively labeled samples [@doi:10.1371/journal.pone.0118432; @arxiv:2006.11278].
As an additional correction for imbalanced labels, in many of the results in the main text we report the difference in AUPR between a classifier fit to true mutation labels and a classifier fit to data where the mutation labels are randomly permuted.
In cases where mutation labels are highly imbalanced (very few mutated samples and many non-mutated samples), a classifier with permuted labels may perform well simply by chance, e.g. by predicting the negative/non-mutated class for most samples.
To maintain the same label balance for the classifiers with permuted labels as the classifiers with the true labels, we permuted labels separately in the train and test sets for each cross-validation split.
Additionally, to maintain the same label proportions within each cancer type after permuting the labels, we permuted labels independently for each cancer type.

Recall that for each target gene and each -omics dataset, we ran two replicates of 4-fold cross-validation, for a total of eight performance results.
To make a statistical comparison between two models using these performance distributions, we used paired-sample _t_-tests, where performance measurements derived from the same cross-validation fold are considered paired measurements.
We used this approach to compare a model trained on true labels with a model trained on permuted labels (addressing the question, "for the given gene using the given data type, can we predict mutation status better than random"), and to compare a model trained on data type A with a model trained on data type B (addressing the question, "for the given gene, can we make more effective mutation status predictions using data type A or data type B").

We corrected for multiple tests using a Benjamini-Hochberg false discovery rate correction.
For experiments where we chose a binary threshold for accepting/rejecting $H_0$ we set a conservative corrected threshold of $p = 0.001$; we were able to estimate the number of false positives by examining genes with better performance for permuted mutation labels than true labels.
We chose this threshold to ensure that none of the observed false positive genes were considered significant, since we would never expect permuting labels to improve performance.
However, our results were not sensitive to the choice of this threshold, and we display cutoffs of $p = 0.05$ and $p = 0.01$ in many of our plots as well.

#### Survival prediction using -omics datasets

As a complementary comparison to mutation prediction, we constructed predictors of patient survival using the clinical data available from the GDC, in the `TCGA-CDR-SupplementalTableS1.xlsx` file.
Following the methods described in [@doi:10.1101/2021.06.01.446243], as the clinical endpoint we used overall survival (OS), except in nine cancer types with few deaths observed where we used progression-free intervals (PFI) as the clinical endpoint (BRCA, DLBC, LGG, PCPG, PRAD, READ, TGCT, THCA and THYM).
For prediction, we used Cox regression as implemented in the `scikit-survival` Python package [@url:https://jmlr.org/papers/v21/20-729.html], with patient age at diagnosis and log~10~(sample mutation count) included as covariates, as well as a one-hot encoded variable for cancer type in the pan-cancer case.
To ensure that the per-feature information content was comparable between -omics data types, we preprocessed the -omics datasets using PCA and extracted the top $k$ principal components; in the case where the number of features in the original dataset was less than $k$ we used all available PCs (that is, we set $k = \min(p, k)$ where $p$ is the number of features in the unprocessed dataset).
For the pan-cancer models we plot results over multiple values of $k$: $k \in \{10, 100, 500, 1000, 5000\}$; for the individual cancer type models we set $k = 10$.

To model pan-cancer survival (results shown in main paper), we used the elastic net Cox regression implementation in `scikit-survival` (i.e. the `CoxnetSurvivalAnalysis` method).
To select hyperparameters for the elastic net Cox regression model, we performed a grid search over $\lambda$ = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] and $\alpha$ = [0, 1e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000].
To select the regularization parameter $\alpha$, we used the default selection procedure implemented in `scikit-survival` to determine a range of potential $\alpha$ values based on the data.
This procedure begins by deriving the maximum $\alpha$ value as the smallest value for which all coefficients are 0 (call this $\alpha_{\text{max}}$), then it selects 100 possibilities for alpha spaced evenly on a log scale between $\alpha_{\text{max}}$ and $0.01 \cdot \alpha_{\text{max}}$.
We found that for individual cancer types, this data-driven procedure resulted in more consistent and stable model convergence than choosing a fixed set of alphas to search over as in the pan-cancer survival prediction experiments.

We measured survival prediction performance using the censored concordance index (c-index) [@pubmed:8668867], which quantifies agreement between the order of survival time predictions and true outcomes for a held-out dataset; higher c-index values indicate more accurate survival prediction performance.
Similar to the mutation prediction experiments, we calculated c-index values on held-out subsets of the data for two replicates of 4-fold cross-validation, resulting in eight performance measurements for each model.
As a baseline, for both the pan-cancer and cancer type specific datasets, we constructed survival models using only non-omics covariates.
For the pan-cancer data, covariates included patient age at diagnosis, log~10~(sample mutation count), and a one-hot encoded variable for sample cancer type.
The cancer type-specific baseline models were the same, without the cancer type indicator, since all train and test samples were derived from the same cancer type.

#### Multi-omics mutation prediction experiments

To predict mutation presence or absence in cancer genes using multiple data types simultaneously, we concatenated individual datasets into a large feature matrix, then used the same elastic net logistic regression method described previously.
For this task, we considered only the gene expression, 27K methylation, and 450K methylation datasets.
We used only these data types to limit the number of multi-omics combinations; the expression and methylation datasets resulted in the best overall performance across the single-omics experiments, so we limited combinations to those datasets.
In the main text, we report results using the top 5,000 principal components for each dataset , which ensures that most variance is captured (approximately 95-98% of variance for each data type).
In Additional File 1: Fig. S6, we also report results using "raw" features: for gene expression we used all 15,639 genes available in our RNA sequencing dataset, and for the 27K and 450K methylation datasets we used the top 20,000 CpG probes by mean absolute deviation.

To construct the multi-omics models, we considered each of the pairwise combinations of the datasets listed above, as well as a combination of all 3 datasets.
When combining multiple datasets, we concatenated along the column axis and included covariates for cancer type and sample mutation burden as before.
For all multi-omics experiments, we used only the samples from TCGA with data for all three data types (i.e. the same 7,981 samples used in the single-omics experiments comparing expression and methylation data types).
We considered a limited subset of well-performing genes from the merged cancer gene set as target genes, including _EGFR_, _IDH1_, _KRAS_, _PIK3CA_, _SETD2_, and _TP53_.
We selected these genes because we had previously observed that they have good predictive performance and because they represent a combination of alterations that have strong gene expression signatures (_KRAS_, _EGFR_, _IDH1_, _TP53_) and strong DNA methylation signatures (_IDH1_, _SETD2_, _TP53_).

For the experiments predicting mutation status using a 3-layer fully connected neural network, described in the Results section and Additional File 1: Fig. S7, we used the top 5,000 principal components as input for each data type.
We selected hyperparameters for each of the 8 outer cross-validation splits using a single inner train/validation split and a random search over 20 hyperparameter combinations.
The hyperparameter ranges that we sampled from in the random search were: `learning_rate: [0.1, 0.01, 0.001, 5e-4, 1e-4], h1_size: [1000, 500, 250], dropout: [0.5, 0.75, 0.9], weight_decay: [0, 0.1, 1, 100]`.
Here, `h1_size` refers to the size of the first hidden layer, and the size of the second hidden layer was always set to `h1_size / 2`.
As in the elastic net grid search, we chose the combination of hyperparameters with the best AUPR on the validation set, and retrained the model with those hyperparameters to make predictions on the test set.
We trained our networks with the Adam optimizer [@arxiv:1412.6980], with ReLU activation after the hidden layers and sigmoid activation to make predictions, and using binary cross-entropy as the loss function as implemented in the PyTorch `BCEWithLogitsLoss` function, through the `skorch` library which provides interoperability between PyTorch and scikit-learn.



### Results

#### Using diverse data modalities to predict cancer alterations

We collected five different data modalities from cancer samples in the TCGA Pan-Cancer Atlas, capturing five steps of cellular function that are perturbed by genetic alterations in cancer (Figure {@fig:omics_overview}A).
These included gene expression (RNA-seq data), DNA methylation (27K and 450K Illumina BeadChip arrays), protein abundance (RPPA data), microRNA expression data, and patterns of somatic mutation (mutational signatures).
To link these diverse data modalities to changes in mutation status, we used elastic net logistic regression to predict the presence or absence of mutations in cancer genes, using these readouts as predictive features (Figure {@fig:omics_overview}B).
We evaluated the resulting mutation status classifiers in a pan-cancer setting, preserving the proportions of each of the 33 cancer types in TCGA for eight train/test splits (4 folds x 2 replicates) in each of approximately 250 cancer genes (Figure {@fig:omics_overview}C).

We sought to compare classifiers against a baseline where mutation labels are permuted (to identify genes whose mutation status correlates strongly with a functional signature in a given data type) and also to compare classifiers trained on true labels across different data types (to identify data types that are more or less predictive of mutations in a given gene).
To account for variation between dataset splits in making these comparisons, we treat classification metrics from the eight train/test splits as performance distributions, which we compare using _t_-tests.
We summarize performance across all genes in our cancer gene set using a similar approach to a volcano plot, in which each point is a gene.
In our summary plots, the x-axis shows the magnitude of the change in the classification metric between conditions, and the y-axis shows the _p_-value for the associated _t_-test (Figure {@fig:omics_overview}C).

![
**A.** Cancer mutations can perturb cellular function via a variety of cellular processes.
Arrows represent major potential paths of information flow from a somatic mutation in DNA to its resulting cell phenotype; circular arrow represents the ability of certain mutations (e.g. in DNA damage repair genes) to alter somatic mutation patterns.
Note that this does not reflect all possible relationships between cellular processes: for instance, changes in gene expression can lead to changes in somatic mutation rates.
**B.** Predicting presence/absence of somatic alterations in cancer from diverse data modalities.
In this study, we use functional readouts from TCGA as predictive features and the presence or absence of mutation in a given gene as labels.
This reverses the primary direction of information flow shown in Panel A.
**C.** Schematic of evaluation pipeline.
](images/omics/figure_1.png){#fig:omics_overview}

#### Selection of cancer-related genes improves predictive signal

As a baseline, we evaluated prediction of mutation status from gene expression data across several different gene sets.
Past work has evaluated mutation prediction for the top 50 most mutated genes in TCGA [@doi:10.1186/s13059-020-02021-3], and we sought to extend this to a broader list of gene sets.
To evaluate whether using known cancer-related genes tends to improve prediction, we compiled a set of cancer-related genes (n=268) from Vogelstein et al. 2013 [@doi:10.1126/science.1235122], Bailey et al. 2018 [@doi:10.1016/j.cell.2018.02.060], and the COSMIC Cancer Gene Census [@doi:10.1038/s41568-018-0060-1].
We compared performance on this curated gene set with performance on an equal number of genes sampled randomly after applying a mutation frequency threshold (n=268, see Methods for sampling details) and an equal number of the most mutated genes in TCGA (n=268).
For all gene sets, we used only the set of TCGA samples for which both gene expression and somatic mutation data exists, resulting in a total of 9,074 samples across all 33 cancer types.
This set of samples was further filtered for each target gene to cancer types containing at least 15 mutated samples and at least 5% of samples mutated for that cancer type.
As an alternate approach, we tried including/excluding entire genes using similar filters, and the results were consistent across filtering strategies (Additional File 1: Fig. S4).
We then evaluated the performance for each target gene in each of the three gene sets.

Overall, genes from the cancer-related gene set were more predictable than randomly chosen genes or those selected by total mutation count (Figure {@fig:expression_gene_sets}A).
In total, for a significance threshold of $\alpha = 0.001$, 120/268 genes (44.8%) in the cancer-related gene set are significantly predictable from gene expression data, compared to 14/268 genes (5.22%) in the random gene set and 80/268 genes (29.9%) in the most mutated gene set.
Of the 14 significantly predictable genes in the random gene set, 13 of them are also in the cancer-related gene set (highlighted with 'X' in Figure {@fig:expression_gene_sets}B), and of the 80 significantly predictable genes in the most mutated gene set, 26 of them are also in the cancer-related gene set (highlighted in red in Figure {@fig:expression_gene_sets}C).
These results suggest that selecting target genes for mutation prediction based on prior knowledge of their involvement in cancer pathways and processes, rather than randomly or based on mutation frequency alone, can improve predictive signal and identify more highly predictable mutations from gene expression data.

![
**A.** Overall distribution of performance across three gene sets, using gene expression (RNA-seq) data to predict mutations.
Each data point represents the mean cross-validated AUPR difference, compared with a baseline model trained on permuted mutation presence/absence labels, for one gene in the given gene set; notches show bootstrapped 95% confidence intervals.
"random" = 268 random genes, "most mutated" = 268 most mutated genes, "cancer gene set" = 268 cancer related genes from curated gene sets.
Significance stars indicate results of Bonferroni-corrected pairwise Wilcoxon tests: \*\*: _p_ < 0.01, \*\*\*: _p_ < 0.001, ns: not statistically significant for a cutoff of _p_ = 0.05.
**B, C, D.** Volcano-like plots showing mutation presence/absence predictive performance for each gene in each of the three gene sets.
The _x_-axis shows the difference in mean AUPR compared with a baseline model trained on permuted labels, and the _y_-axis shows _p_-values for a paired _t_-test comparing cross-validated AUPR values within folds.
Points (genes) marked with an "X" are overlapping between the cancer gene set and either the random or most mutated gene set.
](images/omics/figure_2.png){#fig:expression_gene_sets width="90%"}

#### Gene expression predicts cancer mutation status more effectively than DNA methylation

We compared gene expression with DNA methylation as downstream readouts of the effects of cancer alterations.
In these analyses, we considered both the 27K probe and 450K probe methylation datasets generated for the TCGA Pan-Cancer Atlas.
As target genes, we used the same combined cancer-related gene set described in the "Selection of cancer-related genes" section.
We used samples that had data for each of the data types being compared, including somatic mutation data to generate mutation labels.
This process retained 7,981 samples in the intersection of the expression, 27K methylation, 450K methylation, and mutation datasets, which we used for subsequent analyses.
The most frequent missing data types were somatic mutation data (1,114 samples) and 450K methylation data (1,072 samples) (Figure {@fig:methylation}A).

For many genes, predictions are better than our baseline model where labels are permuted (values greater than 0 in the box plots), suggesting that there is considerable predictive signal in both expression and methylation datasets across the cancer-related gene set (Figure {@fig:methylation}B).
On aggregate across all genes, predictive performance is best overall for gene expression.
Both before and after filtering for genes that exceed the significance threshold, gene expression with raw gene features provides a significant performance improvement relative to the 27K methylation and 450K methylation datasets (Figure {@fig:methylation}B-C).
Results were similar with PCA-compressed gene expression features or raw CpG probes as predictors (Additional File 1: Fig. S5).

Considering each target gene in the cancer-related gene set individually, we observed that 113/272 genes significantly outperformed the permuted baseline using gene expression data, as compared to 62/272 genes for 27K methylation and 77/272 genes for 450K methylation (Figure {@fig:methylation}D-F, more information about specific genes in Additional File 1: Fig. S2).
Some "well-predicted" genes that outperformed the permuted baseline tended to be similar between data types (Figure {@fig:methylation}D-F; genes in the top right of each plot).
For example, _CIC_ appears in the top right of all 3 plots, and _CCND1_ appears in the top right of the gene expression and 450K methylation plots, suggesting that mutations in these genes have strong gene expression and DNA methylation signatures, and these signatures tend to be preserved across cancer types.

In addition to comparing mutation classifiers trained on different data types to the permuted baseline, we also compared classifiers trained on true labels directly to each other, for genes that performed significantly better than the baseline for both of the data types under consideration (Figure {@fig:methylation}G-H).
We observed that 18/58 genes were significantly more predictable from expression data than 27K methylation data, and 21/69 genes were significantly more predictable from expression data than 450K methylation data.
In both cases, no genes were significantly more predictable using the methylation data types.
Still, we observed that some points were clustered around the origin, indicating that the data types appear to confer similar information about mutation status.
That is, in these cases, matching the gene being studied with the "correct" data modality seems to be unimportant: mutation status has a strong signature which can be extracted from both expression and DNA methylation data roughly equally.

We additionally compared pan-cancer survival prediction performance using principal components derived from each data type; in general, results were comparable across the three data types (Figure {@fig:methylation}I).
All data types outperformed the covariate-only baseline (see [Methods](#survival-prediction-using--omics-datasets)) for lower numbers of PC features included, although performance was similar to the baseline for higher numbers of PCs.
Confidence intervals between the best- and worst-performing data types overlap at most PC counts (with the exception of gene expression at 5,000 PC features), suggesting that similarly to mutation prediction, the three data types tend to have comparable effectiveness for pan-cancer survival prediction.

![
**A.** Count of overlapping samples between gene expression, 27K methylation, 450K methylation, and somatic mutation data used from TCGA.
Only non-zero overlap counts are shown.
Somatic mutation sample information is included because it is needed to generate the mutation presence/absence labels.
**B.** Predictive performance for genes in the cancer-related gene set, using each of the three data types as predictors.
The gene expression predictor uses the top 8000 gene features by mean absolute deviation, and the methylation predictors use the top 5000 principal components as predictive features.
Significance stars indicate results of Bonferroni-corrected pairwise Wilcoxon tests: \*\*: _p_ < 0.01, \*\*\*: _p_ < 0.001, ns: not statistically significant for a cutoff of _p_ = 0.05.
**C.** Predictive performance for genes where at least one of the considered data types predicts mutation labels significantly better than the permuted baseline.
**D-F.** Predictive performance for each gene in the cancer-related gene set, for each data type, compared with a baseline model trained on permuted labels.
**G-H.** Direct comparison of performance using gene expression and each methylation dataset, for genes that perform significantly better than the baseline for both data types. Points (genes) to the left of y=0 perform better using gene expression-derived features, and points to the right perform better using methylation-derived features.
**I.** Pan-cancer survival prediction performance, quantified using c-index on the _y_-axis, for gene expression, 27K methylation, and 450K methylation. The _x_-axis shows results with varying numbers of principal components included for each data type. Models also included covariates for patient age, sample mutation burden, and sample cancer type; grey dotted line indicates mean performance for a covariate-only baseline model.
](images/omics/figure_3.png){#fig:methylation width="90%"}

Focusing on several selected genes of interest, we observed that relative classifier performance varies by gene (Figure {@fig:methylation_genes}).
Past work has indicated that mutations in _TP53_ are highly predictable from gene expression data [@doi:10.1016/j.celrep.2018.03.076], and we observed that the methylation datasets provided similar predictive performance (Figure {@fig:methylation_genes}A).
Similarly, for _IDH1_ both expression and methylation features result in similar performance, consistent with the previously observed role of IDH1 in regulating both DNA methylation and gene expression (Figure {@fig:methylation_genes}D) [@doi:10.1038/ng.3457].
Mutations in _KRAS_ and _ERBB2_ (_HER2_) were most predictable from gene expression data, and in both cases the methylation datasets significantly outperformed the baseline as well (Figure {@fig:methylation_genes}B and {@fig:methylation_genes}E).
Gene expression signatures of _ERBB2_ alterations are historically well-studied in breast cancer [@doi:10.1038/sj.onc.1207361], and samples with activating _ERBB2_ mutations have recently been shown to share sensitivities to some small-molecule inhibitors across cancer types [@doi:10.1016/j.ccell.2019.09.001].
These observations are consistent with the pan-cancer _ERBB2_ mutant-associated expression signature that we observed in this study.
_NF1_ mutations were also most predictable from gene expression data, although the gene expression-based _NF1_ mutation classifier did not significantly outperform the baseline with permuted labels at a cutoff of $\alpha = 0.001$ (Figure {@fig:methylation_genes}C).
_SETD2_ is an example of a gene that is more predictable from the methylation datasets than from gene expression, although gene expression with raw gene features significantly outperformed the permuted baseline as well (Figure {@fig:methylation_genes}F).
_SETD2_ is widely mutated across cancer types and affects H3K36 histone methylation most directly, but SETD2-mediated changes in H3K36 methylation have been linked to dysregulation of diverse cellular processes including DNA methylation and RNA splicing [@doi:10.1101/cshperspect.a026468; @doi:10.1007/s00018-017-2517-x].

![
Performance across varying PCA dimensions for specific genes of interest. Dotted lines represent results for "raw" features (8,000 gene features for gene expression data and 8,000 CpG probes for both methylation datasets, selected by largest mean absolute deviation). Error bars and shaded regions show bootstrapped 95% confidence intervals. Stars in boxes show statistical testing results compared with permuted baseline model; each box refers to the model using the number of PCA components it is over (far right box = models with raw features). \*\*: _p_ < 0.01, \*\*\*: _p_ < 0.001, no stars: not statistically significant for a cutoff of _p_ = 0.05.
](images/omics/figure_4.png){#fig:methylation_genes width="90%"}

#### Comparing six different readouts favors expression and DNA methylation

Next, we expanded our comparison to all five functional data modalities (six total readouts, since there are two DNA methylation platforms) available in the TCGA Pan-Cancer Atlas.
As with previous experiments, we limited our comparison to the set of samples profiled for each readout, resulting in 5,226 samples with data for all readouts.
The data types with the most missing samples were RPPA data (2,215 samples that were missing RPPA data) and 450K methylation (630 samples that were missing 450K methylation data) (Figure {@fig:all_data}A).
Summarized over all genes in the cancer-related gene set, we observed that gene expression tended to produce better predictions than the other data types (Figure {@fig:all_data}B).
This remained true when we looked only at the set of genes having at least one significant predictor (i.e. "well-predicted" genes) (Figure {@fig:all_data}C).

On the individual gene level, mutations in 33/217 genes were significantly predictable from RPPA data relative to the permuted baseline, compared to 25/217 genes from microRNA data and 2/217 genes from mutational signatures data (Figure {@fig:all_data}D-F).
For the remaining data types on this smaller set of samples, 79/217 genes outperformed the baseline for gene expression data, 31/217 for 27k methylation, and 42/217 for 450k methylation.
Compared to the methylation experiments (Figure {@fig:methylation}), we observed fewer "well-predicted" genes for the expression and methylation datasets here (likely due to the considerably smaller sample size) but relative performance was comparable (Additional File 1: Fig. S3).
Direct comparisons between each added data type and gene expression data showed that for most "well-predicted" genes, RPPA, microRNA and mutational signatures data generally provide similar or worse performance compared to gene expression (Figure {@fig:all_data}G-I).

Performance using RPPA data (Figure {@fig:all_data}G) is notable because of its drastically smaller dimensionality than the other data types (190 proteins, compared to thousands of features for the expression and methylation data types).
This suggests that each protein abundance measurement provides a high information content, although this is by design as the antibody probes used for the TCGA analysis were selected to cover established cancer-related pathways [@doi:10.1038/nmeth.2650].
Similarly, the scope of the features captured by the mutational signatures data we used is limited to single-base substitution signatures; a broader spectrum of possible signatures is described in previous work [@doi:10.1038/s41586-020-1943-3; @doi:10.1038/s41586-019-1913-9] including doublet-substitution signatures, small indel signatures, and signatures of structural variation, but these were not readily available for the TCGA exome sequencing data.
The relatively poor predictive ability of mutational signatures likely stems from a combination of biological and technical factors, as there is no reason to expect that changes in somatic mutation patterns would be directly caused by most cancer driver mutations.
Two exceptions are _KMT2C_ and _KMT2D_ (Figure {@fig:all_data}F), which may have a role in mediating DNA damage response [@doi:10.1158/0008-5472.CAN-21-0688].

As in the expression/methylation comparison, we also compared pan-cancer survival prediction performance between all six readouts, using the top principal components derived from each data type to ensure comparable information content (Figure {@fig:all_data}J).
All six readouts performed comparably for this smaller set of samples, with slightly better performance across PC feature dimensions for the 450K methylation array.
The covariate-only baseline predictor performed considerably worse than it did in the expression/methylation comparisons, with all -omics data types outperforming the baseline predictor at all PC numbers.

![
**A.** Overlap of TCGA samples between all data types used in mutation prediction comparisons.
Only overlaps with more than 100 samples are shown.
Somatic mutation sample information is included because it is needed to generate the mutation presence/absence labels.
**B.** Overall distribution of performance per data type across 217 genes from the cancer-related gene set.
Each data point represents mean cross-validated AUPR difference, compared with a baseline model trained on permuted labels, for one gene; notches show bootstrapped 95% confidence intervals.
Significance stars indicate results of Bonferroni-corrected pairwise Wilcoxon tests: \*\*: _p_ < 0.01, \*\*\*: _p_ < 0.001, ns: not statistically significant for a cutoff of _p_ = 0.05.
All pairwise tests were run, and corrected for, but only neighboring test results are shown.
**C.** Overall performance distribution per data type for genes where the permuted baseline model is significantly outperformed for one or more data types, resulting in a total of 39 genes.
**D, E, F.** Volcano-like plots showing predictive performance for each gene in the cancer-related gene set, in each of the added data types (RPPA, microRNA, mutational signatures). The _x_-axis shows the difference in mean AUPR compared with a baseline model trained on permuted labels, and the _y_-axis shows _p_-values for a paired _t_-test comparing cross-validated AUPR values within folds.
**G, H, I.** Direct comparison of performance using gene expression and each added data type, showing only genes that perform significantly better than the baseline model for both data types. Points (genes) to the left of y=0 perform better using gene expression-derived features, and points to the right perform better using the added data type (RPPA, microRNA, and mutational signatures respectively).
**J.** Pan-cancer survival prediction performance, quantified using c-index on the _y_-axis, for all data types. The _x_-axis shows results with varying numbers of principal components included for each data type. Models also included covariates for patient age, sample mutation burden, and sample cancer type; grey dotted line indicates mean performance for a covariate-only baseline model.
](images/omics/figure_5.png){#fig:all_data width="90%"}

When we constructed a heatmap depicting predictive performance for each gene across data types, we found that many genes tended to be well-predicted by more than one data type (Figure {@fig:heatmap}).
Of the 86 genes that are well-predicted using at least one data type (grey circles in Figure {@fig:heatmap}), 52/86 (60.5%) are well-predicted by multiple data types, meaning that multiple -omics readouts contain a detectable signature of presence/absence of a mutation in the corresponding gene.
Of the remaining 34 genes, 28/34 (82.4%) are well-predicted by gene expression alone.
This supports our observation that in a surprising number of cases, choosing the "correct" data modality is unimportant for driver genes with strong functional signatures, although gene expression may be the best "default" choice as it tends to be a strong predictor in the majority of cases.
Exceptions included _ERBB4_, _KMT2A_, _PIK3R1_, and _RPL22_ (only well-predicted using RPPA data), _FAT4_ (only well-predicted using microRNA data), and _KDM6A_ (only well-predicted using 450K methylation data).

![
Heatmap displaying predictive performance for mutations in each of the 217 genes from the cancer-related gene set, across all six TCGA data modalities. Each cell quantifies performance for a target gene, using predictive features derived from a particular data type. Grey shaded dots indicate that the given data type provides significantly better predictions than the permuted baseline for the given gene; black inner dots indicate the same and additionally that the given data type provides statistically equivalent performance to the data type with the best average performance (determined by pairwise _t_-tests across data types with FDR correction).
](images/omics/figure_6.png){#fig:heatmap width="90%"}

#### Simple multi-omics integration provides little performance benefit

We also trained "multi-omics" classifiers to predict mutations in six well-studied and widely mutated driver genes across various cancer types: _EGFR_, _IDH1_, _KRAS_, _PIK3CA_, _SETD2_, and _TP53_.
Each of these genes is well-predicted from several data types in our earlier experiments (Figure {@fig:heatmap}), consistent with having strong pan-cancer driver effects.
For the multi-omics classifiers, we considered all pairwise combinations of the top three performing individual data types (gene expression, 27K methylation, and 450K methylation), in addition to a model using all three data types.
We trained a classifier for multiple data types by concatenating features from the individual data types, then fitting the same elastic net logistic regression model as we used for the single-omics models.
Here, we show results using the top 5,000 principal components from each data type as predictive features, to ensure that feature count and scale is comparable among data types; results for raw features are shown in Additional File 1: Fig. S6.
We additionally ran the same experiments using a 3-layer fully-connected neural network for classification, with principal components as input, and results are shown in Additional File 1: Fig. S7.
In general, we found predictions using elastic net logistic regression to be more robust across cross-validation folds and hyperparameter choices than predictions using the neural network, although the neural network provided a slight performance improvement using multiple -omics types for some genes.

For each of the six target genes, we observed comparable performance between the best single-omics classifier (blue boxes in Figure {@fig:multi_omics}A) and the best multi-omics classifier (orange boxes in Figure {@fig:multi_omics}A).
Across all classifiers and data types, we found varied patterns based on the target gene.
For _IDH1_ and _TP53_ performance is relatively consistent regardless of data type(s), suggesting that baseline performance is high and there is little room for improvement as data is added (Figure {@fig:multi_omics}C, G).
The _TP53_ classifier using raw features showed a statistically significant improvement when multiple data types were integrated, although the difference in mean performance was relatively small (Additional File 1: Fig. S6, _p_=0.0078).
For _EGFR_, _KRAS_, and _PIK3CA_, combining gene expression with methylation data results in statistically equivalent or worse performance to gene expression alone; classifiers trained only on methylation data generally do not perform as well as those that integrate expression data (Figure {@fig:multi_omics}B, D, E).
Previously, we saw that the best classifiers for _SETD2_ used methylation data alone (Figure {@fig:heatmap}).
When we added multiple data types to our _SETD2_ classifier, we did observe an increase in performance (Figure {@fig:multi_omics}F), although this improvement was not statistically significant in a paired-sample _t_-test for $\alpha$=0.05 (_p_=0.078).
Overall, we observed that combining data types in a relatively simple manner, by concatenating features from each individual data type, provided little or no improvement in predictive ability over the best individual data type.
This supports our earlier observations of the redundancy of gene expression and methylation data as functional readouts, since our multi-omics classifiers are not in general able to extract gains in predictive performance as more data types are added for this set of cancer drivers.

![
**A.** Comparing the best-performing model (i.e. highest mean AUPR relative to permuted baseline) trained on a single data type against the best "multi-omics" model for each target gene. None of the differences between single-omics and multi-omics models were statistically significant using paired-sample Wilcoxon tests across cross-validation folds, for a threshold of 0.05.
**B-G.** Classifier performance, relative to baseline with permuted labels, for mutation prediction models trained on various combinations of data types. Each panel shows performance for one of the six target genes; box plots show performance distribution over 8 evaluation sets (4 cross-validation folds x 2 replicates).
](images/omics/figure_7.png){#fig:multi_omics width="90%"}


### Discussion

We carried out a large-scale comparison of data types in the TCGA Pan-Cancer Atlas as functional readouts of genetic alterations in cancer, integrating results across cancer types and across driver genes.
Overall, we found that gene expression captures signatures of mutation state most effectively in general, relative to a baseline model, but we saw that for many genes other data types could be equally effective at predicting mutation presence or absence.
For pan-cancer survival prediction, we found that the functional readouts tended to be similarly effective, outperforming a simple baseline using age and sample mutation burden in most cases.
Our multi-omics modeling experiment indicated that the mutation state information captured by gene expression and DNA methylation is highly redundant, as added data types resulted in no gain or modest gains in classifier performance.

Comparing mutation status prediction using raw and PCA compressed expression and DNA methylation data, we observed that feature extraction using PCA provided no benefit compared to using raw gene or CpG probe features.
Other studies using DNA methylation array data have found that nonlinear dimension reduction methods, such as variational autoencoders and capsule networks, can be effective for extracting predictive features [@doi:10.1186/s12859-020-3443-8; @pubmed:34417465].
The latter approach is especially interesting because capsule networks and "capsule-like methods" can be constrained to extract features that align with known biology (i.e. that correspond to known disease pathways or CpG site annotations).
This can improve model interpretability as well as predictive performance.
Similar methods have been applied to extract biologically informed features from gene expression data (see, for instance, [@doi:10.1016/j.ccell.2020.09.014; @doi:10.1101/2021.05.25.445604]).
A more comprehensive study of dimension reduction methods in the context of mutation prediction, including the features selected by these methods and their biological relevance and interpretation, would be a beneficial area of future work.
In addition to methods for extracting features, another aspect of the study that could be explored further is methods for labeling samples as mutated or not more efficiently.
Although the mutation calls we used from MC3 represent the consensus of multiple algorithms aggregated through a standard pipeline, benchmarking other methods for identifying mutated samples could improve the utility of our method, such as calling mutations directly from RNA-seq data to avoid the need for paired samples [@doi:10.1016/j.ajhg.2013.08.008; @doi:10.7717/peerj.5362].

In contrast to many other studies demonstrating the benefits of integrating multiple -omics data types for various cancer-related prediction problems [@doi:10.1038/nrc3721; @doi:10.1016/j.jbi.2015.05.019; @doi:10.1186/s13040-018-0184-6; @doi:10.1093/bioinformatics/btz318; @doi:10.1371/journal.pcbi.1008878], we found that combining multiple data types to predict mutation status was generally not effective for this problem.
The method we used to integrate different data types by concatenating feature sets is sometimes referred to as "early" data integration (discussed in more detail in [@doi:10.1098/rsif.2015.0571] and [@doi:10.1016/j.inffus.2018.09.012]).
It is possible that more sophisticated data integration methods, such as "intermediate" integration methods that learn a set of features jointly across datasets, would produce improved predictions.
We do not interpret our results as concrete evidence that multi-omics integration is not effective for this problem; rather, we see them as an indication that this is a challenging data integration problem for which further investigation is needed.
We also present this problem as a set of benchmark tasks on which multi-omics integration methods can be evaluated.
In addition to the methodological questions, the issue of data integration also has implications for the underlying biology: a more nuanced understanding of when different data readouts provide redundant information, and when they can contribute unique information about cancer pathology and development, could have many translational applications.

One limitation of the current study is that, for the mutation prediction problem, we only evaluated classifiers that were trained on pan-cancer data.
Considering every possible combination of target gene and TCGA cancer type (85 target genes x 33 cancer types x 6 data types) would have drastically increased the computational load and presented a large multiple testing burden.
Alternatively, choosing only a subset of gene/cancer type combinations to study would have biased our results toward known driver gene/cancer type relationships, which we aimed to avoid.
In future work it would be interesting to identify classifiers that perform well in a certain cancer type but not in the pan-cancer context and to compare these instances across different cancer types.
As a motivating example, other studies have shown that activating mutations in Ras isoforms (_HRAS_, _KRAS_, _NRAS_) tend to have similar effects to one another in thyroid cancer, producing similar gene expression signatures [@doi:10.1142/9789811215636_0031].
In multiple myeloma, however, activating _KRAS_ and _NRAS_ mutations produce distinct expression signatures, necessitating separate classifiers [@doi:10.1182/bloodadvances.2019000303].
A high-throughput computational pipeline to identify cases where functional signatures of a particular cancer driver are either concordant or discordant between cancer types could identify opportunities for context-specific protein function prediction, improve biomarker identification, and suggest cases where drugs targeting specific alterations might produce discordant results in different cancer types.

As with any study relying on observational, cross-sectional data such as the TCGA Pan-Cancer Atlas, the conclusions that we can draw are limited by the data.
In particular, for any of our "well-predicted" genes (i.e. genes that, when mutated, have strong signatures in one or more data types), we cannot definitively distinguish correlation from causation.
To directly assess the effects of particular mutations on various data modalities, some studies use cell line data from sources such as the Cancer Cell Line Encyclopedia (CCLE) [@doi:10.1038/s41586-019-1186-3].
While this approach could help to isolate the causal effect of a given mutation on a given cell line, cell lines are sometimes an imperfect match for the cancers they are derived from [@doi:10.1038/s41467-019-11415-2].
We are also limited in that we cannot assign timing or clonal status to mutations, or fully characterize intratumor heterogeneity, with certainty from the bulk sequencing data generated by TCGA (although some features of tumor mutational processes over time can be estimated from bulk data, e.g. [@doi:10.1126/scitranslmed.aaa1408]).
As methods for generating large longitudinal datasets at single-cell resolution mature and scale, we will need to revise the way we think about cellular function and dysregulation in cancer cells, as dynamic and adaptive processes rather than a single representative snapshot of a tumor.


### Conclusions

Based on our results, for studies focused on the functional consequences of cancer mutations, we recommend that researchers cancers prioritize downstream readouts based on the gene or genes of interest (Figure {@fig:heatmap}).
On balance, prediction of mutation status is best in general using gene expression data, and prediction of patient survival is similar for all data types in the study.
However, the finding that for many genes, multiple functional profiles contain much of the same information will be useful for some study designs, given varying cost and stability of different readouts.
In addition to gene expression, results using DNA methylation and RPPA measurements as predictive features were promising, especially considering the substantially lower dimensionality of the RPPA dataset compared to other data types.
It is important to note that the specific technologies chosen by TCGA, and the tradeoffs inherent in such a high-throughput study, could influence the comparison: it is possible that, for instance, another technology for measuring DNA methylation (such as bisulfite sequencing) or another technique for measuring protein abundance (such as mass spectrometry-based proteomics) could change performance for those data types.
Future technology advances, in both quality and quantity of data, are likely to improve our understanding of the full picture of functional consequences of mutations in cancer cells.


### Declarations

#### Availability of data and materials

The datasets analyzed during this study were previously published as part of the TCGA Pan-Cancer Atlas project, and are publicly available from the NIH NCI Genomic Data Commons (GDC) [@pancanatlas]. The mutational signatures dataset was downloaded from the ICGC Data Portal [@mut_sigs]. Scripts used to download and preprocess the datasets for this study are available at `https://github.com/greenelab/mpmp/tree/master/00_download_data`.

All analyses were implemented in the Python programming language and are available at Zenodo [@mpmp_zenodo] and in the following GitHub repository: [`https://github.com/greenelab/mpmp`](https://github.com/greenelab/mpmp) [@mpmp_code] under the open-source BSD 3-clause license.
Scripts to download large data files from GDC and other sources are located in the `00_download_data` directory.
Scripts to run experiments comparing data modalities used individually are located in the `02_classify_mutations` directory, scripts to run multi-omics experiments are located in the `05_classify_mutations_multimodal` directory, and scripts to run survival prediction experiments are located in the `06_predict_survival` directory.
The Python environment was managed using `conda`, and directions for setting up the environment can be found in the `README.md` file.
Most analyses were run on the HTC CPU cluster at the University of Pittsburgh, except the neural networks which were trained and evaluated on the PMACS LPC GPU cluster at the University of Pennsylvania; scripts for training classifiers both locally for a single gene and on a Slurm cluster to reproduce the analysis of many genes in parallel are provided in the linked GitHub repo.
This manuscript was written using Manubot [@doi:10.1371/journal.pcbi.1007128] and is available on GitHub at [`https://github.com/greenelab/mpmp-manuscript`](https://github.com/greenelab/mpmp-manuscript) under the CC0-1.0 license [@manuscript_web] and at Zenodo [@manuscript_zenodo].

As a data resource, coefficients and hyperparameter choices for final models fit using all data types are available on Figshare: coefficients are available at [`https://doi.org/10.6084/m9.figshare.19576012`](https://doi.org/10.6084/m9.figshare.19576012) [@figshare_coefs] and hyperparameters are at [`https://doi.org/10.6084/m9.figshare.19576048`](https://doi.org/10.6084/m9.figshare.19576048) [@figshare_params]. File format/entries are described in the supplementary material in Additional File 1.

#### Acknowledgements

We would like to thank Alexandra Lee, Ariel Hippen, Ben Heil, Milton Pividori, and Natalie Davidson for reviewing the software associated with this work and providing insightful feedback.
This research was supported in part by the University of Pittsburgh Center for Research Computing through the resources provided.
Figure 1 (the schematic of the background and evaluation pipeline) was created using BioRender.com.



## Supplementary material for "Widespread redundancy in -omics profiles of cancer mutation states"

A version of the main paper figures using the area under the receiver-operator curve (AUROC) metric rather than AUPR is available at [`https://doi.org/10.6084/m9.figshare.14919729`](https://doi.org/10.6084/m9.figshare.14919729).

In a previous version of this paper, we ran our analysis only for the genes in the Vogelstein et al. 2013 gene set.
While there were some gene-to-gene differences in this set, we did not observe large differences between methylation and gene expression performances overall.
Scaling up the gene set by combining cancer gene sets from the literature as described in the methods/results sections affected the study results somewhat, as mutations in the added genes tend to be better predicted using gene expression than other data types.
During the revision, we explored the difference between the genes in this gene set and the genes in the "merged" cancer-related gene set but not in the Vogelstein genes.
GO analysis results for the Vogelstein genes are available at [`https://doi.org/10.6084/m9.figshare.19565890`](https://doi.org/10.6084/m9.figshare.19565890), and results for the non-Vogelstein genes are available at [`https://doi.org/10.6084/m9.figshare.19565887`](https://doi.org/10.6084/m9.figshare.19565887).
We noticed that the non-Vogelstein genes tend to be enriched for terms relating to transcription factors and transcriptional regulation.

As a data resource, coefficients and hyperparameter choices for final models fit using all data types are available on Figshare: coefficients are available at [`https://doi.org/10.6084/m9.figshare.19576012`](https://doi.org/10.6084/m9.figshare.19576012) and hyperparameters are at [`https://doi.org/10.6084/m9.figshare.19576048`](https://doi.org/10.6084/m9.figshare.19576048).
Columns in the coefficients dataset correspond to target genes (gene symbols), and rows correspond either to PCA components (for 27K and 450K methylation), -omics features (for all other data types), or covariates (cancer type indicator variables or log(mutation burden)).
An 'NA' value in a cell indicates that feature was not used in the model for the corresponding gene (for an -omics feature this could mean it was not in the top 8000 features by MAD, for a cancer type feature this means that cancer type was not included in the training set based on our mutation filters).
A 0 value in a cell indicates that feature was included in model training, but it was not selected by the elastic net feature selection algorithm.
Columns in the hyperparameters dataset correspond to hyperparameters (alpha and l1_ratio for elastic net logistic regression) and rows correspond to target genes.
For the methylation data types, PCA results (score and loading matrices) corresponding to the coefficients data are also available at [`https://doi.org/10.6084/m9.figshare.19908034`](https://doi.org/10.6084/m9.figshare.19908034).
These contain the top 5,000 principal components for each data type, which were used in the classifiers evaluated in the main paper.

Regarding the hyperparameters for the final models, recall that for the main figures in the paper, we evaluate each of our models using 2 replicates of 4-fold cross-validation.
For each of these folds (train/test splits), we further split the training set into train and validation sets to select hyperparameters, independently for each fold, and evaluate the models on the test set to get the results in the paper.
Because we are evaluating performance over multiple folds, it is not perfectly straightforward to get a single set of regression coefficients, since we have a (potentially different) set of coefficients for each cross-validation fold.
In order to synthesize these results into a single model for each gene in each data type, we selected one of the 8 sets of hyperparameters (from the 8 best models, 1 per CV fold) at random, with probability proportional to performance (AUPR) on the validation set used to select the hyperparameters, described above (so test set performance is not used here).
We then used the selected hyperparameters to train a single model on the entire dataset.

![
Proportion of samples from each TCGA cancer type that are "dropped" as more data types are added to our analyses. We started with gene expression data, and for each added data type, we took the intersection of samples that were profiled for that data type and the previous data types, dropping all samples that were missing 1 or more data types. Overall, at each step, the proportions of "dropped" samples appear to be fairly evenly spread between cancer types, showing that in general we are not disproportionately losing one or several cancer types as more data modalities are added to our analyses.
](images/omics/supp_figure_8.png){#fig:cancer_type_proportions width=80%}

![
Heatmap displaying predictive performance for mutations in each of the 272 genes from the cancer-related gene set, across gene expression and the two DNA methylation arrays. Each cell quantifies performance for a target gene, using predictive features derived from a particular data type. Grey shaded dots indicate that the given data type provides significantly better predictions than the permuted baseline for the given gene; black inner dots indicate the same and additionally that the given data type provides statistically equivalent performance to the data type with the best average performance (determined by pairwise _t_-tests across data types with FDR correction).
](images/omics/supp_figure_9.png){#fig:methylation_heatmap width=90%}

![
Volcano-like plots showing predictive performance for each gene in the cancer-related gene set for expression and DNA methylation, on the sample set used for the "all data types" experiments. The first row shows performance relative to the permuted baseline, and the second row shows direct comparisons between data types for genes that outperformed the permuted baseline only for both data types. The _x_-axis shows the difference in mean AUPR compared with a baseline model trained on permuted labels, and the _y_-axis shows _p_-values for a paired _t_-test comparing cross-validated AUPR values within folds.
](images/omics/supp_figure_10.png){#fig:all_volcano_me width=90%}

![
Volcano-like plots showing predictive performance for each gene in the cancer-related gene set for all data types, relative to the permuted baseline model, when genes are filtered based on the entire dataset rather than by cancer type.
For this filtering approach, we included/excluded entire genes rather than individual cancer types: specifically, we trained a classifier for each gene where all cancer types combined had at least 5% mutated samples and at least 100 total mutated samples, resulting in 182 total classifiers.
The _x_-axis shows the difference in mean AUPR compared with a baseline model trained on permuted labels, and the _y_-axis shows _p_-values for a paired _t_-test comparing cross-validated AUPR values within folds.
Counts of genes making the significance threshold of 0.001: gene expression 81/182 (44.5%), 27K methylation 16/182 (8.8%), 450K methylation 1/182 (0.6%), RPPA 41/182 (22.5%), microRNA 25/182 (13.7%), mutational signatures 7/182 (3.9%).
](images/omics/supp_figure_11.png){#fig:all_volcano_filter width=90%}

![
Predictive performance for genes in the cancer-related gene set, using each of the three data types as predictors.
The _x_-axis shows the number of PCA components used as features, "raw" = no PCA compression.
](images/omics/supp_figure_12.png){#fig:me_compress_boxes width=90%}

![
Top plot: comparing the best-performing model (i.e. highest mean AUPR relative to permuted baseline) trained on a single data type against the best "multi-omics" model for each target gene, using raw (not PCA compressed) features. For feature parity between data types, the top 20,000 features by mean absolute deviation were used for each data type.
The difference between single-omics and multi-omics performance for _TP53_ was statistically significant (_p_=0.0078), but other differences between single-omics and multi-omics models were not statistically significant using paired-sample Wilcoxon tests across cross-validation folds, for a threshold of 0.05.
Bottom plots: classifier performance, relative to baseline with permuted labels, for individual genes. Each panel shows performance for one of the six target genes; box plots show performance distribution over 8 evaluation sets (4 cross-validation folds x 2 replicates).
](images/omics/supp_figure_13.png){#fig:multi_omics_raw_feats width=90%}

![
Top plot: comparing the best-performing model (i.e. highest mean AUPR relative to permuted baseline) trained on a single data type against the best "multi-omics" model for each target gene, using a 3-layer fully-connected neural network. The top 5,000 principal components were used as predictive features for each data type.
The difference between single-omics and multi-omics performance for _PIK3CA_ (_p_ = 0.0156, in favor of multi-omics) and _TP53_ (_p_ = 0.0391, in favor of single-omics) were statistically significant, but other differences between single-omics and multi-omics models were not statistically significant using paired-sample Wilcoxon tests across cross-validation folds, for a threshold of 0.05.
Bottom plots: comparison of classifier performance between elastic net and fully-connected NN, relative to baseline with permuted labels, for individual genes. Each panel shows performance for one of the six target genes; box plots show performance distribution over 8 evaluation sets (4 cross-validation folds x 2 replicates).
](images/omics/supp_figure_14.png){#fig:multi_omics_mlp width=90%}


## Chapter 4: Smaller models do not exhibit superior generalization performance

This chapter has been posted as a preprint on bioRxiv (TODO) under the title "Smaller models do not exhibit superior generalization performance".

**Contributions:**
I designed and ran the experiments, created the figures, wrote the initial draft of the manuscript, and edited the manuscript. Casey S. Greene gave feedback and guidance on experiments, and edited the manuscript.


### Abstract

Existing guidelines in statistical modeling for genomics hold that simpler models have advantages over more complex ones.
Potential advantages include cost, interpretability, and improved generalization across datasets or biological contexts.
In cancer transcriptomics, this manifests as a preference for small "gene signatures", or groups of genes whose expression is used to define cancer subtypes or suggest therapeutic interventions.
To test the assumption that small gene signatures generalize better, we examined the generalization of mutation status prediction models across datasets (from cell lines to human tumors and vice-versa) and contexts (holding out entire cancer types from pan-cancer data).
We compared two simple procedures for model selection, one that exclusively relies on cross-validation performance and one that combines cross-validation performance with regularization strength.
We did not observe that more regularized signatures generalized better.
This result held across multiple problems and both linear models (LASSO logistic regression) and non-linear ones (neural networks).
When the goal of an analysis is to produce generalizable predictive models, we recommend choosing the ones that perform best on held-out data or in cross-validation, instead of those that are smaller or more regularized.



### Introduction

Gene expression datasets are typically "wide", with many gene features and relatively few samples.
These feature-rich datasets present obstacles in many aspects of machine learning, including overfitting and multicollinearity, and challenges in interpretation.
To facilitate the use of feature-rich gene expression data in machine learning models, feature selection and/or dimension reduction are commonly used to distill a more condensed data representation from the input space of all genes [@doi:10.1093/bioinformatics/btg062; @doi:10.1186/s13059-019-1861-6].
The intuition is that many gene expression features are likely irrelevant to the prediction problem, redundant, or contain no meaningful variation across samples, so transforming them or selecting a subset can generate a more reliable predictor.

In cancer transcriptomics, this preference for small, parsimonious sets of genes can be seen in the popularity of "gene signatures".
These are groups of genes whose expression levels are used to define cancer subtypes or to predict prognosis or therapeutic response [@doi:10.1038/nrg.2017.96; @doi:10.1016/j.ejca.2013.02.021].
Many studies specify the size of the signature in the paper's title or abstract, suggesting that the fewer genes in a gene signature, the better, e.g. [@doi:10.1056/NEJMoa060096; @doi:10.1158/0008-5472.CAN-08-0436; @doi:10.1056/NEJMoa1602253].
Clinically, there are many reasons why a smaller gene signature may be preferable, including cost (fewer genes may be less expensive to profile or validate, whereas a large signature likely requires a targeted array or NGS analysis [@doi:10.1586/erm.09.32]) and interpretability (it is easier to reason about the function and biological role of a smaller gene set than a large one since even disjoint gene signatures tend to converge on common biological pathways [@doi:10.1056/NEJMe068292; @doi:10.1038/nrclinonc.2011.125]).
There is also an underlying assumption that smaller gene signatures tend to be more robust: that for a new patient or in a new biological context, a smaller gene set or more parsimonious model will be more likely to maintain its predictive performance than a larger one.
This assumption has rarely been explicitly tested in genomics applications, but it is often included in guidelines or rules of thumb for statistical modeling or machine learning in biology, e.g. [@doi:10/bhfhgd; @doi:10.4137/CIN.S408; @doi:10.1371/journal.pcbi.1004961], and it is related in spirit to information-theoretic model selection approaches such as the Akaike information criterion (AIC) and the Bayesian information criterion (BIC) [@doi:10.1109/TAC.1974.1100705; @doi:10.1214/aos/1176344136].

In this study, we sought to test the robustness assumption directly by evaluating model generalization across biological contexts, inspired by previous work on domain adaptation and transfer learning in cancer transcriptomics [@doi:10.1038/s43018-020-00169-2; @doi:10.1038/s42256-021-00408-w; @doi:10.1073/pnas.2106682118].
We used two large, heterogeneous public cancer datasets: The Cancer Genome Atlas (TCGA) for human tumor sample data [@doi:10.1038/ng.2764], and the Cancer Cell Line Encyclopedia (CCLE) for human cell line data [@doi:10.1038/s41586-019-1186-3].
These datasets contain overlapping -omics data types derived from distinct data sources, allowing us to quantify model generalization across data sources.
In addition, each dataset contains samples from a wide range of different cancer types/tissues of origin, allowing us to quantify model generalization across cancer types.
We trained both linear and non-linear models to predict mutation status (presence or absence) from RNA-seq gene expression for approximately 70 cancer driver genes, across varying levels of model simplicity and degrees of regularization, resulting in a variety of gene signature sizes.
We compared two simple procedures for model selection, one that combines cross-validation performance with model parsimony and one that only relies on cross-validation performance, for each classifier in each context.

Our results suggest that, in general, mutation status classification models that perform well in cross-validation within a biological context also generalize well across biological contexts.
There are some individual genes and some individual cancer types where more regularized well-performing models outperform the best-performing model.
However, we do not observe a systematic generalization advantage for smaller/more regularized models across all genes and cancer types.
These results provide evidence that good cross-validation performance within a biological context (data source or cancer type) is a sufficient proxy for robust performance across contexts.



### Methods {.page_break_before}

#### Mutation data download and preprocessing

To generate binary mutated/non-mutated gene labels for our machine learning model, we used mutation calls for TCGA samples from MC3 [@doi:10.1016/j.cels.2018.03.002] and copy number threshold calls from GISTIC2.0 [@doi:10.1186/gb-2011-12-4-r41].
MC3 mutation calls were downloaded from the Genomic Data Commons (GDC) of the National Cancer Institute, at <https://gdc.cancer.gov/about-data/publications/pancanatlas>.
Thresholded copy number calls are from an older version of the GDC data and are available here: <https://figshare.com/articles/dataset/TCGA_PanCanAtlas_Copy_Number_Data/6144122>.
We removed hypermutated samples, defined as two or more standard deviations above the mean non-silent somatic mutation count, from our dataset to reduce the number of false positives (i.e., non-driver mutations).
Any sample with either a non-silent somatic variant or a copy number variation (copy number gain in the target gene for oncogenes and copy number loss in the target gene for tumor suppressor genes) was included in the positive set; all remaining samples were considered negative for mutation in the target gene.

We followed a similar procedure to generate binary labels for cell lines from CCLE, using the data available on the DepMap download portal at <https://depmap.org/portal/download/all/>.
Mutation information was retrieved from the `OmicsSomaticMutations.csv` data file, and copy number inforamtion was retrieved from the `OmicsCNGene.csv` data file.
We thresholded the CNV log-ratios provided by CCLE into binary gain/loss calls using a lower threshold of log~2~(3/2) (i.e. cell lines with a log-ratio below this threshold were considered to have a full copy loss in the corresponding gene), and an upper threshold of log~2~(5/2) (i.e. cell lines with a log-ratio above this threshold were considered to have a full copy gain in the corresponding gene).
After applying the same hypermutation criteria that we used for TCGA, no cell lines in CCLE were identified as hypermutated.
After preprocesing, 1402 cell lines with mutation and copy number data remained.
We then combined non-silent point mutations and copy number gain/loss information into binary labels using the same criteria as for TCGA.

#### Gene expression data download and preprocessing

RNA sequencing data for TCGA was downloaded from GDC at the same link provided above for the Pan-Cancer Atlas.
We discarded non-protein-coding genes and genes that failed to map, and removed tumors that were measured from multiple sites.
After filtering to remove hypermutated samples and taking the intersection of samples with both mutation and gene expression data, 9074 TCGA samples remained.

RNA sequencing data for CCLE was downloaded from the DepMap download portal, linked above, in the `CCLE_expression.csv` data file.
After taking the intersection of CCLE cell lines with both mutation and gene expression data, 1402 cell lines remained.
For experiments making predictions across datasets (i.e., training models on TCGA and evaluating performance on CCLE, or vice-versa) we took the intersection of genes in both datasets, resulting in 16041 gene features.
For experiments where only TCGA data was used (i.e., evaluating models on held-out cancer types), we used all 16148 gene features present in TCGA after the filtering described above.

#### Cancer gene set construction

In order to study mutation status classification for a diverse set of cancer driver genes, we started with the set of 125 frequently altered genes from Vogelstein et al. [@doi:10.1073/pnas.1616440113] (all genes from Table S2A).
For each target gene, to ensure that the training dataset was reasonably balanced (i.e., that there would be enough mutated samples to train an effective classifier), we included only cancer types with at least 15 mutated samples and at least 5% mutated samples, which we refer to here as "valid" cancer types.
In some cases, this resulted in genes with no valid cancer types, which we dropped from the analysis.
Out of the 125 genes originally listed in the Vogelstein et al. cancer gene set, we retained 71 target genes for the TCGA to CCLE analysis, and 70 genes for the CCLE to TCGA analyses.
For these analyses, each gene needed at least one valid cancer type in TCGA and one valid cancer type in CCLE, to construct the train and test sets.
For the cancer type holdout analysis, we retained 56 target genes: in this case, each gene needed at least two valid cancer types in TCGA to be retained, one to train on and one to hold out.

#### Classifier setup and cross-validation design

We trained logistic regression classifiers to predict whether or not a given sample had a mutational event in a given target gene using gene expression features as explanatory variables.
Our model was trained on gene expression data (X) to predict somatic mutation presence or absence (y) in a target gene.
To control for varying mutation burden per sample and to adjust for potential cancer type-specific expression patterns, we included one-hot encoded cancer type and log~10~(sample mutation count) in the model as covariates.
Since gene expression datasets tend to have many dimensions and comparatively few samples, we used a LASSO penalty to perform feature selection [@doi:10.1111/j.2517-6161.1996.tb02080.x].
LASSO logistic regression has the advantage of generating sparse models (some or most coefficients are 0), as well as having a single tunable hyperparameter which can be easily interpreted as an indicator of regularization strength/model simplicity.

LASSO ($\l_1$-penalized) logistic regression finds the feature weights $\hat{w} \in \mathbb{R}^{p}$ solving the following optimization problem:

$$\hat{w} = \text{argmin}_{w} \ (C \cdot l(X, y; w)) + ||w||_1$$

where $i \in \{1, \dots, n\}$ denotes a sample in the dataset, $X_i \in \mathbb{R}^{p}$ denotes features (gene expression measurements) from the given sample, $y_i \in \{0, 1\}$ denotes the label (mutation presence/absence) for the given sample, and $l(\cdot)$ denotes the negative log-likelihood of the observed data given a particular choice of feature weights, i.e.

$$l(X, y; w) = -\sum_{i=1}^{n} y_i \log\left(\frac{1}{1 + e^{-w^{\top}X_i}}\right) + (1 - y_i) \log\left(1 - \frac{1}{1 + e^{-w^{\top}X_i}}\right)$$

Given weight values $\hat{w}$, it is straightforward to predict the probability of a positive label (mutation in the target gene) $P(y^{*} = 1 \mid X^{*}; \hat{w})$ for a test sample $X^{*}$:

$$P(y^{*} = 1 \mid X^{*}; \hat{w}) = \frac{1}{1 + e^{-\hat{w}^{\top}X^{*}}}$$

and the probability of no mutation in the target gene, $P(y^{*} = 0 \mid X^{*}; \hat{w})$, is given by (1 - the above quantity).

This optimization problem leaves one hyperparameter to select: $C$, which controls the inverse of the strength of the L1 penalty on the weight values (i.e. regularization strength scales with $\frac{1}{C}$).
Although the LASSO optimization problem does not have a closed form solution, the loss function is convex, and iterative optimization algorithms are commonly used for finding reasonable solutions.
For fixed values of $C$, we solved for $\hat{w}$ using `scikit-learn`'s `LogisticRegression` method [@url:https://jmlr.org/papers/v12/pedregosa11a.html], which uses the coordinate descent optimization method implemented in `liblinear` [@url:https://www.jmlr.org/papers/v9/fan08a.html].
We selected this implementation rather than the `SGDClassifier` stochastic gradient descent implementation because coordinate descent/`liblinear` tends to generate sparser models and does not depend on a learning rate parameter, although after hyperparameter tuning performance is generally comparable between the implementations [@doi:10.1101/2023.06.26.546586].

To assess model selection across contexts (datasets and cancer types), we trained models using a variety of LASSO parameters on 75% of the training dataset, holding out 25% of the training dataset as the "cross-validation" set and also evaluating across contexts as the "test" set.
We trained models using $C$ values evenly spaced on a logarithmic scale between (10^-3^, 10^7^); i.e. the output of `numpy.logspace(-3, 7, 21)`.
This range was intended to give evenly distributed coverage across genes and cancer types that included "underfit" models (predicting only the mean or using very few features, poor performance on all datasets), "overfit" models (performing perfectly on training data but comparatively poorly on cross-validation and test data), and a wide variety of models in between that typically included the best fits to the cross-validation and test data.
To assess variability between train/CV splits, we used all 4 splits (25% holdout sets) x 2 random seeds for a total of 8 different training sets for each gene, using the same test set (i.e. all of the held-out context, either one cancer type or one dataset) in each case.

#### "Best model" vs. "smallest good model" analysis

For the "best" vs. "smallest good" model selection comparison, we started with 8 performance measurements (4 cross-validation folds x 2 random seeds) for each of 21 LASSO parameters.
We took the mean over these 8 measurements to get a single performance measurement for each model (LASSO parameter) on the holdout dataset, which has the same composition as the training set.
We used these per-parameter mean performance measurements to select the "best" model (LASSO parameter with the best performance on the holdout dataset), and the "smallest good" model (smallest LASSO parameter with performance in the top 25% of mean values on the holdout dataset, rounded up to the nearest integer).
For the distributions of differences shown in the Results, we took the difference in mean performance for the "best" and "smallest good" models for each gene, with positive differences indicating better performance for the "best" model and negative differences better performance for the "smallest good" model, for each gene.

#### Neural network setup and parameter selection

As a tradeoff between computational cost and ability to represent non-linear decision boundaries, inspired by the architecture of the intermediate-complexity model described in [@doi:10.1371/journal.pcbi.1010984], we trained a three-layer fully connected neural network with ReLU nonlinearities [@https://dl.acm.org/doi/10.5555/3104322.3104425] to predict mutation status.
For the experiments described in the main paper, we varied the size of the first hidden layer in the range {1, 2, 3, 4, 5, 10, 50, 100, 500, 1000}.
We fixed the size of the second hidden layer to be half of the size of the first hidden layer, rounded up to the nearest integer, and the size of the third hidden layer was the number of classes, 2 in our case.
Our models were trained for 100 epochs of mini-batch stochastic gradient descent in PyTorch [@arxiv:1912.01703], using the Adam optimizer [@arxiv:1412.6980] and a fixed batch size of 50.
To select the remaining hyperparameters for each hidden layer size, we performed a random search over 10 combinations, with a single train/test split stratified by cancer type, using the following hyperparameter ranges: learning rate {0.1, 0.01, 0.001, 5e-4, 1e-4}, dropout proportion {0.1, 0.5, 0.75}, weight decay (L2 penalty) {0, 0.1, 1, 10, 100}.
We used the same train/cross-validation split strategy described above, generating 8 different performance measurements for each gene and hidden layer size, for the neural network experiments as well.

For the _EGFR_ gene, we also ran experiments where we varied the dropout proportion and the weight decay hyperparameter as the regularization axis, and selected the remaining hyperparameters (including the hidden layer size) using a random search.
In these cases, we used a fixed range for dropout of {0.0, 0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95}, and a fixed range for weight decay of {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 10.0}.
All neural network analyses were performed on a Ubuntu 18.04 machine with a NVIDIA RTX 2060 GPU.


### Results {.page_break_before}

#### Evaluating model generalization using public cancer data

We collected data from the TCGA Pan-Cancer Atlas and the Cancer Cell Line Encyclopedia to predict the presence or absence of mutations in cancer genes, as a benchmark of cancer-related information content across cancer types and contexts.
We trained mutation status classifiers across approximately 70 genes involved in cancer development and progression from Vogelstein et al. 2013 [@doi:10.1126/science.1235122], using LASSO logistic regression with gene expression (RNA-seq) values as predictive features.
We fit each classifier across a variety of regularization parameters, resulting in models with a variety of different sparsity levels between the extremes of 0 nonzero features and all features included (Supplementary Figure {@fig:average_sparsity}).
Inspired by the generalization experiments across tissues and model systems in [@doi:10.1038/s43018-020-00169-2], we designed experiments to evaluate the generalization of mutation status classifiers across datasets (TCGA to CCLE and CCLE to TCGA) and across biological contexts (cancer types) within TCGA, relative to a within-dataset baseline (Figure {@fig:generalization_overview}).

![
Schematic of experimental design. The colors of the "dots" in the training/model selection/model evaluation panels on the left correspond to train/CV/test curves in the following results figures.
](images/generalization/figure_1.png){#fig:generalization_overview width="90%"}

#### Generalization from human tumor samples to cell lines is more effective than the reverse

To evaluate "cross-dataset" generalization, we trained mutation status classifiers on human tumor data from TCGA and evaluated them on cell line data from CCLE, as well as the reverse from CCLE to TCGA.
As an example, we examined _EGFR_, an oncogenic tyrosine kinase that is commonly mutated in diverse cancer types and cancer cell lines, including lung cancer, colorectal cancer, and glioblastoma [@doi:10.1146/annurev-pathol-011110-130206; @doi:10.1002/cac2.12005].
For EGFR mutation status classifiers trained on TCGA and evaluated on CCLE, we saw that AUPR on cell lines was slightly worse than on held-out tumor samples, but comparable across regularization levels/LASSO parameters (Figure {@fig:tcga_ccle_overall}A).
On the other hand, EGFR classifiers trained on CCLE and evaluated on TCGA performed considerably worse on human tumor samples as compared to held-out cell lines (Figure {@fig:tcga_ccle_overall}B).

To explore these tendencies more generally, we compared performance across all genes in the Vogelstein et al. dataset, for both TCGA to CCLE and CCLE to TCGA generalization.
We measured the difference between performance on the holdout data within the training dataset and performance across datasets, with a positive difference indicating poor generalization (better holdout performance than test performance) and a 0 or negative difference indicating good generalization (comparable test performance to holdout performance).
For generalization from TCGA to CCLE, we observed that median AUPR differences were mostly centered around 0 for most genes, with some exceptions at the extremes (Figure {@fig:tcga_ccle_overall}C; performance differences on the y-axis).
An example of a gene exhibiting poor generalization was _IDH1_, the leftmost gene in Figure {@fig:tcga_ccle_overall}C, with good performance on held-out TCGA data and poor performance on CCLE data.
IDH-mutant glioma cell lines are poorly represented compared to IDH-mutant patient tumors, which may explain the difficulty of generalization to cell lines for _IDH1_ mutation classifiers [@doi:10.1093/noajnl/vdaa088].
For generalization from CCLE to TCGA, we observed a more pronounced upward shift toward better performance on CCLE and worse on TCGA, with most genes performing better on the CCLE holdout data and very few genes generalizing comparably to the TCGA samples (Figure {@fig:tcga_ccle_overall}D).

![
**A.** _EGFR_ mutation status prediction performance on training samples from TCGA (blue), held-out TCGA samples (orange), and CCLE samples (green), across varying LASSO parameters.
**B.** _EGFR_ mutation status prediction performance on training samples from CCLE (blue), held-out CCLE samples (orange), and TCGA samples (green).
**C.** Difference in mutation status prediction performance for models trained on TCGA (holdout data) and evaluated on CCLE (test data), across 71 genes from Vogelstein et al. For each gene, the best model (LASSO parameter) was selected using holdout AUPR performance. Genes on x-axis are ordered by median AUPR difference across cross-validation splits, from highest to lowest.
**D.** Difference in mutation status prediction performance for models trained on CCLE (holdout data) and evaluated on TCGA (test data), across 70 genes from Vogelstein et al.
](images/generalization/figure_2.png){#fig:tcga_ccle_overall width="80%"}

#### "Best" and "smallest good" model selection strategies perform comparably

To address the question of whether more parsimonious models tend to generalize better or not, we designed two model selection schemes and compared them for the TCGA to CCLE and CCLE to TCGA mutation prediction problems (Figure {@fig:tcga_ccle_smallest_best}A).
The "best" model selection scheme chooses the top-performing model/LASSO parameter on the holdout dataset from the same source as the training data and applies it to the test data from the other data source.
The intention of the "smallest good" model selection scheme is to balance parsimony with reasonable performance on the holdout data, since simply selecting the smallest possible model (generally, the dummy regressor/mean predictor) is not likely to generalize well.
To accomplish this, we first identify the top 25% of well-performing models on the holdout dataset; then, from this subset of models, we choose the smallest (i.e., highest LASSO parameter) to apply to the test data.
In both cases, we exclusively use the holdout data to select a model and only apply the model to out-of-dataset samples to evaluate generalization performance _after_ model selection.

For TCGA to CCLE generalization, 31/71 genes (43.7%) had better performance for the "best" model, and 15/71 genes (21.1%) had better generalization performance with the "smallest good" model.
The other 25 genes had the same "best" and "smallest good" model (in other words, the "smallest good" model was also the best-performing overall, and the difference was 0) (Figure {@fig:tcga_ccle_smallest_best}B).
For CCLE to TCGA generalization, 24/70 genes (34.3%) had better performance for the "best" model and 19/70 (27.1%) for the "smallest good," with the other 27 having the same model fulfill both criteria (Figure {@fig:tcga_ccle_smallest_best}C).
Overall, these results do not support the hypothesis that the most parsimonious model generalizes the best: for both generalization problems there are slightly more genes where the best-performing model on the holdout dataset is also the best-performing on the test set, although there are some genes where the "smallest good" approach works well.

We examined genes that fell into either category for TCGA to CCLE generalization (dotted lines on Figure {@fig:tcga_ccle_smallest_best}B).
For _NF1_, the "best" model outperforms the "smallest good" model (Figure {@fig:tcga_ccle_smallest_best}D).
Comparing holdout (orange) and cross-dataset (green) performance, both generally follow a similar trend, with the cross-dataset performance peaking when the holdout performance peaks at a regularization parameter of $\alpha = 0.00316$.
_PIK3CA_ is an example of the opposite, a gene where the "smallest good" model tends to outperform the "best" model (Figure {@fig:tcga_ccle_smallest_best}E).
In this case, the peak for the cross-dataset performance occurs at a higher level of regularization (further left on the x-axis), at $\alpha = 0.01$, than the peak for the holdout performance, at $\alpha = 0.0316$.
This suggests that a _PIK3CA_ mutation status classifier that is more parsimonious, but that has slightly worse performance, does tend to generalize better across datasets to CCLE.

![
**A.** Schematic of "best" vs. "smallest good" model comparison experiments.
**B.** Distribution of performance comparisons between "best" and "smallest good" model selection strategies, for TCGA -> CCLE generalization. Positive x-axis values indicate better performance for the "best" model, negative values indicate better performance for the "smallest good" model.
**C.** Distribution of performance comparisons between "best" and "smallest good" model selection strategies, for CCLE -> TCGA generalization.
**D.** _NF1_ mutation status prediction performance generalizing from TCGA (holdout, orange), to CCLE (green), with "best" and "smallest good" models labeled.
**E.** _PIK3CA_ mutation status prediction performance generalizing from TCGA (holdout, orange), to CCLE (green), with "best" and "smallest good" models labeled.
](images/generalization/figure_3.png){#fig:tcga_ccle_smallest_best width="90%"}

#### Generalization across cancer types yields similar results to generalization across datasets

To evaluate generalization across biological contexts within a dataset, we trained mutation prediction classifiers on all but one cancer type in TCGA, performed model selection on a holdout set stratified by cancer type, and held out the remaining cancer type as a test set.
We performed the same "best" vs. "smallest good" analysis that was previously described, across 294 gene/holdout cancer type combinations (Figure {@fig:cancer_type_holdout}A).
We observed 133/294 gene/cancer type combinations (45.2%) that had better generalization performance with the "best" model, compared to 84/294 (28.6%) for the "smallest good" model.
The other 77 gene/cancer type combinations had the same "best" and "smallest good" model and thus no difference in performance.
This is consistent with our cross-dataset experiments, with slightly more instances where the "best" model on the stratified holdout data also generalizes the best, but no pronounced distributional shift in either direction.

We looked in more detail at two examples of gene/cancer type combinations, one on either side of the 0 point for cross-cancer type generalization.
For prediction of _SETD2_ mutation status in papillary renal cell carcinoma, we observed the best cross-cancer type performance for relatively low levels of regularization/high x-axis values (Figure {@fig:cancer_type_holdout}B).
For prediction of _CDKN2A_ mutation status in low grade glioma, on the other hand, we observed the best cross-cancer generalization for a high level of regularization ($\alpha = 0.01$), and generalization capability for the best parameter on the stratified holdout set ($\alpha = 0.0316$) was lower (Figure {@fig:cancer_type_holdout}C).

We aggregated results across genes for each cancer type, looking at performance in the held-out cancer type compared to performance on the stratified holdout set (Figure {@fig:cancer_type_holdout}D).
Cancer types that were particularly difficult to generalize to (better performance on stratified data than cancer type holdout, or positive y-axis values) include testicular cancer (TGCT) and soft tissue sarcoma (SARC), which are notable because they are not carcinomas like the majority of cancer types included in TCGA, potentially making generalization harder.
We also aggregated results across cancer types for each gene, identifying a distinct set of genes where classifiers tend to generalize poorly no matter what cancer type is held out (Supplementary Figure {@fig:average_perf_by_gene}).
Included in this set of genes with poor generalization performance are _HRAS_, _NRAS_, and _BRAF_, suggesting that a classifier that combines mutations in Ras pathway genes into a single "pathway mutation status" label (as described in [@doi:10.1016/j.celrep.2018.03.046], or using more general computational approaches such as [@doi:10.1142/9789811215636_0031; @doi:10.1038/s41525-022-00328-7]) could be a better approach than separate classifiers for each gene.

In the cancer type aggregation plot (Figure {@fig:cancer_type_holdout}D), thyroid carcinoma (THCA) stood out as a carcinoma that had poor performance when held out.
In our experiments, the only genes in which THCA is included as a held-out cancer type are _BRAF_ and _NRAS_; generalization performance for both genes is below cross-validation performance, but slightly worse for _NRAS_ than _BRAF_ (Supplementary Figure {@fig:thca_by_gene}).
Previous work suggests that _BRAF_ mutation tends to have a different functional signature in THCA than other cancer types, and withholding THCA from the training set improved classifier performance, which could at least in part explain the difficulty of generalizing to THCA we observe [@doi:10.1016/j.celrep.2018.03.046].

![
**A.** Distribution of performance comparisons between "best" and "smallest good" model selection strategies, for generalization across TCGA cancer types. Each point is a gene/cancer type combination; positive x-axis values indicate better performance for the "best" model and negative values indicate better performance for the "smallest good" model.
**B.** _SETD2_ mutation status prediction performance generalizing from other cancer types in TCGA (stratified holdout, orange) to papillary renal cell carcinoma (KIRP, green), with "best" and "smallest good" models labeled.
**C.** _CDKN2A_ mutation status prediction performance generalizing from other cancer types in TCGA (stratified holdout, orange) to low grade glioma (LGG, green), with "best" and "smallest good" models labeled.
**D.** Distributions of performance difference between CV data (same cancer types as train data) and holdout data (cancer types not represented in train data), by held-out cancer type. Each point is a gene whose mutation status classifier was used to make predictions on out-of-dataset samples in the relevant cancer type.
](images/generalization/figure_4.png){#fig:cancer_type_holdout width="90%"}

#### Small neural network hidden layer sizes tend to generalize poorly

To test whether or not findings generalize to non-linear models, we trained a 3-layer neural network to predict mutation status from gene expression for generalization from TCGA to CCLE, and we varied the size of the first hidden layer to control regularization/model complexity.
We fixed the size of the second hidden layer to be half the size of the first layer, rounded up to the nearest integer; further details in Methods.
For _EGFR_ mutation status prediction, we saw that performance for small hidden layer sizes was noisy, but generally lower than for higher hidden layer sizes (Figure {@fig:tcga_ccle_nn}A).
On average, over all 71 genes from Vogelstein et al., performance on both held-out TCGA data and CCLE data tends to increase until a hidden layer size of 10-50, then flatten (Figure {@fig:tcga_ccle_nn}B).
To explore additional approaches to neural network regularization, we also tried varying dropout and weight decay for _EGFR_ and _KRAS_ mutation status classification while holding the hidden layer size constant.
Results followed a similar trend, with generalization performance generally tracking performance on holdout data (Supplementary Figure {@fig:nn_dropout_wd}).

In order to measure which hidden layer sizes tended to perform relatively well or poorly, across different mutated cancer genes with different effect sizes, we ranked the range of hidden layer sizes by their generalization performance on CCLE (with low ranks representing good performance, and high ranks representing poor performance; Figure {@fig:tcga_ccle_nn}C).
For each hidden layer size, we then visualized the distribution of ranks above and below the median rank of 5.5/10; a high proportion of ranks above the median (True, or blue bar) signifies poor overall performance for that hidden layer size, and a high proportion of ranks below the median (False, or orange bar) signifies good performance.
We saw that small hidden layer sizes tended to generalize poorly (<5, but most pronounced for 1 and 2), and intermediate hidden layer sizes tended to generalize well (10-100, and sometimes 500/1000).
This suggests that some degree of parsimony/simplicity could be useful, but very simple models do not tend to generalize well.
We also performed the same "best"/"smallest good" analysis as with the linear models, using hidden layer size as the regularization axis instead of LASSO regularization strength.
We observed a distribution centered around 0, suggesting that the "best" and "smallest good" models tend to generalize similarly (Figure {@fig:tcga_ccle_nn}D).
28/71 genes (45.2%) had better generalization performance with the "best" model, compared to 21/71 (28.6%) for the "smallest good" model and 22 with the same "best" and "smallest good" model.

![
**A.** _EGFR_ mutation status prediction performance on training samples from TCGA (blue), held-out TCGA samples (orange), and CCLE samples (green), across varying neural network hidden layer sizes.
**B.** Mutation status prediction performance summarized across all genes from Vogelstein et al. on training samples from TCGA (blue), held-out TCGA samples (orange), and CCLE samples (green), across varying neural network hidden layer sizes.
**C.** Distribution of ranked performance values above/below the median rank for each gene, for each of the hidden layer sizes evaluated. Lower ranks indicate better performance and higher ranks indicate worse performance, relative to other hidden layer sizes.
**D.** Distribution of performance comparisons between "best" and "smallest good" model selection strategies, for TCGA -> CCLE generalization with neural network hidden layer size as the regularization axis. Positive x-axis values indicate better performance for the "best" model, negative values indicate better performance for the "smallest good" model.
](images/generalization/figure_5.png){#fig:tcga_ccle_nn width="90%"}


### Discussion

Using public cancer genomics and transcriptomics data from TCGA and CCLE, we studied generalization of mutation status classifiers for a wide variety of cancer driver genes.
We designed experiments to evaluate generalization across biological contexts by holding out cancer types in TCGA, and to evaluate generalization across datasets by training models on TCGA and evaluating them on CCLE, and vice-versa.
We found that, in general, smaller or more parsimonious models do not tend to generalize more effectively across cancer types or across datasets, and in the absence of prior knowledge about a prediction problem, simply choosing the model that performs the best on a holdout dataset is at least as effective for selecting models that generalize.

Our results were similar in both linear models (LASSO logistic regression) and non-linear deep neural networks when using hidden layer size as the regularization parameter of interest.
In our non-linear model experiments, we did not observe better generalization across datasets for fully connected neural networks with fewer hidden layer nodes, and our preliminary results indicated a similar trend for dropout and weight decay.
Compared to linear models, it is less clear how to define a "small" or "parsimonious" neural network model since there are many regularization techniques that one may use to control complexity.
Rather than simply removing nodes and keeping the network fully connected, another approach to parsimony could be to select an inductive bias to guide the size reduction of the network.
Existing examples include network structures guided by protein-protein interaction networks or function/pathway ontologies [@doi:10.1016/j.ccell.2020.09.014; @doi:10.1093/bioinformatics/btx624; @doi:10.1186/s13059-020-02100-5; @doi:10.1016/j.patter.2022.100565].
It is possible that a smaller neural network with a structure that corresponds more appropriately to the prediction problem would achieve better generalization results, although choosing an apt network structure or data source can be a challenging aspect of such efforts.

For generalization from CCLE to TCGA, we observed that performance was generally worse on human tumor samples from TCGA than for held-out cell lines.
This could, at least in part, be a function of sample size: the number of cell lines in CCLE is approximately an order of magnitude smaller than the number of tumor samples in TCGA (~10,000 samples in TCGA vs. ~1,500 cell lines in CCLE, although the exact number of samples used to train and evaluate our classifiers varies by gene, see Methods for further detail).
There are also plausible biological and technical explanations for the difficulty of generalizing to human tumor samples.
This result could reflect the imperfect and limited nature of cancer cell lines as a model system for human tumors, which previous studies have pointed out [@doi:10.1093/jnci/djt007; @doi:10.1158/0008-5472.CAN-13-2971; @doi:10.1016/j.cell.2016.06.017].
In addition, the CCLE data is collected and processed uniformly, as described in [@doi:10.1038/s41586-019-1186-3], while the TCGA data is processed by a uniform pipeline but collected from a wide variety of different cancer centers around the US [@doi:10.1038/ng.2764].

When we ranked cancer types in order of their generalization difficulty aggregated across genes, we noticed a slight tendency toward non-carcinoma cancer types (TGCT, SARC, SKCM) being difficult to generalize to.
It has been pointed out in other biological data types that holding out entire contexts or domains is necesssary for a full picture of generalization performance [@doi:10.1186/s13059-020-02177-y; @doi:10.1038/s41576-021-00434-9], which our results corroborate.
This highlights a potential weakness of using TCGA's carcinoma-dominant pan-cancer data as a training set for a broad range of tasks, for instance in foundation models which are becoming feasible for some genomics applications [@arxiv:2306.15794; @doi:10.1101/2023.04.30.538439; @doi:10.1101/2023.05.29.542705].
One caveat of our analysis is that each cancer type is included in the training data or held out for a different subset of genes, so it is difficult to detangle gene-specific effects (some mutations have less distinguishable functional effects on gene expression than others) from cancer type-specific effects (some cancer types are less similar to each other than others) on prediction performance using our experimental design.



## Conclusion

Without directly evaluating model generalization, it is tempting to assume that simpler models will generalize better than more complex ones, and previous studies and sets of guidelines suggest this rule of thumb [@doi:10.1214/088342306000000060; @doi:10/bhfhgd; @doi:10.4137/CIN.S408; @doi:10.1371/journal.pcbi.1004961].
However, we do not observe strong evidence that simpler models inherently generalize more effectively than more complex ones.
There may be other reasons to train small models or to look for the best model of a certain size/sparsity, such as biomarker interpretability or assay cost.
Our results underscore the importance of defining clear goals for each analysis.
If the goal is to achieve generalization across contexts or datasets, whenever possible we recommend directly evaluating generalization.
When it is not feasible, we recommend choosing the model that performs the best on unseen data via cross-validation or a holdout dataset.



### Data and code availability

The data from TCGA analyzed during this study were previously published as part of the TCGA Pan-Cancer Atlas project [@doi:10.1038/ng.2764], and are available from the NIH NCI Genomic Data Commons (GDC).
The data from CCLE analyzed during this study were previously published [@doi:10.1038/s41586-019-1186-3], and are available from the Broad Institute's DepMap Portal.
Raw classification results, performance figures for all genes in the Vogelstein et al. 2013 dataset, and parameter selection results and performance comparisons for each individual gene in the "best vs. smallest good" analyses are available on Figshare at <https://doi.org/10.6084/m9.figshare.23826450>, under a CC0 license.
The scripts used to download and preprocess the datasets for this study are available at <https://github.com/greenelab/pancancer-evaluation/tree/master/00_process_data>.
Scripts for TCGA <-> CCLE comparisons (Figures 2 and 3) and neural network experiments (Figure 5) are available in the <https://github.com/greenelab/pancancer-evaluation/tree/master/08_cell_line_prediction> directory.
Scripts for TCGA cancer type comparisons (Figure 4) are available in the <https://github.com/greenelab/pancancer-evaluation/tree/master/02_cancer_type_classification> directory.
All scripts are available under the open-source BSD 3-clause license.

This manuscript was written using Manubot [@doi:10.1371/journal.pcbi.1007128] and is available on GitHub at <https://github.com/greenelab/generalization-manuscript> under the CC0-1.0 license.
This research was supported in part by the University of Pittsburgh Center for Research Computing through the resources provided. Specifically, this work used the HTC cluster, which is supported by NIH award number S10OD028483.


### Supplementary Material

![Number of nonzero coefficients (model sparsity) across varying regularization parameters, for 71 genes (TCGA to CCLE prediction, top) and 70 genes (CCLE to TCGA prediction, bottom) in the Vogelstein et al. dataset.](images/generalization/supp_figure_1.png){#fig:average_sparsity width="100%"}

![Distributions of performance difference between cross-validation data (same cancer types as training data) and holdout data (cancer types not represented in data), grouped by held-out gene. Each point shows performance for a single train/validation split for one cancer type that was held out, using a classifier trained to predict mutations in the given gene.](images/generalization/supp_figure_2.png){#fig:average_perf_by_gene width="100%" .page_break_before}

![Top row: Distribution of performance differences when thyroid cancer (THCA) data is held out from training setacross seeds/folds, grouped by gene. Bottom row: Distributions of performance differences for genes where THCA is included in training/holdout sets, relative to other cancer types that are included.](images/generalization/supp_figure_3.png){#fig:thca_by_gene width="100%" .page_break_before}

![Performance vs. dropout parameter (first column) and weight decay strength (second column), for EGFR mutation prediction (first row) and KRAS mutation prediction (second row) using a 3-layer fully connected neural network trained on TCGA (blue/orange) and evaluated on CCLE (green).](images/generalization/supp_figure_4.png){#fig:nn_dropout_wd width="100%" .page_break_before}


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

