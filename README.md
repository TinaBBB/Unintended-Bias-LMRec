# Unintended-Bias-LMRec
Accepted paper at IPM2022


### 1. Dataset

This work conducts experiments on the [Yelp](https://www.yelp.com/dataset/download) datasets, specifically, we collected Yelp data for twelve years spanning 2008 to 2020, related to 7 North American cities, including:

* [Atlanta](https://drive.google.com/file/d/1-AWNH8L6TECte-JTWLhqf6CNB7YwSgrT/view?usp=sharing)
* [Austin](https://drive.google.com/file/d/1KOXbXnqfqGDaBNI8M8KHPkacOzRxeE3h/view?usp=sharing)
* [Boston](https://drive.google.com/file/d/1-7S05lYECC0nLtPMDCv_hXGj-kbicyH8/view?usp=sharing)
* [Columbus](https://drive.google.com/file/d/1-3DPqjYJHDkwMQHZ0gBWuuE1j9ZWDme7/view?usp=sharing)
* [Orlando](https://drive.google.com/file/d/1-43kDGH2sm1EMbgJZ-XHFUmDSs_oY07g/view?usp=sharing)
* [Portland](https://drive.google.com/file/d/1L-TL83f0E9vKVT2LfOhYW-Gq95HibPI1/view?usp=sharing)
* [Toronto](https://drive.google.com/file/d/19wrnZVth0YWQvWrk9wtdXmpUvcJsg2s6/view?usp=sharing)

The review dataset for each city can be accessed by clicking the city name above. Each dataset has the following format as csv files.

dataframe columns include: 

```
['business_id', 'review_stars', 'review_date', 'review_id', 'review_text', 
'user_id', 'user_id.1', 'Year', 'Day', 'Month', 
'alias', 'coordinates', 'name', 'price', 'business_stars', 
'latitude', 'longitude', 'date', 'categories']
```

We have filtered the dataset collected by retaining only businesses with at least 100 reviews. The table below provides detailed statistics of the Yelp dataset for each city.

|                            | **[Atlanta](https://drive.google.com/file/d/1-AWNH8L6TECte-JTWLhqf6CNB7YwSgrT/view?usp=sharing)** | **[Austin](https://drive.google.com/file/d/1KOXbXnqfqGDaBNI8M8KHPkacOzRxeE3h/view?usp=sharing)** | **[Boston](https://drive.google.com/file/d/1-7S05lYECC0nLtPMDCv_hXGj-kbicyH8/view?usp=sharing)** | **[Columbus](https://drive.google.com/file/d/1-3DPqjYJHDkwMQHZ0gBWuuE1j9ZWDme7/view?usp=sharing)** | **[Orlando](https://drive.google.com/file/d/1-43kDGH2sm1EMbgJZ-XHFUmDSs_oY07g/view?usp=sharing)** | **[Portland](https://drive.google.com/file/d/1L-TL83f0E9vKVT2LfOhYW-Gq95HibPI1/view?usp=sharing)** | **[Toronto](https://drive.google.com/file/d/19wrnZVth0YWQvWrk9wtdXmpUvcJsg2s6/view?usp=sharing)** |
|----------------------------|-----------------|----------------|----------------|------------------|-----------------|------------------|-----------------|
| **Dataset Size (reviews)** | 535,515         | 739,891        | 462,026        | 171,782          | 393,936         | 689,461          | 229,843         |
| **# Businesses**           | 1,796           | 2,473          | 1,124          | 1,038            | 1,514           | 2,852            | 1,121           |
| **Most Rated Business**    | 3,919           | 5,071          | 7,385          | 1,378            | 3,321           | 9,295            | 2,281           |
| **# Categories**           | 320             | 357            | 283            | 270              | 314             | 375              | 199             |
|                            | Nightlife       | Mexican        | Nightlife      | Nightlife        | Nightlife       | Nightlife        | Coffee          |
| **Top 5**                  | Bars            | Nightlife      | Bars           | Bars             | Bars            | Bars             | Fast Food       |
| **Categories**             | American        | Bars           | Sandwiches     | American         | American        | Sandwiches       | Chinese         |
|                            | Sandwiches      | Sandwiches     | American       | Fast Food        | Sandwiches      | American         | Sandwiches      |
|                            | Fast Food       | Italian        | Italian        | Sandwiches       | Fast Food       | Italian          | Bakeries        |
| **Max Categories**         | 16              | 26             | 17             | 17               | 16              | 18               | 4               |

## One-time installations
run the following line to download the necessary packages:
```
python installations.py
```

## 2. Model Training and Recommendation Performance Results


## 3. Template-based & Attribute-based Bias Analysis
This work leverage a template-based analysis that is popularly used in research work on fairness and bias issues in pretrained language models



## Example labels and Templates
For the bias analysis, we provide the example labels and the templates to generate different test-time input sentences, so that we can analyse the recommendation results accordingly.
All files are located under `data/bias_analysis`:
* `example_labels` contains the labels for different bias types, in the form of dictionaries
* `templates` contains the input sentence templates for yelp

## Test-side input sentence generation
To generate the test-time input sentences, run 
```
python generate_inputSentences.py
```
The generate input sentences will be saved at `<dataset>/input_sentences/<bias_type>.csv`. In this repository, I have uploaded the input sentences files for yelp.

## Output dataframe generation
After getting the test-time input sentences, we can directly make inferences using them. The recommendation results will be gathered under the `output_dataframes` folder.
with the naming convention `<city_name>_output_dataframes_<experiment>.csv`
To generate output dataframes, run:
```
python generate_outputs.py
```
## Generate statistics for the datasets

* `statistics.csv` stores the dataset statistics for gender and race-related name entities. 
* `statistics.txt` is the latex version of the statistics table, which is automatically generated along with the statistics dataframe.
* `gender_statistics.csv` is the dataframe containing the statistics for gender-relation words (e.g., daughter, son, mother, etc.), this is an alternative file to analyse the correlations between dataset
statistics and the recommendation results, since not only names help to indicate gender, but also relationship words.


## Generate bias analysis results and plot figures
After generating the recommendation results and collecting the dataset statistics in the steps above, the bias analysis experiments can be performed by running:
```
python bias_analysis.py --save_figure
```
All figures will be generated for experiment 5f, and saved under the directory `bias_analysis/yelp/figures_5f/`
