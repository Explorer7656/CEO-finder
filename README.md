# CEO-finder
This project can find the name of the CEO/Founder of a company by using company name and 
main-page URL as inputs.

## Features
- Crawls company websites for text content
- Detects mentions of roles like CEO and Founder
- Uses Named Entity Recognition (NER) with spaCy
- Extracts and ranks candidate names
- Saves results (name, role, sentence, email) to CSV

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/ceo-finder.git
cd ceo-finder

pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## Usage



Inputting company name and mainpage url is done in line 17. parameters are in a list with company_url and company_name being str.



The output format is CSV. Output path is inputted inside function "run_pipeline_and_output" in line 201.



## NOTE

This script relies on requests library for crawling websites. requests is not resistant to bot prevention measures and may get a 403 client error.
Take this into account when picking URLs to parse. Works best when root is /about.



