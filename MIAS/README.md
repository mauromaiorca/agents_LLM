# PDF Summary Agent for MIAS Papers

A small toolkit to **score relevance** and **extract structured fields** from scientific PDFs
(e.g., *marine invasive alien species* in the Mediterranean) using an LLM + lightweight RAG.
It supports **OpenAI**, **other API providers** (e.g., DeepSeek), and **local LLMs** via **Ollama**.

> **Code files (place alongside this repo):**
> - `code/pdf_relevance_pipeline.py`
> - `code/run_pdf_relevance_pipeline.py`

## Get Ready
You need to have ChatGPT API, unfortunately this might not be free.
Instructions here: [https://platform.openai.com/](https://platform.openai.com/docs/quickstart)
   
## Quick Start

### setup the ChatGPT API

In the directory where you are going to use the code, create a file named .env that will look like:
   ```bash
OPENAI_API_KEY=YOUR-API-KEY
OPENAI_MODEL=gpt-4o-mini
   ```
and replace "YOUR-API-KEY" with your api key retrieved from OpenAI. Also, it should be possible use other LLM, but I didn't test on that. Further, here the gpt-4o-mini is used because it's cheap, see pricing here: [openAI pricing](https://platform.openai.com/docs/pricing), you can also go for a cheaper gpt-5-nano, or slightly more expensive gpt-5-mini.
Assume the directory you are using the code is 

### Install
Download the code (is in the code section), or use the command "git clone https://github.com/mauromaiorca/agents_LLM"
So, go to MIAS directory (cd MIAS), assume your directory is this: "/home/mauro/test/testLLM/agents_LLM/MIAS" and contains a subdirectory "code" with all the code on it.
To don't mess with other installation in your system create an environment, source it, activate, install required software. In short, this:
   ```python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt
   ```
You should now be ready to use the program "./code/run_pdf_relevance_pipeline.py "

If you are using windows or other systems other than linux or mac, you might have a look here [`docs/Installation.md`](docs/Installation.md) for inspiration.


### Run the program
Always be sure that your environment is activated, so go to your home directory (e.g. cd /home/mauro/test/testLLM/agents_LLM/MIAS/), and do 
   ```deactivate
source .venv/bin/activate
   ```
So you are ready to run the program.

Then, be sure you have a directory with the pdf you want to check (e.g. /home/mauro/test/testLLM/agents_LLM/MIAS/documents), for this example I used the documents here: [random MIAS papers](https://drive.google.com/drive/folders/1ApNg5qFHohkLq_uXEiH0Ow-na8fnDgzt?usp=sharing)

And finally run the program:
   ```bash
   ./code/run_pdf_relevance_pipeline.py --pdf-dir /home/mauro/test/testLLM/agents_LLM/MIAS/documents --threshold 60 --max-pages 20
   ```
this will produce a directory and a file that looks : [/home/mauro/test/testLLM/agents_LLM/MIAS/documents/analysis/extractions_from_pdfs.csv](analysis/extractions_from_pdfs.csv)


### Advanced operations
The task to perform for the agents are in the file "marine_valuation.yml". This file is not pefect, and you need to adjust it to what you are interested to. Also, every paper is different, and the algorithm might want to look for specific details in some of the papers.
In that case you can run
```bash
 ./code/profile_refinement_agent.py --csv code/extractions_from_pdfs.csv --instructions refinement_instructions.yml --base-profile code/marine_valuation.yml --o-basename marine_valuation_v2
```
to get a better version of the marine_valuation_v2.yml using the collected data and the table. And per-file csv instructions, marine_valuation_v2.csv with specific per-file instructions on the findings.

You can then call again ./code/run_pdf_relevance_pipeline.py and hopefully you will get a better analysis table: analysis_run2.csv
```bash
./code/run_pdf_relevance_pipeline.py --pdf-dir documents --config marine_valuation_v2.yml --actions-csv marine_valuation_v2.csv --out-csv analysis/analysis_run2.csv
```

this will produce a directory and a file that looks : [/home/mauro/test/testLLM/agents_LLM/MIAS/documents/analysis/analysis_run2.csv](analysis/analysis_run2.csv)
that hopefully has more informative fields.

