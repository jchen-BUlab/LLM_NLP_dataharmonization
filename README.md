Using large language models (e.g., E5 LLM) to align variables from different studies to support data harmonization. 
Code developed for the Data Harmonization Using Natural Language Processing (NLP harmonization) project and 
the 2025 paper accepted by PLOS One. 

Li Z, Prabhu SP, Popp ZT, Jain SS, Balakundi V, Ang TFA, Au R, Chen J. A natural language processing approach to support biomedical data harmonization: Leveraging large language models. Accepted by PLOS One.


copyright (c) 2025 Zexu li, Jinying Chen, Boston University Chobanian & Avedisian School of Medicine

Licensed under the MIT License (the "License"); you may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

data/: example data for running the single NLP model and the Random Forest model.

code/Single_model_var_matching.ipynb: Python code that uses different individual NLP emthods, including LLM models and the Fuzzy Match method to match variables between two studies: GERAS-EU and GERAS-JP.

code/ML_experiments/: Python code that implements Random Forest model that combines invidual NLP methods to match variables between GERAS-EU and GERAS-JP studies.

code/Sensitivity_analysis.ipynb: Python code that assesses importance of features used by the Random Forest model.
