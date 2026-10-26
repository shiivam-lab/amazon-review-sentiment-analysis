# Amazon Customer Review Sentiment Analysis & Business Dashboard

## Overview
An interactive Streamlit application that explores customer review sentiment, compares products, surfaces common words in positive and negative feedback, and demonstrates a text-classification workflow using TF-IDF and Logistic Regression.

## Dataset
The included `reviews.csv` contains **1,499 review records** and the columns `Product Name`, `Age`, `Gender`, `Review`, `Country`, and `Sentiment`. The direct `Name` field has been removed from the public-project copy to reduce exposure of personal identifiers.

### Dataset sentiment distribution
| Sentiment | Reviews | Share |
|---|---:|---:|
| Positive | 1,040 | 69.38% |
| Negative | 338 | 22.55% |
| Neutral | 121 | 8.07% |
| **Total** | **1,499** | **100.00%** |

These are descriptive counts from the included CSV, not model accuracy results.

## Tools
- Python
- Pandas
- Streamlit
- Plotly Express
- VADER Sentiment Analysis
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Features
- Dataset summary and review search
- Positive, negative, and neutral sentiment distribution
- Sentiment score distribution
- Product-wise sentiment comparison
- Keyword counts in positive and negative reviews
- Age-group analysis when an `Age` column is present
- Interactive filters for product, gender, and sentiment
- Logistic Regression sentiment classification with a held-out test split and confusion matrix
- Business dashboard with review KPIs and general recommendations

## Run locally
1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run main.py
   ```

The app reads `reviews.csv` from the project folder by default. You can also upload a different CSV using the sidebar. The file must include a `Review` column; `Sentiment` can be generated from review text if it is missing.

## Project files
- `main.py` — Streamlit dashboard and analysis app
- `practice.py` — Standalone VADER sentiment scoring script
- `reviews.csv` — Review dataset used by the project
- `requirements.txt` — Python dependencies

## Notes and limitations
- The sentiment labels in the supplied dataset are used for descriptive analysis. VADER sentiment scores are generated if the score column is absent.
- The ML page reports accuracy for the current train/test split. Do not treat a single split's accuracy as proof of generalization.
- Keyword counts are simple text matches, not a full topic-modeling system.
- Review data can contain personal or sensitive information. Check dataset rights and remove identifiers before publishing. This repository copy omits the `Name` column.

## Future improvements
- Add unit tests and data validation
- Compare VADER labels with human-reviewed ground truth
- Use cross-validation and per-class metrics to assess the ML model
- Improve complaint detection with a more robust text-analysis approach

## Model Performance

A TF-IDF vectorizer and Logistic Regression classifier were evaluated on a held-out test set.

- **Test accuracy:** 74.33%
- **Training reviews:** 1,199
- **Testing reviews:** 300
- **Evaluation method:** Confusion matrix and classification report

The model performed better on positive reviews than on negative and neutral reviews. Improving minority-class classification is a potential area for future work. 

## 📸 Application Screenshots

### 1. Dashboard Overview
<img width="1917" height="1010" alt="Screenshot 2026-10-27 025423" src="https://github.com/user-attachments/assets/d58566ac-53a4-45be-9752-a7b92016b07d" />
<img width="1911" height="873" alt="Screenshot 2026-10-27 025507" src="https://github.com/user-attachments/assets/78009412-254a-4229-88e6-0844968623c7" />
<img width="1917" height="1015" alt="Screenshot 2026-10-27 025524" src="https://github.com/user-attachments/assets/8fca024c-dafd-4c31-bad0-137728aefd05" />
<img width="1916" height="1016" alt="Screenshot 2026-10-27 025537" src="https://github.com/user-attachments/assets/d811c983-b3f7-48b3-86a3-07668bd02138" />
<img width="1917" height="1018" alt="Screenshot 2026-10-27 025612" src="https://github.com/user-attachments/assets/5e96034d-9772-4d88-a3fd-ae7242e024c2" />






