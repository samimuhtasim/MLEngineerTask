import os
import csv
import joblib
import pdfminer


def categorize_resumes(dir_path):
  """Categorizes the resumes in the specified directory."""

  # Load the trained model.
  model = joblib.load("joblib_model.pkl")

  # Create a list of all the resumes in the directory.
  resumes = os.listdir(dir_path)

  # For each resume, extract the text and classify it.
  for resume in resumes:
    with open(os.path.join(dir_path, resume), "rb") as pdffile:
      text = pdfminer.high_level.extract_text(pdffile)
    category = model.predict([text])[0]
    if not os.path.exists(os.path.join(dir_path, category)):
      os.mkdir(os.path.join(dir_path, category))
    os.rename(os.path.join(dir_path, resume), os.path.join(dir_path, category, resume))

  # Create a CSV file with the categorized resumes.
  with open("categorized_resumes.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["filename", "category"])
    for resume in resumes:
      with open(os.path.join(dir_path, resume), "rb") as pdffile:
        text = pdfminer.high_level.extract_text(pdffile)
      category = model.predict([text])[0]
      writer.writerow([resume, category])


if __name__ == "__main__":
  dir_path = os.path.realpath(os.path.join(os.getcwd(), "resumes"))
  categorize_resumes(dir_path)