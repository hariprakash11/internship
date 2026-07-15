import pandas as pd

excel = pd.ExcelFile("data/EduPro Online Platform.xlsx")

for sheet in excel.sheet_names:
    print(f"\n===== {sheet} =====")
    df = pd.read_excel(excel, sheet_name=sheet)
    print(df.columns.tolist())
    print(pd.merge.columns.tolist())
 
