from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" /"raw" / "creditcard.csv")
class_counts = df["Class"].value_counts().sort_index()
bars =plt.bar(["Legitimate","Fraud"], class_counts.values)
plt.bar_label(bars, padding = 3)
plt.title("Fraud vs Legitimate Transactions")

plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.savefig(BASE_DIR / "reports" / "class_distribution.png")
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(df["Amount"], bins= 50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.savefig(BASE_DIR /"reports" /"amount_distribution.png")
print("Histogram saved successfully!")

plt.close()
#transaction amount box plot
plt.figure(figsize =(10,6))
plt.boxplot(df["Amount"])
plt.title("Transaction Amount Box plot")
plt.ylabel("Transaction Amount")
plt.savefig(BASE_DIR /"reports" /"amount_boxplot.png")
plt.show()
