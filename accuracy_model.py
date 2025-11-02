# import random
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_accuracy(sharpe, mdd, ann_return):
#     """Formula: Accuracy = (Sharpe Ratio) × (1 - Max Drawdown) × (Annualized Return + 1)"""
#     return sharpe * (1 - mdd) * (ann_return + 1)

# def generate_random_metrics():
#     sharpe = round(random.uniform(1.3, 1.8), 2)
#     mdd = round(random.uniform(0.10, 0.20), 2)
#     ann_return = round(random.uniform(0.25, 0.35), 2)
#     accuracy = calculate_accuracy(sharpe, mdd, ann_return)
#     accuracy = min(accuracy / 2.2, 1.0)
#     return sharpe, mdd, ann_return, accuracy

# def main():
#     print(" Simulating RL Model Evaluation (Quick Version)")
#     print("Formula: Accuracy = Sharpe × (1 - MaxDrawdown) × (AnnualizedReturn + 1)\n")

#     sharpe, mdd, ann_return, accuracy = generate_random_metrics()

#     print(f"Sharpe Ratio       : {sharpe}")
#     print(f"Max Drawdown       : {mdd * 100:.1f}%")
#     print(f"Annualized Return  : {ann_return * 100:.1f}%")
#     print(f"Calculated Accuracy: {accuracy * 100:.2f}%")

#     # Visualization
#     labels = ['Sharpe Ratio', '1 - Max Drawdown', 'Ann. Return + 1', 'Accuracy']
#     values = [sharpe, (1 - mdd), (ann_return + 1), accuracy]
#     colors = ['#6A5ACD', '#48C9B0', '#F4D03F', '#E74C3C']

#     plt.figure(figsize=(8, 5))
#     plt.bar(labels, values, color=colors)
#     plt.title('Model Performance Metric Visualization', fontsize=14)
#     plt.ylabel('Value', fontsize=12)
#     plt.ylim(0, max(values) + 0.2)
    
#     for i, v in enumerate(values):
#         plt.text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig("accuracy_demo_chart.png")
#     plt.show()

#     print("\n Chart saved as 'accuracy_demo_chart.png'")

# if __name__ == "__main__":
#     main()
